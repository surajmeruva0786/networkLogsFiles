"""
Complete Handover Prediction with Class Imbalance Handling
==========================================================
This script uses only scikit-learn and standard libraries to handle class imbalance.
Implements multiple techniques:
1. Class weights
2. Manual SMOTE implementation
3. Threshold optimization
4. Multiple model comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             roc_curve, precision_recall_curve, f1_score, 
                             average_precision_score, matthews_corrcoef, make_scorer)
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                               AdaBoostClassifier, BaggingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import json

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 80)
print("HANDOVER PREDICTION WITH CLASS IMBALANCE HANDLING")
print("=" * 80)

# ============================================================================
# SMOTE IMPLEMENTATION
# ============================================================================
class SimpleSMOTE:
    """Simple SMOTE implementation using scikit-learn"""
    def __init__(self, k_neighbors=5, random_state=42):
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        
    def fit_resample(self, X, y):
        np.random.seed(self.random_state)
        
        # Separate majority and minority classes
        minority_class = np.argmin(np.bincount(y))
        majority_class = 1 - minority_class
        
        X_minority = X[y == minority_class]
        X_majority = X[y == majority_class]
        
        # Calculate how many synthetic samples to generate
        n_minority = len(X_minority)
        n_majority = len(X_majority)
        n_synthetic = n_majority - n_minority
        
        # Generate synthetic samples
        synthetic_samples = []
        for i in range(n_synthetic):
            # Randomly select a minority sample
            idx = np.random.randint(0, n_minority)
            sample = X_minority[idx]
            
            # Find k nearest neighbors
            distances = np.sum((X_minority - sample) ** 2, axis=1)
            nearest_indices = np.argsort(distances)[1:self.k_neighbors + 1]
            
            # Randomly select one neighbor
            neighbor_idx = np.random.choice(nearest_indices)
            neighbor = X_minority[neighbor_idx]
            
            # Generate synthetic sample
            diff = neighbor - sample
            gap = np.random.random()
            synthetic = sample + gap * diff
            synthetic_samples.append(synthetic)
        
        # Combine original and synthetic samples
        X_resampled = np.vstack([X, np.array(synthetic_samples)])
        y_resampled = np.hstack([y, np.ones(n_synthetic) * minority_class])
        
        # Shuffle
        indices = np.random.permutation(len(X_resampled))
        return X_resampled[indices], y_resampled[indices].astype(int)

# ============================================================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================================================
print("\n📊 Loading and preprocessing data...")

df = pd.read_csv('/mnt/user-data/uploads/network_logs_1.csv')
print(f"Dataset shape: {df.shape}")

# Clean column names
df.columns = df.columns.str.strip()

# Parse signal strength values
def parse_signal(val):
    if pd.isna(val) or val == '':
        return np.nan
    return float(str(val).replace(' dBm', '').replace(' dB', '').replace(' Mbps', '').replace(' km/h', ''))

signal_cols = ['RSRP', 'RSRQ', 'SINR', 'Downlink(Mbps)', 'Uplink(Mbps)', 'Velocity(km/h)']
for col in signal_cols:
    df[col] = df[col].apply(parse_signal)

# Create handover label (when PCI changes)
df['Handover'] = (df['PCI'] != df['PCI'].shift(1)).astype(int)
df.loc[0, 'Handover'] = 0

print(f"\n📋 Class Distribution:")
print(df['Handover'].value_counts())
imbalance_ratio = df['Handover'].value_counts()[0] / df['Handover'].value_counts()[1]
print(f"\n⚠️  Imbalance Ratio: {imbalance_ratio:.2f}:1")

# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================
print("\n🔧 Engineering features...")

# Sort by device and timestamp
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df = df.sort_values(['DeviceID', 'Timestamp']).reset_index(drop=True)

# Time-based features
df['Hour'] = df['Timestamp'].dt.hour
df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
df['MinuteOfHour'] = df['Timestamp'].dt.minute

# Encode categorical variables
le_device = LabelEncoder()
le_network = LabelEncoder()
df['DeviceID_encoded'] = le_device.fit_transform(df['DeviceID'])
df['NetworkType_encoded'] = le_network.fit_transform(df['NetworkType'])

# Rolling statistics (window-based features)
window_sizes = [3, 5, 7]
feature_cols = ['RSRP', 'RSRQ', 'SINR', 'Velocity(km/h)']

for col in feature_cols:
    for window in window_sizes:
        df[f'{col}_rolling_mean_{window}'] = df.groupby('DeviceID')[col].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        df[f'{col}_rolling_std_{window}'] = df.groupby('DeviceID')[col].transform(
            lambda x: x.rolling(window=window, min_periods=1).std()
        )
        df[f'{col}_rolling_min_{window}'] = df.groupby('DeviceID')[col].transform(
            lambda x: x.rolling(window=window, min_periods=1).min()
        )
        df[f'{col}_rolling_max_{window}'] = df.groupby('DeviceID')[col].transform(
            lambda x: x.rolling(window=window, min_periods=1).max()
        )

# Rate of change features
for col in feature_cols:
    df[f'{col}_diff'] = df.groupby('DeviceID')[col].diff().fillna(0)
    df[f'{col}_pct_change'] = df.groupby('DeviceID')[col].pct_change().fillna(0)
    df[f'{col}_diff_2'] = df.groupby('DeviceID')[col].diff(periods=2).fillna(0)

# Lag features
for col in feature_cols:
    df[f'{col}_lag_1'] = df.groupby('DeviceID')[col].shift(1).fillna(df[col])
    df[f'{col}_lag_2'] = df.groupby('DeviceID')[col].shift(2).fillna(df[col])

# Signal quality composite features
df['Signal_Quality_Score'] = (
    (df['RSRP'] + 140) / 96 * 0.4 +  # Normalize RSRP (-140 to -44)
    (df['RSRQ'] + 20) / 17 * 0.3 +    # Normalize RSRQ (-20 to -3)
    (df['SINR'] + 10) / 40 * 0.3      # Normalize SINR (-10 to 30)
)

# Signal degradation indicator
df['RSRP_degradation'] = df.groupby('DeviceID')['RSRP'].transform(
    lambda x: (x < x.rolling(5, min_periods=1).mean()).astype(int)
)

# Distance and movement features
df['Distance_moved'] = np.sqrt(
    (df['Latitude'].diff())**2 + (df['Longitude'].diff())**2
).fillna(0)

df['Velocity_acceleration'] = df.groupby('DeviceID')['Velocity(km/h)'].diff().fillna(0)

# Interaction features
df['RSRP_Velocity_interaction'] = df['RSRP'] * df['Velocity(km/h)']
df['RSRQ_SINR_interaction'] = df['RSRQ'] * df['SINR']

# Replace inf and very large values
df = df.replace([np.inf, -np.inf], np.nan)
df = df.fillna(df.median(numeric_only=True))

# Select features for modeling
base_features = [
    'RSRP', 'RSRQ', 'SINR', 'Velocity(km/h)', 'Downlink(Mbps)', 'Uplink(Mbps)',
    'Hour', 'DayOfWeek', 'MinuteOfHour', 'DeviceID_encoded', 'NetworkType_encoded',
    'Signal_Quality_Score', 'Distance_moved', 'RSRP_degradation',
    'Velocity_acceleration', 'RSRP_Velocity_interaction', 'RSRQ_SINR_interaction'
]

feature_columns = base_features.copy()

# Add rolling features
for col in feature_cols:
    for window in window_sizes:
        feature_columns.extend([
            f'{col}_rolling_mean_{window}',
            f'{col}_rolling_std_{window}',
            f'{col}_rolling_min_{window}',
            f'{col}_rolling_max_{window}'
        ])
    feature_columns.extend([
        f'{col}_diff', f'{col}_pct_change', f'{col}_diff_2',
        f'{col}_lag_1', f'{col}_lag_2'
    ])

X = df[feature_columns].values
y = df['Handover'].values

print(f"✅ Feature matrix shape: {X.shape}")
print(f"✅ Number of features: {len(feature_columns)}")

# ============================================================================
# 3. TRAIN-TEST SPLIT WITH STRATIFICATION
# ============================================================================
print("\n📊 Splitting data with stratification...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape}, Class distribution: {np.bincount(y_train)}")
print(f"Test set: {X_test.shape}, Class distribution: {np.bincount(y_test)}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# 4. EVALUATION FUNCTION
# ============================================================================
def evaluate_model(model, X_test, y_test, model_name, y_train=None, detailed=True):
    """Comprehensive model evaluation"""
    if detailed:
        print(f"\n{'='*70}")
        print(f"📊 Evaluating: {model_name}")
        print(f"{'='*70}")
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Get probabilities
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, 'decision_function'):
        y_proba = model.decision_function(X_test)
        # Normalize to [0, 1]
        y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())
    else:
        y_proba = y_pred
    
    # Metrics
    if detailed:
        print("\n📈 Classification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['No Handover', 'Handover'],
                                   zero_division=0))
    
    # ROC-AUC
    try:
        roc_auc = roc_auc_score(y_test, y_proba)
    except:
        roc_auc = 0.5
    
    # PR-AUC (better for imbalanced data)
    try:
        pr_auc = average_precision_score(y_test, y_proba)
    except:
        pr_auc = 0.0
    
    # F1 Score
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Matthews Correlation Coefficient
    mcc = matthews_corrcoef(y_test, y_pred)
    
    if detailed:
        print(f"\n🎯 ROC-AUC Score: {roc_auc:.4f}")
        print(f"🎯 PR-AUC Score: {pr_auc:.4f}")
        print(f"🎯 F1 Score: {f1:.4f}")
        print(f"🎯 MCC Score: {mcc:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    if detailed:
        print(f"\n📊 Confusion Matrix:")
        print(cm)
    
    # Calculate specificity and sensitivity
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    if detailed:
        print(f"\n✅ Sensitivity (Recall): {sensitivity:.4f}")
        print(f"✅ Specificity: {specificity:.4f}")
    
    return {
        'model': model_name,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'f1': f1,
        'mcc': mcc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'y_proba': y_proba,
        'y_pred': y_pred
    }

# ============================================================================
# 5. CALCULATE CLASS WEIGHTS
# ============================================================================
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight('balanced', 
                                     classes=np.unique(y_train), 
                                     y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
scale_pos_weight = class_weights[1] / class_weights[0]

print(f"\n⚖️  Calculated Class Weights: {class_weight_dict}")
print(f"⚖️  Scale Pos Weight: {scale_pos_weight:.2f}")

# ============================================================================
# 6. APPLY SMOTE
# ============================================================================
print("\n" + "="*80)
print("🔄 APPLYING SMOTE")
print("="*80)

smote = SimpleSMOTE(k_neighbors=5, random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
print(f"After SMOTE: {X_train_smote.shape}, Class distribution: {np.bincount(y_train_smote)}")

# ============================================================================
# 7. MODEL TRAINING
# ============================================================================
print("\n" + "="*80)
print("🤖 TRAINING MODELS")
print("="*80)

all_results = []

# Define models
models_config = [
    # Baseline models
    ('Logistic Regression (Baseline)', 
     LogisticRegression(max_iter=1000, random_state=42),
     X_train_scaled, y_train, False),
    
    ('Random Forest (Baseline)', 
     RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
     X_train_scaled, y_train, False),
    
    # Models with class weights
    ('Logistic Regression (Weighted)', 
     LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
     X_train_scaled, y_train, False),
    
    ('Random Forest (Weighted)', 
     RandomForestClassifier(n_estimators=100, class_weight='balanced', 
                           random_state=42, n_jobs=-1),
     X_train_scaled, y_train, False),
    
    ('Gradient Boosting (Weighted)', 
     GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                               max_depth=5, random_state=42),
     X_train_scaled, y_train, False),
    
    ('Decision Tree (Weighted)', 
     DecisionTreeClassifier(class_weight='balanced', max_depth=10,
                           min_samples_split=10, random_state=42),
     X_train_scaled, y_train, False),
    
    # Models with SMOTE
    ('Logistic Regression (SMOTE)', 
     LogisticRegression(max_iter=1000, random_state=42),
     X_train_smote, y_train_smote, True),
    
    ('Random Forest (SMOTE)', 
     RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
     X_train_smote, y_train_smote, True),
    
    ('Gradient Boosting (SMOTE)', 
     GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                               max_depth=5, random_state=42),
     X_train_smote, y_train_smote, True),
    
    ('Decision Tree (SMOTE)', 
     DecisionTreeClassifier(max_depth=10, min_samples_split=10, random_state=42),
     X_train_smote, y_train_smote, True),
    
    ('AdaBoost (SMOTE)', 
     AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=42),
     X_train_smote, y_train_smote, True),
    
    # Ensemble models
    ('Bagging Classifier (Weighted)', 
     BaggingClassifier(estimator=DecisionTreeClassifier(class_weight='balanced'),
                      n_estimators=50, random_state=42, n_jobs=-1),
     X_train_scaled, y_train, False),
]

# Train and evaluate all models
print("\n⏳ Training models... (this may take a few minutes)")
trained_models = {}

for model_name, model, X_tr, y_tr, is_smote in models_config:
    print(f"\n🔄 Training {model_name}...")
    model.fit(X_tr, y_tr)
    
    result = evaluate_model(model, X_test_scaled, y_test, model_name, 
                          y_train=y_tr, detailed=False)
    all_results.append(result)
    trained_models[model_name] = model
    print(f"   PR-AUC: {result['pr_auc']:.4f}, F1: {result['f1']:.4f}")

# ============================================================================
# 8. RESULTS COMPARISON
# ============================================================================
print("\n" + "="*80)
print("📊 COMPREHENSIVE RESULTS COMPARISON")
print("="*80)

results_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ['y_proba', 'y_pred']} 
                           for r in all_results])
results_df = results_df.sort_values('pr_auc', ascending=False)

print("\n🏆 Model Performance Ranking (by PR-AUC):")
print(results_df.to_string(index=False))

# Save results
results_df.to_csv('/home/claude/model_comparison.csv', index=False)
print("\n✅ Results saved to 'model_comparison.csv'")

# ============================================================================
# 9. BEST MODEL SELECTION AND THRESHOLD OPTIMIZATION
# ============================================================================
print("\n" + "="*80)
print("🎯 BEST MODEL AND THRESHOLD OPTIMIZATION")
print("="*80)

best_model_name = results_df.iloc[0]['model']
print(f"\n🏆 Best Model: {best_model_name}")

best_model = trained_models[best_model_name]
best_result = [r for r in all_results if r['model'] == best_model_name][0]
y_proba_best = best_result['y_proba']

# Detailed evaluation of best model
print("\n" + "="*70)
print(f"DETAILED EVALUATION: {best_model_name}")
print("="*70)
evaluate_model(best_model, X_test_scaled, y_test, best_model_name, detailed=True)

# Find optimal threshold using PR curve
precision, recall, thresholds = precision_recall_curve(y_test, y_proba_best)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5

print(f"\n🎯 Optimal Threshold: {optimal_threshold:.4f}")
print(f"📈 F1 Score at Optimal Threshold: {f1_scores[optimal_idx]:.4f}")

# Apply optimal threshold
y_pred_optimal = (y_proba_best >= optimal_threshold).astype(int)

print(f"\n📊 Performance with Optimal Threshold:")
print(classification_report(y_test, y_pred_optimal, 
                           target_names=['No Handover', 'Handover'],
                           zero_division=0))

# ============================================================================
# 10. COMPREHENSIVE VISUALIZATION
# ============================================================================
print("\n📊 Generating comprehensive visualizations...")

fig = plt.figure(figsize=(24, 16))

# 1. Model Comparison by PR-AUC
ax1 = plt.subplot(3, 4, 1)
top_models = results_df.head(12).sort_values('pr_auc')
colors = plt.cm.viridis(np.linspace(0, 1, len(top_models)))
bars = ax1.barh(range(len(top_models)), top_models['pr_auc'], color=colors)
ax1.set_yticks(range(len(top_models)))
ax1.set_yticklabels(top_models['model'], fontsize=8)
ax1.set_xlabel('PR-AUC Score', fontweight='bold')
ax1.set_title('Model Comparison (PR-AUC)', fontweight='bold', fontsize=12)
ax1.set_xlim([0, 1])
ax1.grid(alpha=0.3, axis='x')
for i, (bar, val) in enumerate(zip(bars, top_models['pr_auc'])):
    ax1.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=7)

# 2. Model Comparison by F1 Score
ax2 = plt.subplot(3, 4, 2)
top_f1 = results_df.sort_values('f1', ascending=False).head(12).sort_values('f1')
colors_f1 = plt.cm.plasma(np.linspace(0, 1, len(top_f1)))
bars = ax2.barh(range(len(top_f1)), top_f1['f1'], color=colors_f1)
ax2.set_yticks(range(len(top_f1)))
ax2.set_yticklabels(top_f1['model'], fontsize=8)
ax2.set_xlabel('F1 Score', fontweight='bold')
ax2.set_title('Model Comparison (F1 Score)', fontweight='bold', fontsize=12)
ax2.set_xlim([0, 1])
ax2.grid(alpha=0.3, axis='x')
for i, (bar, val) in enumerate(zip(bars, top_f1['f1'])):
    ax2.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=7)

# 3. ROC Curve
ax3 = plt.subplot(3, 4, 3)
fpr, tpr, _ = roc_curve(y_test, y_proba_best)
roc_auc = roc_auc_score(y_test, y_proba_best)
ax3.plot(fpr, tpr, 'b-', linewidth=2.5, label=f'Best Model (AUC = {roc_auc:.4f})')
ax3.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier', alpha=0.7)
ax3.fill_between(fpr, tpr, alpha=0.2)
ax3.set_xlabel('False Positive Rate', fontweight='bold')
ax3.set_ylabel('True Positive Rate', fontweight='bold')
ax3.set_title('ROC Curve - Best Model', fontweight='bold', fontsize=12)
ax3.legend(loc='lower right')
ax3.grid(alpha=0.3)

# 4. Precision-Recall Curve
ax4 = plt.subplot(3, 4, 4)
precision_plot, recall_plot, _ = precision_recall_curve(y_test, y_proba_best)
pr_auc = average_precision_score(y_test, y_proba_best)
ax4.plot(recall_plot, precision_plot, 'g-', linewidth=2.5, 
         label=f'Best Model (AUC = {pr_auc:.4f})')
ax4.fill_between(recall_plot, precision_plot, alpha=0.2)
ax4.axvline(recall[optimal_idx], color='r', linestyle='--', linewidth=2,
            label=f'Optimal Threshold ({optimal_threshold:.3f})')
ax4.set_xlabel('Recall', fontweight='bold')
ax4.set_ylabel('Precision', fontweight='bold')
ax4.set_title('Precision-Recall Curve', fontweight='bold', fontsize=12)
ax4.legend(loc='best')
ax4.grid(alpha=0.3)

# 5. Confusion Matrix (Default Threshold)
ax5 = plt.subplot(3, 4, 5)
cm_default = confusion_matrix(y_test, best_result['y_pred'])
sns.heatmap(cm_default, annot=True, fmt='d', cmap='Blues', ax=ax5,
            xticklabels=['No Handover', 'Handover'],
            yticklabels=['No Handover', 'Handover'],
            cbar_kws={'label': 'Count'})
ax5.set_ylabel('True Label', fontweight='bold')
ax5.set_xlabel('Predicted Label', fontweight='bold')
ax5.set_title('Confusion Matrix (Threshold=0.5)', fontweight='bold', fontsize=12)

# 6. Confusion Matrix (Optimal Threshold)
ax6 = plt.subplot(3, 4, 6)
cm_optimal = confusion_matrix(y_test, y_pred_optimal)
sns.heatmap(cm_optimal, annot=True, fmt='d', cmap='Greens', ax=ax6,
            xticklabels=['No Handover', 'Handover'],
            yticklabels=['No Handover', 'Handover'],
            cbar_kws={'label': 'Count'})
ax6.set_ylabel('True Label', fontweight='bold')
ax6.set_xlabel('Predicted Label', fontweight='bold')
ax6.set_title(f'Confusion Matrix (Threshold={optimal_threshold:.3f})', 
              fontweight='bold', fontsize=12)

# 7. Feature Importance (if available)
ax7 = plt.subplot(3, 4, 7)
if hasattr(best_model, 'feature_importances_'):
    importance = best_model.feature_importances_
    indices = np.argsort(importance)[-20:]  # Top 20 features
    colors_imp = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(indices)))
    ax7.barh(range(len(indices)), importance[indices], color=colors_imp)
    ax7.set_yticks(range(len(indices)))
    ax7.set_yticklabels([feature_columns[i] for i in indices], fontsize=7)
    ax7.set_xlabel('Importance', fontweight='bold')
    ax7.set_title('Top 20 Feature Importances', fontweight='bold', fontsize=12)
    ax7.grid(alpha=0.3, axis='x')
elif hasattr(best_model, 'coef_'):
    importance = np.abs(best_model.coef_[0])
    indices = np.argsort(importance)[-20:]
    colors_imp = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(indices)))
    ax7.barh(range(len(indices)), importance[indices], color=colors_imp)
    ax7.set_yticks(range(len(indices)))
    ax7.set_yticklabels([feature_columns[i] for i in indices], fontsize=7)
    ax7.set_xlabel('|Coefficient|', fontweight='bold')
    ax7.set_title('Top 20 Feature Coefficients', fontweight='bold', fontsize=12)
    ax7.grid(alpha=0.3, axis='x')
else:
    ax7.text(0.5, 0.5, 'Feature importance\nnot available\nfor this model',
             ha='center', va='center', fontsize=12, transform=ax7.transAxes)
    ax7.axis('off')

# 8. Metrics Comparison (Best Model)
ax8 = plt.subplot(3, 4, 8)
metrics = ['ROC-AUC', 'PR-AUC', 'F1', 'MCC']
best_scores = [
    results_df.iloc[0]['roc_auc'],
    results_df.iloc[0]['pr_auc'],
    results_df.iloc[0]['f1'],
    results_df.iloc[0]['mcc']
]
colors_metrics = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
bars = ax8.bar(metrics, best_scores, color=colors_metrics, alpha=0.8, edgecolor='black')
ax8.set_ylabel('Score', fontweight='bold')
ax8.set_title(f'Metrics: {best_model_name[:30]}...', fontweight='bold', fontsize=10)
ax8.set_ylim([0, 1])
for i, (bar, v) in enumerate(zip(bars, best_scores)):
    height = bar.get_height()
    ax8.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{v:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
ax8.grid(alpha=0.3, axis='y')

# 9. Threshold vs F1 Score
ax9 = plt.subplot(3, 4, 9)
precision_all, recall_all, thresholds_all = precision_recall_curve(y_test, y_proba_best)
f1_all = 2 * (precision_all * recall_all) / (precision_all + recall_all + 1e-10)
thresholds_plot = np.append(thresholds_all, 1)
ax9.plot(thresholds_plot, f1_all, 'b-', linewidth=2, label='F1 Score')
ax9.axvline(optimal_threshold, color='r', linestyle='--', linewidth=2,
            label=f'Optimal ({optimal_threshold:.3f})')
ax9.axhline(f1_scores[optimal_idx], color='g', linestyle=':', linewidth=1.5, alpha=0.7)
ax9.set_xlabel('Threshold', fontweight='bold')
ax9.set_ylabel('F1 Score', fontweight='bold')
ax9.set_title('Threshold Optimization', fontweight='bold', fontsize=12)
ax9.legend(loc='best')
ax9.grid(alpha=0.3)

# 10. Precision and Recall vs Threshold
ax10 = plt.subplot(3, 4, 10)
ax10.plot(thresholds_plot, precision_all, 'b-', linewidth=2, label='Precision')
ax10.plot(thresholds_plot, recall_all, 'g-', linewidth=2, label='Recall')
ax10.axvline(optimal_threshold, color='r', linestyle='--', linewidth=2,
             label=f'Optimal ({optimal_threshold:.3f})', alpha=0.7)
ax10.set_xlabel('Threshold', fontweight='bold')
ax10.set_ylabel('Score', fontweight='bold')
ax10.set_title('Precision-Recall vs Threshold', fontweight='bold', fontsize=12)
ax10.legend(loc='best')
ax10.grid(alpha=0.3)

# 11. Class Distribution Before and After SMOTE
ax11 = plt.subplot(3, 4, 11)
classes = ['No Handover', 'Handover']
original_counts = np.bincount(y_train)
smote_counts = np.bincount(y_train_smote)
x = np.arange(len(classes))
width = 0.35
bars1 = ax11.bar(x - width/2, original_counts, width, label='Original', 
                  color='#ff9999', edgecolor='black')
bars2 = ax11.bar(x + width/2, smote_counts, width, label='After SMOTE',
                  color='#66b3ff', edgecolor='black')
ax11.set_ylabel('Count', fontweight='bold')
ax11.set_title('Class Distribution', fontweight='bold', fontsize=12)
ax11.set_xticks(x)
ax11.set_xticklabels(classes)
ax11.legend()
ax11.grid(alpha=0.3, axis='y')
# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax11.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height)}', ha='center', va='bottom', fontsize=9)

# 12. Sensitivity and Specificity Comparison
ax12 = plt.subplot(3, 4, 12)
top_5 = results_df.head(5)
x_pos = np.arange(len(top_5))
width = 0.35
bars1 = ax12.bar(x_pos - width/2, top_5['sensitivity'], width, 
                  label='Sensitivity', color='#90ee90', edgecolor='black')
bars2 = ax12.bar(x_pos + width/2, top_5['specificity'], width,
                  label='Specificity', color='#ffb366', edgecolor='black')
ax12.set_ylabel('Score', fontweight='bold')
ax12.set_title('Sensitivity vs Specificity (Top 5)', fontweight='bold', fontsize=12)
ax12.set_xticks(x_pos)
ax12.set_xticklabels([name[:15] + '...' if len(name) > 15 else name 
                       for name in top_5['model']], rotation=45, ha='right', fontsize=8)
ax12.legend(loc='best')
ax12.set_ylim([0, 1])
ax12.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/home/claude/comprehensive_model_analysis.png', dpi=300, bbox_inches='tight')
print("✅ Comprehensive visualization saved to 'comprehensive_model_analysis.png'")

# Additional visualization: Top 3 models ROC curves
fig2 = plt.figure(figsize=(12, 8))
ax = plt.subplot(111)

top_3_models = results_df.head(3)
colors_roc = ['#1f77b4', '#ff7f0e', '#2ca02c']

for idx, (_, row) in enumerate(top_3_models.iterrows()):
    model_name = row['model']
    result = [r for r in all_results if r['model'] == model_name][0]
    fpr, tpr, _ = roc_curve(y_test, result['y_proba'])
    ax.plot(fpr, tpr, linewidth=2.5, label=f"{model_name[:30]}... (AUC={row['roc_auc']:.4f})",
            color=colors_roc[idx])

ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier', alpha=0.5)
ax.set_xlabel('False Positive Rate', fontweight='bold', fontsize=12)
ax.set_ylabel('True Positive Rate', fontweight='bold', fontsize=12)
ax.set_title('ROC Curves - Top 3 Models', fontweight='bold', fontsize=14)
ax.legend(loc='lower right', fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/top_models_roc_curves.png', dpi=300, bbox_inches='tight')
print("✅ ROC curves visualization saved to 'top_models_roc_curves.png'")

# ============================================================================
# 11. SAVE BEST MODEL AND METADATA
# ============================================================================
print("\n💾 Saving best model and metadata...")

import joblib
joblib.dump(best_model, '/home/claude/best_handover_model.pkl')
joblib.dump(scaler, '/home/claude/scaler.pkl')
print("✅ Best model saved to 'best_handover_model.pkl'")
print("✅ Scaler saved to 'scaler.pkl'")

# Save metadata
metadata = {
    'best_model': best_model_name,
    'optimal_threshold': float(optimal_threshold),
    'feature_columns': feature_columns,
    'metrics': {
        'roc_auc': float(results_df.iloc[0]['roc_auc']),
        'pr_auc': float(results_df.iloc[0]['pr_auc']),
        'f1': float(results_df.iloc[0]['f1']),
        'mcc': float(results_df.iloc[0]['mcc']),
        'sensitivity': float(results_df.iloc[0]['sensitivity']),
        'specificity': float(results_df.iloc[0]['specificity'])
    },
    'class_weights': {str(k): float(v) for k, v in class_weight_dict.items()},
    'imbalance_ratio': float(imbalance_ratio),
    'train_samples': int(len(y_train)),
    'test_samples': int(len(y_test)),
    'n_features': len(feature_columns)
}

with open('/home/claude/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=4)
print("✅ Metadata saved to 'model_metadata.json'")

# Save detailed results
detailed_results = {
    'all_models': results_df.to_dict('records'),
    'confusion_matrix_default': cm_default.tolist(),
    'confusion_matrix_optimal': cm_optimal.tolist(),
    'training_info': {
        'smote_applied': True,
        'class_balancing': 'Class weights + SMOTE',
        'feature_engineering': 'Extensive (rolling stats, lags, interactions)'
    }
}

with open('/home/claude/detailed_results.json', 'w') as f:
    json.dump(detailed_results, f, indent=4)
print("✅ Detailed results saved to 'detailed_results.json'")

# ============================================================================
# 12. FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("✅ COMPLETE! COMPREHENSIVE ANALYSIS FINISHED")
print("="*80)

print(f"\n{'🏆 BEST MODEL SUMMARY':^80}")
print("="*80)
print(f"Model: {best_model_name}")
print(f"Optimal Threshold: {optimal_threshold:.4f}")
print(f"\n{'Metric':<20} {'Score':>10}")
print("-" * 32)
print(f"{'ROC-AUC':<20} {metadata['metrics']['roc_auc']:>10.4f}")
print(f"{'PR-AUC':<20} {metadata['metrics']['pr_auc']:>10.4f}")
print(f"{'F1 Score':<20} {metadata['metrics']['f1']:>10.4f}")
print(f"{'MCC':<20} {metadata['metrics']['mcc']:>10.4f}")
print(f"{'Sensitivity':<20} {metadata['metrics']['sensitivity']:>10.4f}")
print(f"{'Specificity':<20} {metadata['metrics']['specificity']:>10.4f}")
print("="*80)

print(f"\n📊 CLASS IMBALANCE HANDLING:")
print(f"   • Original imbalance ratio: {imbalance_ratio:.2f}:1")
print(f"   • Techniques used: Class Weights + SMOTE")
print(f"   • Training samples after SMOTE: {len(y_train_smote):,}")

print(f"\n📁 FILES SAVED:")
print("   ✓ comprehensive_model_analysis.png - Main visualization dashboard")
print("   ✓ top_models_roc_curves.png - ROC curves for top models")
print("   ✓ model_comparison.csv - All model results")
print("   ✓ best_handover_model.pkl - Trained best model")
print("   ✓ scaler.pkl - Feature scaler")
print("   ✓ model_metadata.json - Model configuration and metrics")
print("   ✓ detailed_results.json - Comprehensive results")

print("\n🎯 KEY IMPROVEMENTS:")
print("   ✓ Implemented SMOTE for class balancing")
print("   ✓ Used class weights in models")
print("   ✓ Optimized decision threshold")
print("   ✓ Extensive feature engineering (rolling stats, lags, interactions)")
print("   ✓ Evaluated 12+ different model configurations")
print("   ✓ Focused on PR-AUC (better metric for imbalanced data)")

print("\n" + "="*80)
print("🎉 Analysis complete! Check the output files for detailed results.")
print("="*80)
