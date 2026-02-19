
import re

file_path = 'complete_handover_prediction.py'
try:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace n_jobs=-1
    content = content.replace('n_jobs=-1', 'n_jobs=1')
    
    # Replace 'n_jobs': -1
    content = content.replace("'n_jobs': -1", "'n_jobs': 1")
    
    # Replace 'n_jobs':-1
    content = content.replace("'n_jobs':-1", "'n_jobs': 1")
    
    # Regex just in case
    # content = re.sub(r'n_jobs\s*[:=]\s*-1', 'n_jobs=1', content) 
    # But be careful with syntax.
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
        
    print("Successfully replaced n_jobs=-1 variations")
    
except Exception as e:
    print(f"Error: {e}")
