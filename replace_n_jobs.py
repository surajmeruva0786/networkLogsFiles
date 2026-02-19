
file_path = 'complete_handover_prediction.py'
try:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    new_content = content.replace('n_jobs=-1', 'n_jobs=1')
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
        
    print("Successfully replaced n_jobs=-1 with n_jobs=1")
    
except Exception as e:
    print(f"Error: {e}")
