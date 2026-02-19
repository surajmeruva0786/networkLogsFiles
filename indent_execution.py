import os

file_path = 'complete_handover_prediction.py'
try:
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Locate the start of main execution
    # Looking for print("=" * 90) which dominates the start
    split_idx = -1
    for i, line in enumerate(lines):
        if 'print("=" * 90)' in line:
            split_idx = i
            break
    
    if split_idx == -1:
        print("Could not find split point")
        exit(1)

    # Check if already indented/wrapped (idempotency)
    if 'if __name__ == "__main__":' in lines[split_idx-1]:
        print("Already wrapped")
        exit(0)

    header = lines[:split_idx]
    # Ensure there's a newline before the guard
    if not header[-1].strip() == '':
        header.append('\n')
        
    guard = ['if __name__ == "__main__":\n']
    
    execution = lines[split_idx:]
    indented_execution = ['    ' + line for line in execution]
    
    new_content = header + guard + indented_execution
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_content)
        
    print("Successfully wrapped in if __name__ == '__main__':")

except Exception as e:
    print(f"Error: {e}")
