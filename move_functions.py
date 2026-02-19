
import re

file_path = 'complete_handover_prediction.py'

def extract_function(lines, func_name):
    start_idx = -1
    end_idx = -1
    # Look for indented definition: "    def func_name"
    pattern = f"    def {func_name}\("
    
    for i, line in enumerate(lines):
        if re.match(pattern, line):
            start_idx = i
            break
            
    if start_idx == -1:
        return None, None, None
        
    # Find end of function (next dedent or start of next function)
    # Assumes valid python indentation (4 spaces)
    # The function body is indented by 8 spaces
    # So we look for line starting with 4 spaces or less, but NOT empty lines
    for i in range(start_idx + 1, len(lines)):
        line = lines[i]
        if line.strip() == "":
            continue # Skip empty lines
        
        # Check indentation level (count leading spaces)
        indent = len(line) - len(line.lstrip())
        if indent <= 4:
            end_idx = i
            break
            
    if end_idx == -1:
        end_idx = len(lines)
        
    func_lines = lines[start_idx:end_idx]
    # Un-indent (remove first 4 spaces)
    unindented_lines = [line[4:] if line.startswith("    ") else line for line in func_lines]
    
    return start_idx, end_idx, unindented_lines

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    # Functions to move
    funcs_to_move = ['parse_signal', 'objective_xgb', 'objective_lgb', 'objective_cat']
    
    blocks = []
    
    # We must extract from bottom to top to preserve indices?
    # No, extraction creates gaps.
    # Better: just set the lines to empty strings or special marker, then rebuild.
    
    for func in funcs_to_move:
        start, end, block = extract_function(lines, func)
        if start is not None:
            print(f"Found {func} at {start}-{end}")
            blocks.extend(block)
            blocks.append("\n") # Add spacing
            
            # Mark lines for deletion
            for i in range(start, end):
                lines[i] = "##DELETED##\n"
        else:
            print(f"Warning: Could not find {func}")
            
    # Rebuild file
    # Insert blocks before "if __name__"
    # Find insertion point
    insert_idx = -1
    for i, line in enumerate(lines):
        if 'if __name__ == "__main__":' in line:
            insert_idx = i
            break
            
    if insert_idx == -1:
        print("Could not find insertion point")
        exit(1)
        
    final_lines = []
    # 1. Header (up to insertion point)
    for i in range(insert_idx):
        if lines[i] != "##DELETED##\n":
            final_lines.append(lines[i])
            
    # 2. Insert extracted functions
    final_lines.extend(blocks)
    
    # 3. Main block
    for i in range(insert_idx, len(lines)):
        if lines[i] != "##DELETED##\n":
            final_lines.append(lines[i])
            
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(final_lines)
        
    print("Successfully moved functions.")
    
except Exception as e:
    print(f"Error: {e}")
