import textwrap
import re
import sys

filepath = sys.argv[1]
with open(filepath, 'r') as f:
    lines = f.readlines()

new_lines = []
in_env = False
env_skip = ['tabular', 'equation', 'algorithm', 'algorithmic', 'figure', 'table', 'table*', 'figure*', 'thebibliography', 'itemize', 'enumerate']

for line in lines:
    stripped = line.strip()
    
    begin_match = re.search(r'\\begin\{([^}]+)\}', stripped)
    end_match = re.search(r'\\end\{([^}]+)\}', stripped)
    
    if begin_match and begin_match.group(1) in env_skip:
        in_env = True
    
    is_prose = not stripped.startswith('\\') or stripped.startswith('\\textbf') or stripped.startswith('\\textit')
    
    if not in_env and len(stripped) > 100 and is_prose:
        wrapped = textwrap.fill(stripped, width=100)
        new_lines.append(wrapped + '\n')
    else:
        new_lines.append(line)
        
    if end_match and end_match.group(1) in env_skip:
        in_env = False

with open(filepath, 'w') as f:
    f.writelines(new_lines)
print(f"Formatting complete for {filepath}.")
