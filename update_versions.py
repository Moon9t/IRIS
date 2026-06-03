import os
import glob
import re

def replace_in_file(path, pattern, replacement):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    new_content = re.sub(pattern, replacement, content)
    if content != new_content:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f'Updated {path}')

for root, _, files in os.walk('.'):
    if 'target' in root or '.git' in root or 'node_modules' in root:
        continue
    for file in files:
        if file.endswith(('.md', '.sh', '.ps1', '.json', '.iss')):
            path = os.path.join(root, file)
            # Update specific versions to 0.6.1
            replace_in_file(path, r'(?<!\d)0\.5\.0(?!\d)', '0.6.1')
            replace_in_file(path, r'(?<!\d)0\.6\.0(?!\d)', '0.6.1')
