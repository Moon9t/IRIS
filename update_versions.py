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
    # Exclude build target, git internals, and node modules
    if 'target' in root or '.git' in root or 'node_modules' in root:
        continue
    for file in files:
        if file.endswith(('.md', '.sh', '.ps1', '.json', '.iss', '.toml', '.py')):
            path = os.path.join(root, file)
            # Skip this script itself and Cargo.lock / package-lock.json (we let package managers regenerate lockfiles)
            if file == 'update_versions.py' or file == 'Cargo.lock' or file == 'package-lock.json':
                continue
            # Update specific versions to 1.0.0-rc1
            replace_in_file(path, r'(?<!\d)0\.5\.0(?!\d)', '1.0.0-rc1')
            replace_in_file(path, r'(?<!\d)0\.6\.0(?!\d)', '1.0.0-rc1')
            replace_in_file(path, r'(?<!\d)0\.6\.1(?!\d)', '1.0.0-rc1')
