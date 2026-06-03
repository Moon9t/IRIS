import os
import json
import glob

history_dir = os.path.expandvars('%APPDATA%/Code/User/History')
for root, _, files in os.walk(history_dir):
    if 'entries.json' in files:
        with open(os.path.join(root, 'entries.json'), 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                resource = data.get('resource', '')
                if any(x in resource for x in ['multimodal_ai_orchestrator', 'tensorrt.rs', 'ros2_bridge', 'ros2.iris', 'upgrade.rs']):
                    print('FOUND:', resource)
                    print('DIR:', root)
                    entries = data.get('entries', [])
                    if entries:
                        latest = entries[-1]['id']
                        print('LATEST FILE:', os.path.join(root, latest))
            except Exception as e:
                pass
