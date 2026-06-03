import os
import re

def replace_in_file(path, old, new):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    if old in content:
        content = content.replace(old, new)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Replaced in {path}")
    else:
        print(f"Not found in {path}")

replace_in_file('docs/ROADMAP.md', 'marketplace publish pending', 'published to marketplace')
replace_in_file('docs/ROADMAP.md', '| 4 | Publish VS Code extension to marketplace | High | Pending |', '| 4 | Publish VS Code extension to marketplace | High | ? Done |')

with open('docs/STABILITY.md', 'r', encoding='utf8') as f:
    text = f.read()

# move ML stuff to Tier 1
text = re.sub(
    r'### Tier 3 - Experimental\n\nThese features are available but subject to redesign or removal\.\n\n- ML built-ins: `tensor<T,\[dims\]>`, `einsum`, `grad<T>`, `sparse<T>`\n- Model DSL \(`model \{ \.\.\. \}`\)\n- ONNX/CUDA/SIMD codegen targets\n- DAP debugger protocol\n- `atomic<T>`, `mutex<T>`',
    r'- ML built-ins: `tensor<T,[dims]>`, `einsum`, `grad<T>`, `sparse<T>`\n- Model DSL (`model { ... }`)\n- ONNX/CUDA/SIMD codegen targets\n- DAP debugger protocol\n- `atomic<T>`, `mutex<T>`\n\n### Tier 3 - Experimental\n\nThese features are available but subject to redesign or removal.\n- Undocumented CLI flags',
    text
)

with open('docs/STABILITY.md', 'w', encoding='utf8') as f:
    f.write(text)

print("done")
