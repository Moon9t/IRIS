path = 'C:/Users/Moon/Desktop/Projects/IRIS/src/ir/module.rs'
content = open(path).read()

old = '    /// Allocates a fresh `ValueId` without attaching it to any instruction.\n    /// Used by the lowerer when pre-allocating result values.\n    pub fn fresh_value(&mut self) -> ValueId {'

new = '    /// Returns true if the current block already ends with a terminator.\n    pub fn is_current_block_terminated(&self) -> bool {\n        if let Some(block_id) = self.current_block {\n            self.func.blocks[block_id.0 as usize].is_sealed()\n        } else {\n            false\n        }\n    }\n\n    /// Allocates a fresh `ValueId` without attaching it to any instruction.\n    /// Used by the lowerer when pre-allocating result values.\n    pub fn fresh_value(&mut self) -> ValueId {'

if old in content:
    content = content.replace(old, new)
    open(path, 'w').write(content)
    print('module.rs patched successfully')
else:
    print('ERROR: marker not found')
    idx = content.find('fresh_value')
    print('fresh_value at:', idx)
    print(repr(content[idx-100:idx+100]))
