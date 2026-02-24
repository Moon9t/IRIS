base = 'C:/Users/Moon/Desktop/Projects/IRIS'
path = base + '/src/parser/parse.rs'
content = open(path).read()

# Find and replace the section
old_marker = '            if matches!(self.peek_tok(), Token::Let) {\n                stmts.push(self.parse_let_stmt()?);\n                continue;\n            }\n\n            // Expression'

new_insertion = '            if matches!(self.peek_tok(), Token::Let) {\n                stmts.push(self.parse_let_stmt()?);\n                continue;\n            }\n\n            // `while` statement\n            if matches!(self.peek_tok(), Token::While) {\n                stmts.push(self.parse_while_stmt()?);\n                continue;\n            }\n\n            // `loop` statement\n            if matches!(self.peek_tok(), Token::Loop) {\n                stmts.push(self.parse_loop_stmt()?);\n                continue;\n            }\n\n            // `break` statement\n            if matches!(self.peek_tok(), Token::Break) {\n                let span = self.advance().span;\n                if matches!(self.peek_tok(), Token::Semi) {\n                    self.advance();\n                }\n                stmts.push(AstStmt::Break { span });\n                continue;\n            }\n\n            // `continue` statement\n            if matches!(self.peek_tok(), Token::Continue) {\n                let span = self.advance().span;\n                if matches!(self.peek_tok(), Token::Semi) {\n                    self.advance();\n                }\n                stmts.push(AstStmt::Continue { span });\n                continue;\n            }\n\n            // Expression'

if old_marker in content:
    content = content.replace(old_marker, new_insertion)
    open(path, 'w').write(content)
    print('parse_block patched successfully')
else:
    print('ERROR: marker not found')
    # debug
    idx = content.find('Token::Let')
    print('Token::Let found at idx:', idx)
    print(repr(content[idx-50:idx+200]))
