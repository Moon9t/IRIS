//! Handwritten recursive-descent parser for the IRIS DSL.
//!
//! The parser consumes a flat `&[Spanned<Token>]` produced by the lexer and
//! builds an `AstModule`. It reports errors with source spans for diagnostics.
//!
//! Grammar (informal):
//! ```text
//! module      := (def_def | record_def | model_def)*
//! model_def   := "model" IDENT "{" model_body "}"
//! model_body  := model_input* layer_def* model_output+
//! model_input := "input" IDENT ":" type
//! layer_def   := "layer" IDENT IDENT layer_params?
//! layer_params := "(" (layer_param ("," layer_param)*)? ")"
//! layer_param := IDENT "=" primary
//! model_output := "output" IDENT
//! def_def  := "def" IDENT "(" params ")" "->" type block
//! params   := (param ("," param)*)?
//! param    := IDENT ":" type
//! type     := scalar_type | tensor_type | named_type
//! scalar   := "f32" | "f64" | "i32" | "i64" | "bool"
//! tensor   := "tensor" "<" scalar "," "[" dims "]" ">"
//! dims     := (dim ("," dim)*)?
//! dim      := INT_LIT | IDENT
//! block    := "{" stmt* expr? "}"
//! stmt     := "val" IDENT [":" type] "=" expr ";"
//!           | expr ";"
//! expr     := add_expr ("to" type)?
//! add_expr := mul_expr (("+" | "-") mul_expr)*
//! mul_expr := cmp_expr (("*" | "/") cmp_expr)*
//! cmp_expr := primary (("==" | "!=" | "<" | "<=" | ">" | ">=") primary)*
//! primary  := IDENT [ "(" args ")" ]
//!           | INT_LIT | FLOAT_LIT | BOOL_LIT | STRING_LIT
//!           | "(" expr ")"
//!           | "if" expr block ("else" block)?
//!           | block
//! ```

use crate::error::ParseError;
use crate::ir::instr::BinOp;
use crate::parser::ast::{
    AstAssocTypeDecl, AstAttribute, AstBinOp, AstBlock, AstBring, AstConst, AstDim, AstEffectDef, AstEffectOperation, AstEnumDef, AstEnumVariant, AstExpr,
    AstFieldDef, AstFunction, AstGenericParam, AstHandlerArm, AstImplDef, AstLayer, AstLayerParam, AstModel,
    AstModelInput, AstModelOutput, AstModule, AstParam, AstScalarKind, AstStmt, AstStructDef,
    AstTraitDef, AstTraitMethod, AstType, AstTypeAlias, AstUnaryOp, AstWhenArm, AstWhenPattern,
    BringPath, Ident, Variance,
};
use crate::parser::lexer::{Span, Spanned, Token};

pub struct Parser<'t> {
    tokens: &'t [Spanned<Token>],
    pos: usize,
    /// Accumulated parse errors (for recovery mode).
    errors: Vec<ParseError>,
    /// Maximum number of errors before aborting.
    max_errors: usize,
}

impl<'t> Parser<'t> {
    pub fn new(tokens: &'t [Spanned<Token>]) -> Self {
        Self {
            tokens,
            pos: 0,
            errors: Vec::new(),
            max_errors: 50,
        }
    }

    /// Return all accumulated errors (empty if parsing succeeded).
    pub fn errors(&self) -> &[ParseError] {
        &self.errors
    }

    /// Parse the module with error recovery. Returns a partial AST and any
    /// accumulated errors. When `errors` is non-empty the AST may be
    /// incomplete but will still contain all successfully-parsed items.
    pub fn parse_module_recovering(&mut self) -> (AstModule, Vec<ParseError>) {
        let module = self.parse_module_inner();
        let errors = std::mem::take(&mut self.errors);
        (module, errors)
    }

    // -----------------------------------------------------------------------
    // Synchronization (error recovery)
    // -----------------------------------------------------------------------

    /// Skip tokens until we reach a token that can start a new top-level
    /// declaration (or EOF). This is the primary recovery point.
    fn synchronize(&mut self) {
        while !self.at_eof() {
            match self.peek_tok() {
                Token::Def
                | Token::Record
                | Token::Choice
                | Token::Model
                | Token::Const
                | Token::Type
                | Token::Trait
                | Token::Impl
                | Token::Bring
                | Token::Extern
                | Token::Pub
                | Token::Async => return,
                _ => {
                    self.advance();
                }
            }
        }
    }

    /// Record an error and synchronize.
    fn record_error(&mut self, err: ParseError) {
        self.errors.push(err);
        self.synchronize();
    }

    // -----------------------------------------------------------------------
    // Token stream helpers
    // -----------------------------------------------------------------------

    fn peek_tok(&self) -> &Token {
        &self.tokens[self.pos].node
    }

    fn current_span(&self) -> Span {
        self.tokens[self.pos].span
    }

    fn advance(&mut self) -> &Spanned<Token> {
        let t = &self.tokens[self.pos];
        if self.pos + 1 < self.tokens.len() {
            self.pos += 1;
        }
        t
    }

    fn is_handle_stmt_at_pos(&self) -> bool {
        if !matches!(self.peek_tok(), Token::Ident(name) if name == "handle") {
            return false;
        }
        let mut i = self.pos + 1;
        let mut brace_depth = 0i32;
        while i < self.tokens.len() {
            match &self.tokens[i].node {
                Token::LBrace => brace_depth += 1,
                Token::RBrace => {
                    brace_depth -= 1;
                    if brace_depth < 0 {
                        return false;
                    }
                }
                Token::With if brace_depth == 0 => return true,
                Token::Semi | Token::Eof if brace_depth == 0 => return false,
                Token::Eq | Token::PlusEq | Token::MinusEq | Token::StarEq | Token::SlashEq | Token::PercentEq if brace_depth == 0 => return false,
                _ => {}
            }
            i += 1;
        }
        false
    }

    fn expect(&mut self, expected: &Token) -> Result<Span, ParseError> {
        if self.peek_tok() == expected {
            Ok(self.advance().span)
        } else {
            Err(ParseError::UnexpectedToken {
                expected: format!("'{}'", expected),
                found: format!("{}", self.peek_tok()),
                span: self.current_span(),
            })
        }
    }

    fn expect_ident(&mut self) -> Result<Ident, ParseError> {
        match self.peek_tok().clone() {
            Token::Ident(name) => {
                let span = self.advance().span;
                Ok(Ident { name, span })
            }
            _ => Err(ParseError::UnexpectedToken {
                expected: "identifier".to_owned(),
                found: format!("{}", self.peek_tok()),
                span: self.current_span(),
            }),
        }
    }

    fn expect_ident_or_keyword(&mut self) -> Result<Ident, ParseError> {
        let span = self.current_span();
        let name = match self.peek_tok() {
            Token::Ident(s) => s.clone(),
            // Keywords
            Token::Def => "def".to_owned(),
            Token::Val => "val".to_owned(),
            Token::Var => "var".to_owned(),
            Token::Let => "let".to_owned(),
            Token::Return => "return".to_owned(),
            Token::If => "if".to_owned(),
            Token::Else => "else".to_owned(),
            Token::While => "while".to_owned(),
            Token::Loop => "loop".to_owned(),
            Token::Break => "break".to_owned(),
            Token::Continue => "continue".to_owned(),
            Token::Record => "record".to_owned(),
            Token::Bring => "bring".to_owned(),
            Token::When => "when".to_owned(),
            Token::Choice => "choice".to_owned(),
            Token::For => "for".to_owned(),
            Token::In => "in".to_owned(),
            Token::Spawn => "spawn".to_owned(),
            Token::Par => "par".to_owned(),
            Token::Async => "async".to_owned(),
            Token::Await => "await".to_owned(),
            Token::Const => "const".to_owned(),
            Token::Type => "type".to_owned(),
            Token::Trait => "trait".to_owned(),
            Token::Impl => "impl".to_owned(),
            Token::Pub => "pub".to_owned(),
            Token::Extern => "extern".to_owned(),
            Token::Mod => "mod".to_owned(),
            Token::Model => "model".to_owned(),
            Token::F32 => "f32".to_owned(),
            Token::F64 => "f64".to_owned(),
            Token::I32 => "i32".to_owned(),
            Token::I64 => "i64".to_owned(),
            Token::Bool => "bool".to_owned(),
            Token::Tensor => "tensor".to_owned(),
            Token::Str => "str".to_owned(),
            _ => {
                return Err(ParseError::UnexpectedToken {
                    expected: "identifier or keyword".to_owned(),
                    found: format!("{}", self.peek_tok()),
                    span,
                });
            }
        };
        self.advance();
        Ok(Ident { name, span })
    }

    fn peek_next_tok(&self) -> &Token {
        self.peek_at(1)
    }

    fn peek_at(&self, offset: usize) -> &Token {
        let idx = self.pos + offset;
        if idx < self.tokens.len() {
            &self.tokens[idx].node
        } else {
            &Token::Eof
        }
    }

    fn at_eof(&self) -> bool {
        matches!(self.peek_tok(), Token::Eof)
    }

    // -----------------------------------------------------------------------
    // Top-level
    // -----------------------------------------------------------------------

    pub fn parse_module(&mut self) -> Result<AstModule, ParseError> {
        let module = self.parse_module_inner();
        // If we accumulated errors, return the first one for backward compat.
        if !self.errors.is_empty() {
            return Err(self.errors.remove(0));
        }
        Ok(module)
    }

    /// Internal: parse the full module, recovering from errors in individual
    /// top-level declarations.
    fn parse_module_inner(&mut self) -> AstModule {
        let mut enums = Vec::new();
        let mut structs = Vec::new();
        let mut functions = Vec::new();
        let mut models = Vec::new();
        let mut consts = Vec::new();
        let mut type_aliases = Vec::new();
        let mut traits = Vec::new();
        let mut impls = Vec::new();
        let mut brings = Vec::new();
        let mut extern_fns = Vec::new();
        let mut effects = Vec::new();
        while !self.at_eof() {
            if self.errors.len() >= self.max_errors {
                break;
            }
            match self.peek_tok().clone() {
                Token::Choice => match self.parse_enum_def() {
                    Ok(e) => enums.push(e),
                    Err(e) => self.record_error(e),
                },
                Token::Record => match self.parse_struct_def() {
                    Ok(s) => structs.push(s),
                    Err(e) => self.record_error(e),
                },
                Token::Def | Token::Async | Token::At => match self.parse_fn() {
                    Ok(f) => functions.push(f),
                    Err(e) => self.record_error(e),
                },
                Token::Model => match self.parse_model() {
                    Ok(m) => models.push(m),
                    Err(e) => self.record_error(e),
                },
                Token::Const => match self.parse_const_decl() {
                    Ok(c) => consts.push(c),
                    Err(e) => self.record_error(e),
                },
                Token::Type => match self.parse_type_alias() {
                    Ok(t) => type_aliases.push(t),
                    Err(e) => self.record_error(e),
                },
                Token::Trait => match self.parse_trait_def() {
                    Ok(t) => traits.push(t),
                    Err(e) => self.record_error(e),
                },
                Token::Impl => match self.parse_impl_def() {
                    Ok(i) => impls.push(i),
                    Err(e) => self.record_error(e),
                },
                Token::Effect => match self.parse_effect_def() {
                    Ok(eff) => effects.push(eff),
                    Err(e) => self.record_error(e),
                },
                Token::Bring => {
                    let bring_span = self.current_span();
                    self.advance(); // consume 'bring'
                    let bring = match self.peek_tok().clone() {
                        // bring "path/to/file.iris"
                        Token::StringLit(path) => {
                            self.advance();
                            Ok(AstBring {
                                path: BringPath::File(path),
                                span: bring_span,
                            })
                        }
                        // bring std.name  OR  bring module_name
                        _ => {
                            match self.expect_ident_or_keyword() {
                                Ok(ident) => {
                                    let name = ident.name;
                                    if name == "std" && matches!(self.peek_tok(), Token::Dot) {
                                        self.advance(); // consume '.'
                                        match self.expect_ident_or_keyword() {
                                            Ok(lib) => Ok(AstBring {
                                                path: BringPath::Stdlib(lib.name),
                                                span: bring_span,
                                            }),
                                            Err(e) => Err(e),
                                        }
                                    } else {
                                        // Legacy: bring module_name → treat as File("module_name.iris")
                                        Ok(AstBring {
                                            path: BringPath::File(format!("{}.iris", name)),
                                            span: bring_span,
                                        })
                                    }
                                }
                                Err(_) => Err(ParseError::UnexpectedToken {
                                    expected: "module path (\"file.iris\", std.name, or identifier)".to_owned(),
                                    found: format!("{}", self.peek_tok()),
                                    span: self.current_span(),
                                }),
                            }
                        }
                    };
                    match bring {
                        Ok(b) => brings.push(b),
                        Err(e) => self.record_error(e),
                    }
                }
                Token::Extern => match self.parse_extern_fn() {
                    Ok(f) => extern_fns.push(f),
                    Err(e) => self.record_error(e),
                },
                Token::Pub => {
                    self.advance(); // consume 'pub'
                    match self.peek_tok().clone() {
                        Token::Def | Token::Async => match self.parse_fn() {
                            Ok(mut func) => {
                                func.is_pub = true;
                                functions.push(func);
                            }
                            Err(e) => self.record_error(e),
                        },
                        Token::Record => match self.parse_struct_def() {
                            Ok(mut s) => {
                                s.is_pub = true;
                                structs.push(s);
                            }
                            Err(e) => self.record_error(e),
                        },
                        Token::Choice => match self.parse_enum_def() {
                            Ok(mut e2) => {
                                e2.is_pub = true;
                                enums.push(e2);
                            }
                            Err(e) => self.record_error(e),
                        },
                        Token::Const => match self.parse_const_decl() {
                            Ok(mut c) => {
                                c.is_pub = true;
                                consts.push(c);
                            }
                            Err(e) => self.record_error(e),
                        },
                        Token::Type => match self.parse_type_alias() {
                            Ok(mut t) => {
                                t.is_pub = true;
                                type_aliases.push(t);
                            }
                            Err(e) => self.record_error(e),
                        },
                        Token::Trait => match self.parse_trait_def() {
                            Ok(t) => traits.push(t),
                            Err(e) => self.record_error(e),
                        },
                        _ => {
                            self.record_error(ParseError::UnexpectedToken {
                                expected: "'def', 'record', 'choice', 'const', 'type', or 'trait' after 'pub'".to_owned(),
                                found: format!("{}", self.peek_tok()),
                                span: self.current_span(),
                            });
                        }
                    }
                }
                _ => {
                    self.record_error(ParseError::UnexpectedToken {
                        expected: "'choice', 'record', 'def', 'extern', 'model', 'const', 'type', 'trait', 'impl', 'effect', or 'bring'".to_owned(),
                        found: format!("{}", self.peek_tok()),
                        span: self.current_span(),
                    });
                }
            }
        }
        AstModule {
            enums,
            structs,
            functions,
            models,
            consts,
            type_aliases,
            traits,
            impls,
            effects,
            brings,
            extern_fns,
        }
    }

    fn parse_effect_def(&mut self) -> Result<AstEffectDef, ParseError> {
        let start = self.current_span();
        self.expect(&Token::Effect)?;
        let name = self.expect_ident()?;
        self.expect(&Token::LBrace)?;
        let mut operations = Vec::new();
        while !matches!(self.peek_tok(), Token::RBrace | Token::Eof) {
            let op_start = self.current_span();
            self.expect(&Token::Def)?;
            let op_name = self.expect_ident()?;
            self.expect(&Token::LParen)?;
            let params = self.parse_params()?;
            self.expect(&Token::RParen)?;
            self.expect(&Token::Arrow)?;
            let ret_ty = self.parse_type()?;
            if matches!(self.peek_tok(), Token::Semi) {
                self.advance();
            }
            let op_span = op_start.merge(self.current_span());
            operations.push(AstEffectOperation {
                name: op_name,
                params,
                ret_ty,
                span: op_span,
            });
        }
        let rbrace_span = self.expect(&Token::RBrace)?;
        let span = start.merge(rbrace_span);
        Ok(AstEffectDef {
            name,
            operations,
            span,
        })
    }

    /// Parses `extern ["C"] def name(params) -> ret_ty` (no body).
    /// Optionally preceded by `@link(name = "lib")` attributes.
    fn parse_extern_fn(&mut self) -> Result<crate::parser::ast::AstExternFn, ParseError> {
        use crate::parser::ast::AstExternFn;

        // Check for @link(name = "lib") attribute before `extern`
        let mut link_lib: Option<String> = None;
        while matches!(self.peek_tok(), &Token::At) {
            let attr_span = self.advance().span;
            let attr_name = self.expect_ident()?;
            if attr_name.name == "link" || attr_name.name == "link_name" {
                if matches!(self.peek_tok(), &Token::LParen) {
                    self.advance();
                    // Eat optional `name = ` prefix
                    if matches!(self.peek_tok(), &Token::Ident(_)) {
                        if let Token::Ident(ref s) = self.peek_tok().clone() {
                            if s == "name" {
                                self.advance();
                                if matches!(self.peek_tok(), &Token::Eq) {
                                    self.advance();
                                }
                            }
                        }
                    }
                    if let Token::StringLit(s) = self.advance().node.clone() {
                        link_lib = Some(s);
                    }
                    if matches!(self.peek_tok(), &Token::RParen) {
                        self.advance();
                    }
                }
            }
            let _ = attr_span;
        }

        let span_start = self.current_span();
        self.expect(&Token::Extern)?;

        // Optional calling convention: extern "C" def ...
        let abi = if matches!(self.peek_tok(), &Token::StringLit(_)) {
            match self.advance().node.clone() {
                Token::StringLit(s) => Some(s),
                _ => None,
            }
        } else {
            None
        };

        self.expect(&Token::Def)?;
        let name = self.expect_ident()?;
        self.expect(&Token::LParen)?;
        let mut params = Vec::new();
        while !matches!(self.peek_tok(), &Token::RParen) {
            let param_name = self.expect_ident()?;
            self.expect(&Token::Colon)?;
            let ty = self.parse_type()?;
            params.push(crate::parser::ast::AstParam {
                name: param_name,
                ty,
                default: None,
            });
            if matches!(self.peek_tok(), &Token::Comma) {
                self.advance();
            }
        }
        self.expect(&Token::RParen)?;
        self.expect(&Token::Arrow)?;
        let ret_ty = self.parse_type()?;
        let span = span_start.merge(self.current_span());
        Ok(AstExternFn {
            name,
            params,
            ret_ty,
            abi,
            link_lib,
            span,
        })
    }

    /// Parses a type name as a plain string (handles keywords like `i64`, `f64`, `bool`, `str`
    /// in addition to bare identifiers). Used for `impl Trait for TypeName`.
    fn parse_type_name_str(&mut self) -> Result<String, ParseError> {
        let name = match self.peek_tok().clone() {
            Token::I64 => {
                self.advance();
                "i64".to_owned()
            }
            Token::I32 => {
                self.advance();
                "i32".to_owned()
            }
            Token::F64 => {
                self.advance();
                "f64".to_owned()
            }
            Token::F32 => {
                self.advance();
                "f32".to_owned()
            }
            Token::Bool => {
                self.advance();
                "bool".to_owned()
            }
            Token::Str => {
                self.advance();
                "str".to_owned()
            }
            Token::Ident(n) => {
                let n = n.clone();
                self.advance();
                n
            }
            _ => {
                return Err(ParseError::UnexpectedToken {
                    expected: "type name".to_owned(),
                    found: format!("{}", self.peek_tok()),
                    span: self.current_span(),
                })
            }
        };
        let mut full_name = name;
        while matches!(self.peek_tok(), Token::Dot) {
            if let Token::Ident(ref subname) = self.peek_next_tok() {
                let subname = subname.clone();
                self.advance(); // consume '.'
                self.advance(); // consume subname
                full_name = format!("{}__{}", full_name, subname);
            } else {
                break;
            }
        }
        Ok(full_name)
    }

    /// Parses `trait Name { (type Name;)? (def method(params) -> type)* }`.
    fn parse_trait_def(&mut self) -> Result<AstTraitDef, ParseError> {
        let start = self.current_span();
        self.expect(&Token::Trait)?;
        let name = self.expect_ident()?;
        self.expect(&Token::LBrace)?;
        let mut assoc_types = Vec::new();
        let mut methods = Vec::new();
        while !matches!(self.peek_tok(), Token::RBrace | Token::Eof) {
            if matches!(self.peek_tok(), Token::Type) {
                self.advance(); // consume 'type'
                let aname = self.expect_ident()?;
                assoc_types.push(AstAssocTypeDecl {
                    name: aname.clone(),
                    span: aname.span,
                });
                continue;
            }
            let m_start = self.current_span();
            self.expect(&Token::Def)?;
            let m_name = self.expect_ident()?;
            self.expect(&Token::LParen)?;
            let mut params = Vec::new();
            while !matches!(self.peek_tok(), Token::RParen | Token::Eof) {
                let pname = self.expect_ident()?;
                // Allow bare `self` without type annotation.
                let pty = if pname.name == "self"
                    && matches!(self.peek_tok(), Token::Comma | Token::RParen)
                {
                    AstType::Named("self".to_string(), self.current_span())
                } else {
                    self.expect(&Token::Colon)?;
                    self.parse_type()?
                };
                params.push(AstParam {
                    name: pname,
                    ty: pty,
                    default: None,
                });
                if matches!(self.peek_tok(), Token::Comma) {
                    self.advance();
                }
            }
            self.expect(&Token::RParen)?;
            self.expect(&Token::Arrow)?;
            let ret = self.parse_type()?;
            let m_end = ret.span();
            methods.push(AstTraitMethod {
                name: m_name,
                params,
                return_ty: ret,
                span: m_start.merge(m_end),
            });
        }
        let end = self.expect(&Token::RBrace)?;
        Ok(AstTraitDef {
            name,
            assoc_types,
            methods,
            span: start.merge(end),
        })
    }

    /// Parses either:
    /// - `impl TraitName for TypeName { ... }` — trait implementation
    /// - `impl TypeName { ... }` — standalone struct methods (trait_name = "")
    fn parse_impl_def(&mut self) -> Result<AstImplDef, ParseError> {
        let start = self.current_span();
        self.expect(&Token::Impl)?;
        // Disambiguate: if the token after the first ident is `for`, it's a trait impl.
        // Otherwise it's a standalone struct impl block.
        let first_name = self.parse_type_name_str()?;
        let (trait_name, type_name) = if matches!(self.peek_tok(), Token::For) {
            self.advance(); // consume `for`
            let type_name = self.parse_type_name_str()?;
            (first_name, type_name)
        } else {
            // Standalone `impl TypeName { ... }` — no trait
            ("".to_string(), first_name)
        };
        self.expect(&Token::LBrace)?;
        let mut assoc_type_bindings = Vec::new();
        let mut methods = Vec::new();
        while !matches!(self.peek_tok(), Token::RBrace | Token::Eof) {
            if matches!(self.peek_tok(), Token::Type) {
                self.advance(); // consume 'type'
                let aname = self.expect_ident()?.name;
                self.expect(&Token::Eq)?;
                let ty = self.parse_type()?;
                assoc_type_bindings.push((aname, ty));
                continue;
            }
            methods.push(self.parse_fn()?);
        }
        let end = self.expect(&Token::RBrace)?;
        Ok(AstImplDef {
            trait_name,
            type_name,
            assoc_type_bindings,
            methods,
            span: start.merge(end),
        })
    }

    /// Parses `type Name = Type`.
    fn parse_type_alias(&mut self) -> Result<AstTypeAlias, ParseError> {
        let start = self.current_span();
        self.expect(&Token::Type)?;
        let name = self.expect_ident()?.name;
        self.expect(&Token::Eq)?;
        let ty = self.parse_type()?;
        let end = start; // span is approximate — just use the keyword span
        Ok(AstTypeAlias {
            name,
            ty,
            span: start.merge(end),
            is_pub: false,
        })
    }

    /// Parses `const NAME [: type] = expr`.
    fn parse_const_decl(&mut self) -> Result<AstConst, ParseError> {
        let start = self.current_span();
        self.expect(&Token::Const)?;
        let name = self.expect_ident()?;
        let ty = if matches!(self.peek_tok(), Token::Colon) {
            self.advance();
            Some(self.parse_type()?)
        } else {
            None
        };
        self.expect(&Token::Eq)?;
        let value = self.parse_expr()?;
        let end = value.span();
        Ok(AstConst {
            name,
            ty,
            value,
            span: start.merge(end),
            is_pub: false,
        })
    }

    fn parse_enum_def(&mut self) -> Result<AstEnumDef, ParseError> {
        let start = self.current_span();
        self.expect(&Token::Choice)?;
        let name = self.expect_ident()?;
        self.expect(&Token::LBrace)?;
        let mut variants = Vec::new();
        while !matches!(self.peek_tok(), Token::RBrace | Token::Eof) {
            let v_start = self.current_span();
            let v_name = self.expect_ident()?;
            // Optionally parse payload types: `Variant(T1, T2, ...)`.
            let fields = if matches!(self.peek_tok(), Token::LParen) {
                self.advance(); // consume '('
                let mut tys = Vec::new();
                while !matches!(self.peek_tok(), Token::RParen | Token::Eof) {
                    tys.push(self.parse_type()?);
                    if matches!(self.peek_tok(), Token::Comma) {
                        self.advance();
                    }
                }
                self.expect(&Token::RParen)?;
                tys
            } else {
                Vec::new()
            };
            let v_end = self.current_span();
            variants.push(AstEnumVariant {
                name: v_name,
                fields,
                span: v_start.merge(v_end),
            });
            if matches!(self.peek_tok(), Token::Comma) {
                self.advance();
            }
        }
        let end = self.expect(&Token::RBrace)?;
        Ok(AstEnumDef {
            name,
            variants,
            span: start.merge(end),
            is_pub: false,
        })
    }

    fn parse_struct_def(&mut self) -> Result<AstStructDef, ParseError> {
        let start = self.current_span();
        self.expect(&Token::Record)?;
        let name = self.expect_ident()?;
        // Optional type parameters: `[T, U, ...]`
        let type_params = if matches!(self.peek_tok(), Token::LBracket) {
            self.advance(); // consume '['
            let mut ty_params = Vec::new();
            while !matches!(self.peek_tok(), Token::RBracket | Token::Eof) {
                let tp = self.parse_generic_param()?;
                ty_params.push(tp);
                if matches!(self.peek_tok(), Token::Comma) {
                    self.advance();
                } else {
                    break;
                }
            }
            self.expect(&Token::RBracket)?;
            ty_params
        } else {
            Vec::new()
        };
        self.expect(&Token::LBrace)?;
        let mut fields = Vec::new();
        while !matches!(self.peek_tok(), Token::RBrace | Token::Eof) {
            let field_name = self.expect_ident()?;
            self.expect(&Token::Colon)?;
            let ty = self.parse_type()?;
            fields.push(AstFieldDef {
                name: field_name,
                ty,
            });
            if matches!(self.peek_tok(), Token::Comma) {
                self.advance();
            }
        }
        let end = self.expect(&Token::RBrace)?;
        Ok(AstStructDef {
            name,
            type_params,
            fields,
            span: start.merge(end),
            is_pub: false,
        })
    }

    fn parse_generic_param(&mut self) -> Result<AstGenericParam, ParseError> {
        if matches!(self.peek_tok(), Token::Const) {
            self.advance(); // consume "const"
            let name = self.expect_ident()?.name;
            self.expect(&Token::Colon)?;
            let kind = self.parse_type()?;
            Ok(AstGenericParam::Const {
                name,
                kind: Box::new(kind),
            })
        } else {
            let variance = if matches!(self.peek_tok(), Token::Plus) {
                self.advance();
                Variance::Covariant
            } else if matches!(self.peek_tok(), Token::Minus) {
                self.advance();
                Variance::Contravariant
            } else {
                Variance::Invariant
            };
            let tp = self.expect_ident()?;
            
            // Check for HKT: `F[T, U]`
            let mut nested = Vec::new();
            if matches!(self.peek_tok(), Token::LBracket) {
                self.advance(); // consume '['
                while !matches!(self.peek_tok(), Token::RBracket | Token::Eof) {
                    let param = self.parse_generic_param()?;
                    nested.push(param);
                    if matches!(self.peek_tok(), Token::Comma) {
                        self.advance();
                    } else {
                        break;
                    }
                }
                self.expect(&Token::RBracket)?;
            }

            let mut bounds = Vec::new();
            // Parse optional "where T: Trait [, Trait2 ...]" constraint.
            if matches!(self.peek_tok(), Token::Ident(ref w) if w == "where") {
                self.advance(); // consume "where"
                // Consume repeated type param name if present (e.g., "T" in "where T: Ord").
                if matches!(self.peek_tok(), Token::Ident(_)) {
                    self.advance();
                }
                self.expect(&Token::Colon)?;
                while matches!(self.peek_tok(), Token::Ident(_)) {
                    let trait_name = self.expect_ident()?.name;
                    bounds.push(trait_name);
                    if matches!(self.peek_tok(), Token::Comma) {
                        self.advance();
                    } else {
                        break;
                    }
                }
            }
            
            if nested.is_empty() {
                Ok(AstGenericParam::Type(tp.name, bounds, variance))
            } else {
                Ok(AstGenericParam::Hkt(tp.name, nested, bounds, variance))
            }
        }
    }

    fn parse_fn(&mut self) -> Result<AstFunction, ParseError> {
        let start = self.current_span();
        // Optional @attr annotations before async/def
        let mut attrs = Vec::new();
        while matches!(self.peek_tok(), Token::At) {
            let attr_span = self.current_span();
            self.advance(); // consume '@'
            let attr_name = self.expect_ident()?.name;
            // Optional arguments: @attr(arg1, arg2=value, ...)
            let mut args = Vec::new();
            if matches!(self.peek_tok(), Token::LParen) {
                self.advance(); // consume '('
                if !matches!(self.peek_tok(), Token::RParen) {
                    loop {
                        let arg_expr = self.parse_expr()?;
                        args.push(arg_expr);
                        if matches!(self.peek_tok(), Token::Comma) {
                            self.advance();
                            if matches!(self.peek_tok(), Token::RParen) {
                                break;
                            }
                        } else {
                            break;
                        }
                    }
                }
                self.expect(&Token::RParen)?;
            }
            attrs.push(AstAttribute {
                name: attr_name,
                args,
                span: attr_span,
            });
        }
        // Optional async keyword before def
        let is_async = if matches!(self.peek_tok(), Token::Async) {
            self.advance();
            true
        } else {
            false
        };
        self.expect(&Token::Def)?;
        let name = self.expect_ident()?;
        // Optional type parameters: `[T, U, ...]`
        // Supports optional "where T: Trait" constraint annotation (parsed and discarded).
        // Example: `def max[T where T: Ord](a: T, b: T) -> T`
        let type_params = if matches!(self.peek_tok(), Token::LBracket) {
            self.advance(); // consume '['
            let mut ty_params = Vec::new();
            while !matches!(self.peek_tok(), Token::RBracket | Token::Eof) {
                let tp = self.parse_generic_param()?;
                ty_params.push(tp);
                if matches!(self.peek_tok(), Token::Comma) {
                    self.advance();
                } else {
                    break;
                }
            }
            self.expect(&Token::RBracket)?;
            ty_params
        } else {
            Vec::new()
        };
        self.expect(&Token::LParen)?;
        let params = self.parse_params()?;
        self.expect(&Token::RParen)?;
        self.expect(&Token::Arrow)?;
        let return_ty = self.parse_type()?;
        let effects = if self.peek_tok() == &Token::Effect || self.peek_tok() == &Token::With {
            self.advance();
            let mut effs = Vec::new();
            effs.push(self.expect_ident()?.name);
            while self.peek_tok() == &Token::Comma {
                self.advance();
                effs.push(self.expect_ident()?.name);
            }
            effs
        } else {
            Vec::new()
        };
        let body = self.parse_block()?;
        let span = start.merge(body.span);
        Ok(AstFunction {
            name,
            is_pub: false, // set to true by parse_module when preceded by `pub`
            type_params,
            params,
            return_ty,
            effects,
            body,
            span,
            is_async,
            attrs,
        })
    }

    // -----------------------------------------------------------------------
    // Model definitions
    // -----------------------------------------------------------------------

    fn parse_model(&mut self) -> Result<AstModel, ParseError> {
        let start = self.current_span();
        self.expect(&Token::Model)?;
        let name = self.expect_ident()?;
        self.expect(&Token::LBrace)?;

        let mut inputs = Vec::new();
        let mut layers = Vec::new();
        let mut outputs = Vec::new();

        loop {
            match self.peek_tok().clone() {
                Token::RBrace | Token::Eof => break,
                Token::Ident(ref kw) if kw == "input" => inputs.push(self.parse_model_input()?),
                Token::Ident(ref kw) if kw == "layer" => layers.push(self.parse_layer()?),
                Token::Ident(ref kw) if kw == "output" => outputs.push(self.parse_model_output()?),
                _ => {
                    return Err(ParseError::UnexpectedToken {
                        expected: "'input', 'layer', or 'output'".to_owned(),
                        found: format!("{}", self.peek_tok()),
                        span: self.current_span(),
                    })
                }
            }
        }

        let end = self.expect(&Token::RBrace)?;
        Ok(AstModel {
            name,
            inputs,
            layers,
            outputs,
            span: start.merge(end),
        })
    }

    fn parse_model_input(&mut self) -> Result<AstModelInput, ParseError> {
        let start = self.current_span();
        self.advance(); // consume 'input' (already matched as Ident("input"))
        let name = self.expect_ident()?;
        self.expect(&Token::Colon)?;
        let ty = self.parse_type()?;
        let end = ty.span();
        Ok(AstModelInput {
            name,
            ty,
            span: start.merge(end),
        })
    }

    fn parse_layer(&mut self) -> Result<AstLayer, ParseError> {
        let start = self.current_span();
        self.advance(); // consume 'layer' (already matched as Ident("layer"))
        let name = self.expect_ident()?;
        let op = self.expect_ident()?;
        let (input_refs, params) = if matches!(self.peek_tok(), Token::LParen) {
            self.parse_layer_params()?
        } else {
            (vec![], vec![])
        };
        let end = self.tokens[self.pos - 1].span;
        Ok(AstLayer {
            name,
            op,
            input_refs,
            params,
            span: start.merge(end),
        })
    }

    /// Parses `( [arg, ...] )` where each arg is either:
    /// - `IDENT "=" primary`  → keyword hyperparameter
    /// - `IDENT`              → explicit input reference (bare ident, no `=`)
    fn parse_layer_params(&mut self) -> Result<(Vec<Ident>, Vec<AstLayerParam>), ParseError> {
        self.expect(&Token::LParen)?;
        let mut input_refs = Vec::new();
        let mut params = Vec::new();
        while !matches!(self.peek_tok(), Token::RParen | Token::Eof) {
            if matches!(self.peek_tok(), Token::Ident(_))
                && matches!(self.peek_next_tok(), Token::Eq)
            {
                // keyword param: key = value
                let key = self.expect_ident()?;
                self.expect(&Token::Eq)?;
                let value = self.parse_primary()?;
                let end = value.span();
                params.push(AstLayerParam {
                    span: key.span.merge(end),
                    key,
                    value,
                });
            } else {
                // input ref: bare ident
                input_refs.push(self.expect_ident()?);
            }
            if matches!(self.peek_tok(), Token::Comma) {
                self.advance();
            }
        }
        self.expect(&Token::RParen)?;
        Ok((input_refs, params))
    }

    fn parse_model_output(&mut self) -> Result<AstModelOutput, ParseError> {
        let start = self.current_span();
        self.advance(); // consume 'output' (already matched as Ident("output"))
        let name = self.expect_ident()?;
        let end = name.span;
        Ok(AstModelOutput {
            name,
            span: start.merge(end),
        })
    }

    fn parse_params(&mut self) -> Result<Vec<AstParam>, ParseError> {
        let mut params = Vec::new();
        if matches!(self.peek_tok(), Token::RParen) {
            return Ok(params);
        }
        params.push(self.parse_param()?);
        while matches!(self.peek_tok(), Token::Comma) {
            self.advance(); // consume ','
            if matches!(self.peek_tok(), Token::RParen) {
                break; // trailing comma
            }
            params.push(self.parse_param()?);
        }
        Ok(params)
    }

    fn parse_param(&mut self) -> Result<AstParam, ParseError> {
        let name = self.expect_ident()?;
        // Allow bare `self` without type annotation (trait/impl methods).
        if name.name == "self" && matches!(self.peek_tok(), Token::Comma | Token::RParen) {
            return Ok(AstParam {
                name,
                ty: AstType::Named("self".to_string(), self.current_span()),
                default: None,
            });
        }
        self.expect(&Token::Colon)?;
        let ty = self.parse_type()?;
        let default = if matches!(self.peek_tok(), Token::Eq) {
            self.advance(); // consume '='
            Some(self.parse_expr()?)
        } else {
            None
        };
        Ok(AstParam { name, ty, default })
    }

    // -----------------------------------------------------------------------
    // Types
    // -----------------------------------------------------------------------

    fn parse_type(&mut self) -> Result<AstType, ParseError> {
        let span = self.current_span();
        match self.peek_tok().clone() {
            Token::F32 => {
                self.advance();
                Ok(AstType::Scalar(AstScalarKind::F32, span))
            }
            Token::F64 => {
                self.advance();
                Ok(AstType::Scalar(AstScalarKind::F64, span))
            }
            Token::I32 => {
                self.advance();
                Ok(AstType::Scalar(AstScalarKind::I32, span))
            }
            Token::I64 => {
                self.advance();
                Ok(AstType::Scalar(AstScalarKind::I64, span))
            }
            Token::Bool => {
                self.advance();
                Ok(AstType::Scalar(AstScalarKind::Bool, span))
            }
            Token::I8 => {
                self.advance();
                Ok(AstType::Scalar(AstScalarKind::I8, span))
            }
            Token::U8 => {
                self.advance();
                Ok(AstType::Scalar(AstScalarKind::U8, span))
            }
            Token::U32 => {
                self.advance();
                Ok(AstType::Scalar(AstScalarKind::U32, span))
            }
            Token::U64 => {
                self.advance();
                Ok(AstType::Scalar(AstScalarKind::U64, span))
            }
            Token::Usize => {
                self.advance();
                Ok(AstType::Scalar(AstScalarKind::USize, span))
            }
            Token::Tensor => {
                self.advance();
                self.expect(&Token::LAngle)?;
                let dtype = self.parse_scalar_kind()?;
                self.expect(&Token::Comma)?;
                self.expect(&Token::LBracket)?;
                let dims = self.parse_dims()?;
                self.expect(&Token::RBracket)?;
                let end = self.expect(&Token::RAngle)?;
                Ok(AstType::Tensor {
                    dtype,
                    dims,
                    span: span.merge(end),
                })
            }
            Token::Str => {
                self.advance();
                Ok(AstType::Named("str".to_string(), span))
            }
            Token::LBracket => {
                // [T; N] — fixed-length array type
                self.advance(); // consume '['
                let elem = self.parse_type()?;
                self.expect(&Token::Semi)?;
                let (len, len_expr) = match self.peek_tok().clone() {
                    Token::IntLit(n) => {
                        self.advance();
                        (n as usize, None)
                    }
                    Token::Ident(name) => {
                        self.advance();
                        (0, Some(Box::new(AstExpr::Ident(Ident { name, span: self.current_span() }))))
                    }
                    _ => {
                        return Err(ParseError::UnexpectedToken {
                            expected: "integer length or identifier for array type".to_owned(),
                            found: format!("{}", self.peek_tok()),
                            span: self.current_span(),
                        })
                    }
                };
                let end = self.expect(&Token::RBracket)?;
                Ok(AstType::Array {
                    elem: Box::new(elem),
                    len,
                    len_expr,
                    span: span.merge(end),
                })
            }
            Token::Ident(ref name) if name == "chan" => {
                let _ = name.clone();
                self.advance(); // consume "chan"
                self.expect(&Token::LAngle)?;
                let inner = self.parse_type()?;
                let end = self.expect(&Token::RAngle)?;
                Ok(AstType::Chan(Box::new(inner), span.merge(end)))
            }
            Token::Ident(ref name) if name == "atomic" => {
                let _ = name.clone();
                self.advance();
                self.expect(&Token::LAngle)?;
                let inner = self.parse_type()?;
                let end = self.expect(&Token::RAngle)?;
                Ok(AstType::Atomic(Box::new(inner), span.merge(end)))
            }
            Token::Ident(ref name) if name == "mutex" => {
                let _ = name.clone();
                self.advance();
                self.expect(&Token::LAngle)?;
                let inner = self.parse_type()?;
                let end = self.expect(&Token::RAngle)?;
                Ok(AstType::Mutex(Box::new(inner), span.merge(end)))
            }
            Token::Ident(ref name) if name == "grad" => {
                let _ = name.clone();
                self.advance();
                self.expect(&Token::LAngle)?;
                let inner = self.parse_type()?;
                let end = self.expect(&Token::RAngle)?;
                Ok(AstType::Grad(Box::new(inner), span.merge(end)))
            }
            Token::Ident(ref name) if name == "sparse" => {
                let _ = name.clone();
                self.advance();
                self.expect(&Token::LAngle)?;
                let inner = self.parse_type()?;
                let end = self.expect(&Token::RAngle)?;
                Ok(AstType::Sparse(Box::new(inner), span.merge(end)))
            }
            Token::Ident(ref name) if name == "weak_ref" => {
                let _ = name.clone();
                self.advance();
                self.expect(&Token::LAngle)?;
                let inner = self.parse_type()?;
                let end = self.expect(&Token::RAngle)?;
                Ok(AstType::WeakRef(Box::new(inner), span.merge(end)))
            }
            Token::Ident(ref name) if name == "list" => {
                let _ = name.clone();
                self.advance();
                self.expect(&Token::LAngle)?;
                let inner = self.parse_type()?;
                let end = self.expect(&Token::RAngle)?;
                Ok(AstType::List(Box::new(inner), span.merge(end)))
            }
            Token::Ident(ref name) if name == "map" => {
                let _ = name.clone();
                self.advance();
                self.expect(&Token::LAngle)?;
                let k = self.parse_type()?;
                self.expect(&Token::Comma)?;
                let v = self.parse_type()?;
                let end = self.expect(&Token::RAngle)?;
                Ok(AstType::Map(Box::new(k), Box::new(v), span.merge(end)))
            }
            Token::Ident(ref name) if name == "option" => {
                let name = name.clone();
                let _ = name;
                self.advance(); // consume "option"
                self.expect(&Token::LAngle)?;
                let inner = self.parse_type()?;
                let end = self.expect(&Token::RAngle)?;
                Ok(AstType::Option(Box::new(inner), span.merge(end)))
            }
            Token::Ident(ref name) if name == "result" => {
                let name = name.clone();
                let _ = name;
                self.advance(); // consume "result"
                self.expect(&Token::LAngle)?;
                let ok_ty = self.parse_type()?;
                self.expect(&Token::Comma)?;
                let err_ty = self.parse_type()?;
                let end = self.expect(&Token::RAngle)?;
                Ok(AstType::Result(
                    Box::new(ok_ty),
                    Box::new(err_ty),
                    span.merge(end),
                ))
            }
            Token::Dyn => {
                self.advance();
                let trait_name = self.expect_ident()?.name;
                let end = self.current_span();
                Ok(AstType::DynTrait {
                    trait_name,
                    span: span.merge(end),
                })
            }
            Token::Ident(name) => {
                self.advance();
                // Check for generic type args: `Name<T, U, ...>`
                if matches!(self.peek_tok(), Token::LAngle) {
                    self.advance(); // consume '<'
                    let mut args = Vec::new();
                    while !matches!(self.peek_tok(), Token::RAngle | Token::Eof) {
                        let arg = self.parse_type()?;
                        args.push(arg);
                        if matches!(self.peek_tok(), Token::Comma) {
                            self.advance();
                        } else {
                            break;
                        }
                    }
                    let end = self.expect(&Token::RAngle)?;
                    return Ok(AstType::Generic {
                        name,
                        args,
                        span: span.merge(end),
                    });
                }
                // Check for associated type: `Self::Item` or `T::Item`
                if matches!(self.peek_tok(), Token::DoubleColon) {
                    self.advance(); // consume '::'
                    let assoc_ident = self.expect_ident()?;
                    return Ok(AstType::AssocType {
                        base: name,
                        assoc_name: assoc_ident.name,
                        span: span.merge(assoc_ident.span),
                    });
                }
                let mut full_name = name;
                let mut current_span = span;
                while matches!(self.peek_tok(), Token::Dot) {
                    if let Token::Ident(ref subname) = self.peek_next_tok() {
                        let subname = subname.clone();
                        self.advance(); // consume '.'
                        let next_span = self.advance().span; // consume subname
                        full_name = format!("{}__{}", full_name, subname);
                        current_span = current_span.merge(next_span);
                    } else {
                        break;
                    }
                }
                Ok(AstType::Named(full_name, current_span))
            }
            Token::LParen => {
                self.advance(); // consume '('
                let mut elems = Vec::new();
                if !matches!(self.peek_tok(), Token::RParen) {
                    elems.push(self.parse_type()?);
                    while matches!(self.peek_tok(), Token::Comma) {
                        self.advance();
                        if matches!(self.peek_tok(), Token::RParen) {
                            break;
                        }
                        elems.push(self.parse_type()?);
                    }
                }
                let end = self.expect(&Token::RParen)?;
                // Check for function type: (T1, T2) -> R
                if matches!(self.peek_tok(), Token::Arrow) {
                    self.advance(); // consume '->'
                    let ret = self.parse_type()?;
                    let ret_span = ret.span();
                    Ok(AstType::Fn {
                        params: elems,
                        ret: Box::new(ret),
                        span: span.merge(ret_span),
                    })
                } else {
                    Ok(AstType::Tuple(elems, span.merge(end)))
                }
            }
            Token::Pipe => {
                // Closure type: |T1, T2, ...| -> R
                self.advance(); // consume '|'
                let mut params = Vec::new();
                while !matches!(self.peek_tok(), Token::Pipe | Token::Eof) {
                    params.push(self.parse_type()?);
                    if matches!(self.peek_tok(), Token::Comma) {
                        self.advance();
                    }
                }
                let _end = self.expect(&Token::Pipe)?;
                self.expect(&Token::Arrow)?;
                let ret = self.parse_type()?;
                let ret_span = ret.span();
                Ok(AstType::Fn {
                    params,
                    ret: Box::new(ret),
                    span: span.merge(ret_span),
                })
            }
            Token::IntLit(n) => {
                self.advance();
                Ok(AstType::ConstInt(n, span))
            }
            _ => Err(ParseError::UnexpectedToken {
                expected: "type".to_owned(),
                found: format!("{}", self.peek_tok()),
                span,
            }),
        }
    }

    fn parse_scalar_kind(&mut self) -> Result<AstScalarKind, ParseError> {
        let span = self.current_span();
        match self.peek_tok().clone() {
            Token::F32 => {
                self.advance();
                Ok(AstScalarKind::F32)
            }
            Token::F64 => {
                self.advance();
                Ok(AstScalarKind::F64)
            }
            Token::I32 => {
                self.advance();
                Ok(AstScalarKind::I32)
            }
            Token::I64 => {
                self.advance();
                Ok(AstScalarKind::I64)
            }
            Token::Bool => {
                self.advance();
                Ok(AstScalarKind::Bool)
            }
            _ => Err(ParseError::UnexpectedToken {
                expected: "scalar type (f32, f64, i32, i64, bool)".to_owned(),
                found: format!("{}", self.peek_tok()),
                span,
            }),
        }
    }

    fn parse_dims(&mut self) -> Result<Vec<AstDim>, ParseError> {
        let mut dims = Vec::new();
        if matches!(self.peek_tok(), Token::RBracket) {
            return Ok(dims);
        }
        dims.push(self.parse_dim()?);
        while matches!(self.peek_tok(), Token::Comma) {
            self.advance();
            if matches!(self.peek_tok(), Token::RBracket) {
                break;
            }
            dims.push(self.parse_dim()?);
        }
        Ok(dims)
    }

    fn parse_dim(&mut self) -> Result<AstDim, ParseError> {
        let span = self.current_span();
        match self.peek_tok().clone() {
            Token::IntLit(n) => {
                self.advance();
                Ok(AstDim::Literal(n as u64))
            }
            Token::Ident(name) => {
                self.advance();
                Ok(AstDim::Symbol(Ident { name, span }))
            }
            _ => Err(ParseError::UnexpectedToken {
                expected: "integer literal or identifier for dimension".to_owned(),
                found: format!("{}", self.peek_tok()),
                span,
            }),
        }
    }

    // -----------------------------------------------------------------------
    // Blocks and statements
    // -----------------------------------------------------------------------

    fn parse_block(&mut self) -> Result<AstBlock, ParseError> {
        let start = self.expect(&Token::LBrace)?;
        let mut stmts = Vec::new();
        let mut tail: Option<Box<AstExpr>> = None;

        loop {
            if matches!(self.peek_tok(), Token::RBrace | Token::Eof) {
                break;
            }

            // `val` / `var` / `let` binding statement
            if matches!(self.peek_tok(), Token::Val | Token::Var | Token::Let) {
                stmts.push(self.parse_let_stmt()?);
                continue;
            }

            // `while` statement
            if matches!(self.peek_tok(), Token::While) {
                stmts.push(self.parse_while_stmt()?);
                if matches!(self.peek_tok(), Token::Semi) {
                    self.advance();
                }
                continue;
            }

            // `for` range loop
            if matches!(self.peek_tok(), Token::For) {
                stmts.push(self.parse_for_stmt()?);
                if matches!(self.peek_tok(), Token::Semi) {
                    self.advance();
                }
                continue;
            }

            // `par for` parallel range loop
            if matches!(self.peek_tok(), Token::Par) {
                stmts.push(self.parse_par_for_stmt()?);
                if matches!(self.peek_tok(), Token::Semi) {
                    self.advance();
                }
                continue;
            }

            // `spawn { }` concurrent task
            if matches!(self.peek_tok(), Token::Spawn) {
                stmts.push(self.parse_spawn_stmt()?);
                if matches!(self.peek_tok(), Token::Semi) {
                    self.advance();
                }
                continue;
            }

            // `with <effects> { body }` effect mask block
            if matches!(self.peek_tok(), Token::With) {
                stmts.push(self.parse_with_stmt()?);
                if matches!(self.peek_tok(), Token::Semi) {
                    self.advance();
                }
                continue;
            }

            // `handle <expr> with { ... }` effect handler block
            if self.is_handle_stmt_at_pos() {
                stmts.push(self.parse_handle_stmt()?);
                if matches!(self.peek_tok(), Token::Semi) {
                    self.advance();
                }
                continue;
            }

            // `nursery { }` scoped concurrency block
            if matches!(self.peek_tok(), Token::Ident(name) if name == "nursery") && matches!(self.peek_next_tok(), Token::LBrace) {
                stmts.extend(self.parse_nursery_stmt()?);
                if matches!(self.peek_tok(), Token::Semi) {
                    self.advance();
                }
                continue;
            }

            // `loop` statement
            if matches!(self.peek_tok(), Token::Loop) {
                stmts.push(self.parse_loop_stmt()?);
                if matches!(self.peek_tok(), Token::Semi) {
                    self.advance();
                }
                continue;
            }

            // `break` statement
            if matches!(self.peek_tok(), Token::Break) {
                let span = self.advance().span;
                if matches!(self.peek_tok(), Token::Semi) {
                    self.advance();
                }
                stmts.push(AstStmt::Break { span });
                continue;
            }

            // `continue` statement
            if matches!(self.peek_tok(), Token::Continue) {
                let span = self.advance().span;
                if matches!(self.peek_tok(), Token::Semi) {
                    self.advance();
                }
                stmts.push(AstStmt::Continue { span });
                continue;
            }

            // `return [expr]` statement
            if matches!(self.peek_tok(), Token::Return) {
                let start_span = self.advance().span;
                // If the next token could start an expression, parse the return value.
                let value = if matches!(self.peek_tok(), Token::Semi | Token::RBrace | Token::Eof) {
                    None
                } else {
                    Some(Box::new(self.parse_expr()?))
                };
                let end_span = value.as_ref().map_or(start_span, |v| v.span());
                if matches!(self.peek_tok(), Token::Semi) {
                    self.advance();
                }
                stmts.push(AstStmt::Return {
                    value,
                    span: start_span.merge(end_span),
                });
                continue;
            }

            // Expression — either a statement (followed by `;`), an assignment, or the tail.
            let expr = self.parse_expr()?;
            let assign_op = match self.peek_tok() {
                Token::Eq => None,
                Token::PlusEq => Some(BinOp::Add),
                Token::MinusEq => Some(BinOp::Sub),
                Token::StarEq => Some(BinOp::Mul),
                Token::SlashEq => Some(BinOp::Div),
                Token::PercentEq => Some(BinOp::Mod),
                _ => None,
            };
            if assign_op.is_some() || matches!(self.peek_tok(), Token::Eq) {
                // Assignment: lvalue = value or lvalue += value, etc.
                let start_span = expr.span();
                self.advance(); // consume '=' or '+=', etc.
                let value = self.parse_expr()?;
                let end_span = value.span();
                if matches!(self.peek_tok(), Token::Semi) {
                    self.advance();
                }
                stmts.push(AstStmt::Assign {
                    target: Box::new(expr),
                    op: assign_op,
                    value: Box::new(value),
                    span: start_span.merge(end_span),
                });
            } else if matches!(self.peek_tok(), Token::Semi) {
                self.advance(); // consume `;`
                stmts.push(AstStmt::Expr(Box::new(expr)));
            } else if matches!(
                &expr,
                AstExpr::If { .. } | AstExpr::When { .. } | AstExpr::Block(_)
            ) && !matches!(self.peek_tok(), Token::RBrace | Token::Eof)
            {
                // Block-type expressions (if, when, block literal) act as implicit statements
                // when not at block end — no `;` required after their closing `}`.
                stmts.push(AstStmt::Expr(Box::new(expr)));
            } else {
                // No `;` → this is the tail expression.
                tail = Some(Box::new(expr));
                break;
            }
        }

        let end = self.expect(&Token::RBrace)?;
        Ok(AstBlock {
            stmts,
            tail,
            span: start.merge(end),
        })
    }

    fn parse_let_stmt(&mut self) -> Result<AstStmt, ParseError> {
        let start = self.current_span();
        let is_var = matches!(self.peek_tok(), Token::Var);
        self.advance(); // consume 'val', 'var', or 'let' (caller already checked)



        // Destructuring: val (a, b, ...) = expr
        if matches!(self.peek_tok(), Token::LParen) {
            self.advance(); // consume '('
            let mut names = Vec::new();
            if !matches!(self.peek_tok(), Token::RParen) {
                names.push(self.expect_ident()?);
                while matches!(self.peek_tok(), Token::Comma) {
                    self.advance();
                    if matches!(self.peek_tok(), Token::RParen) {
                        break;
                    }
                    names.push(self.expect_ident()?);
                }
            }
            self.expect(&Token::RParen)?;
            self.expect(&Token::Eq)?;
            let init = self.parse_expr()?;
            let end = if matches!(self.peek_tok(), Token::Semi) {
                self.advance().span
            } else {
                init.span()
            };
            return Ok(AstStmt::LetTuple {
                names,
                init: Box::new(init),
                is_var,
                span: start.merge(end),
            });
        }

        // Refutable pattern binding: let some(x) = expr, let none = expr, let _ = expr,
        // let ok(x) = expr, let err(e) = expr, let Enum.Variant(x) = expr
        let is_refutable = match self.peek_tok() {
            Token::Ident(ref name) if name == "_" => true,
            Token::Ident(ref name) if name == "none" => {
                !self.tokens.get(self.pos + 1).map_or(false, |s| matches!(s.node, Token::Dot))
            }
            Token::Ident(ref name) if name == "some" || name == "ok" || name == "err" => {
                self.tokens.get(self.pos + 1).map_or(false, |s| matches!(s.node, Token::LParen))
            }
            Token::Ident(_) => {
                self.tokens.get(self.pos + 1).map_or(false, |s| matches!(s.node, Token::Dot))
            }
            _ => false,
        };

        if is_refutable {
            if is_var {
                return Err(ParseError::UnexpectedToken {
                    expected: "val (mutable refutable bindings are not supported)".to_owned(),
                    found: "var".to_owned(),
                    span: self.current_span(),
                });
            }
            let pat_start = self.current_span();
            let pattern = self.parse_when_sub_pattern()?;
            let pat_end = self.current_span();
            let pat_span = pat_start.merge(pat_end);
            self.expect(&Token::Eq)?;
            let scrutinee = self.parse_expr()?;
            let end = if matches!(self.peek_tok(), Token::Semi) {
                self.advance().span
            } else {
                scrutinee.span()
            };
            let span = start.merge(end);

            // Build panic call: panic("let pattern mismatch: expected <description>")
            let desc = match &pattern {
                AstWhenPattern::OptionSome { .. } => "some".to_string(),
                AstWhenPattern::OptionNone => "none".to_string(),
                AstWhenPattern::ResultOk { .. } => "ok".to_string(),
                AstWhenPattern::ResultErr { .. } => "err".to_string(),
                AstWhenPattern::Wildcard => "_".to_string(),
                AstWhenPattern::EnumVariant { enum_name, variant_name, .. } => {
                    if enum_name.is_empty() {
                        variant_name.clone()
                    } else {
                        format!("{}.{}", enum_name.replace("__", "."), variant_name)
                    }
                }
                _ => "pattern".to_string(),
            };
            let panic_msg = format!("let pattern mismatch: expected {}", desc);
            let panic_expr = AstExpr::Call {
                callee: Ident { name: "panic".to_string(), span },
                args: vec![AstExpr::StringLit { value: panic_msg, span }],
                span,
            };

            // Determine the binding name from the pattern (before moving pattern into arms)
            let binding_name = match &pattern {
                AstWhenPattern::OptionSome { binding: Some(name) } => Some(name.clone()),
                AstWhenPattern::ResultOk { binding: Some(name) } => Some(name.clone()),
                AstWhenPattern::ResultErr { binding: Some(name) } => Some(name.clone()),
                AstWhenPattern::EnumVariant { bindings, .. } if !bindings.is_empty() => {
                    Some(bindings[0].clone())
                }
                _ => None,
            };

            // Build success body: extract the bound value
            let success_body = match &pattern {
                AstWhenPattern::OptionSome { binding: Some(name) }
                | AstWhenPattern::ResultOk { binding: Some(name) }
                | AstWhenPattern::ResultErr { binding: Some(name) } => {
                    AstExpr::Ident(Ident { name: name.clone(), span: pat_span })
                }
                AstWhenPattern::EnumVariant { bindings, .. } if !bindings.is_empty() => {
                    if bindings.len() == 1 {
                        AstExpr::Ident(Ident { name: bindings[0].clone(), span: pat_span })
                    } else {
                        let elements: Vec<AstExpr> = bindings.iter()
                            .map(|b| AstExpr::Ident(Ident { name: b.clone(), span: pat_span }))
                            .collect();
                        AstExpr::Tuple { elements, span: pat_span }
                    }
                }
                _ => AstExpr::IntLit { value: 0, span },
            };

            // Build matching complement pattern for option/result to avoid
            // 'partial' mode in lowerer (which discards the arm body value)
            let failure_pattern = match &pattern {
                AstWhenPattern::OptionSome { .. } => AstWhenPattern::OptionNone,
                AstWhenPattern::OptionNone => AstWhenPattern::OptionSome { binding: None },
                AstWhenPattern::ResultOk { .. } => AstWhenPattern::ResultErr { binding: None },
                AstWhenPattern::ResultErr { .. } => AstWhenPattern::ResultOk { binding: None },
                _ => AstWhenPattern::Wildcard,
            };
            let failure_name = match &failure_pattern {
                AstWhenPattern::OptionNone => "none",
                AstWhenPattern::OptionSome { .. } => "some",
                AstWhenPattern::ResultOk { .. } => "ok",
                AstWhenPattern::ResultErr { .. } => "err",
                _ => "_",
            };

            let arms = vec![
                AstWhenArm {
                    pattern,
                    guard: None,
                    body: Box::new(success_body),
                    span: pat_span,
                    enum_name: String::new(),
                    variant_name: String::new(),
                },
                AstWhenArm {
                    pattern: failure_pattern,
                    guard: None,
                    body: Box::new(panic_expr),
                    span,
                    enum_name: failure_name.to_string(),
                    variant_name: failure_name.to_string(),
                },
            ];

            let when_expr = AstExpr::When {
                scrutinee: Box::new(scrutinee),
                arms,
                span,
            };

            return match binding_name {
                Some(name) => Ok(AstStmt::Let {
                    name: Ident { name, span: pat_span },
                    ty: None,
                    init: Box::new(when_expr),
                    is_var: false,
                    span,
                }),
                None => Ok(AstStmt::Expr(Box::new(when_expr))),
            };
        }

        let name = self.expect_ident()?;
        let ty = if matches!(self.peek_tok(), Token::Colon) {
            self.advance();
            Some(self.parse_type()?)
        } else {
            None
        };
        self.expect(&Token::Eq)?;
        let init = self.parse_expr()?;
        // Semicolon is optional after `val` to support both styles:
        //   val x = expr;   (explicit terminator)
        //   val x = expr    (newline-terminated, block-expression style)
        let end = if matches!(self.peek_tok(), Token::Semi) {
            self.advance().span
        } else {
            init.span()
        };
        Ok(AstStmt::Let {
            name,
            ty,
            init: Box::new(init),
            is_var,
            span: start.merge(end),
        })
    }

    // -----------------------------------------------------------------------
    // Expressions (precedence climbing)
    // -----------------------------------------------------------------------

    fn parse_expr(&mut self) -> Result<AstExpr, ParseError> {
        self.parse_or_expr()
    }

    fn parse_or_expr(&mut self) -> Result<AstExpr, ParseError> {
        let mut lhs = self.parse_and_expr()?;
        loop {
            if !matches!(self.peek_tok(), Token::PipePipe) {
                break;
            }
            self.advance();
            let rhs = self.parse_and_expr()?;
            let span = lhs.span().merge(rhs.span());
            lhs = AstExpr::BinOp {
                op: AstBinOp::Or,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
                span,
            };
        }
        Ok(lhs)
    }

    fn parse_and_expr(&mut self) -> Result<AstExpr, ParseError> {
        let mut lhs = self.parse_cmp_expr()?;
        loop {
            if !matches!(self.peek_tok(), Token::AmpAmp) {
                break;
            }
            self.advance();
            let rhs = self.parse_cmp_expr()?;
            let span = lhs.span().merge(rhs.span());
            lhs = AstExpr::BinOp {
                op: AstBinOp::And,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
                span,
            };
        }
        Ok(lhs)
    }

    fn parse_add_expr(&mut self) -> Result<AstExpr, ParseError> {
        let mut lhs = self.parse_mul_expr()?;
        loop {
            let op = match self.peek_tok() {
                Token::Plus => AstBinOp::Add,
                Token::Minus => AstBinOp::Sub,
                _ => break,
            };
            self.advance();
            let rhs = self.parse_mul_expr()?;
            let span = lhs.span().merge(rhs.span());
            lhs = AstExpr::BinOp {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
                span,
            };
        }
        Ok(lhs)
    }

    fn parse_mul_expr(&mut self) -> Result<AstExpr, ParseError> {
        let mut lhs = self.parse_cast_expr()?;
        loop {
            let op = match self.peek_tok() {
                Token::Star => AstBinOp::Mul,
                Token::Slash => AstBinOp::Div,
                Token::Percent => AstBinOp::Mod,
                _ => break,
            };
            self.advance();
            let rhs = self.parse_cast_expr()?;
            let span = lhs.span().merge(rhs.span());
            lhs = AstExpr::BinOp {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
                span,
            };
        }
        Ok(lhs)
    }

    /// Parses a cmp expression, then checks for a postfix `to Type` cast.
    fn parse_cast_expr(&mut self) -> Result<AstExpr, ParseError> {
        let mut expr = self.parse_unary()?;
        while matches!(self.peek_tok(), Token::To) {
            let start = expr.span();
            self.advance(); // consume 'to'
            let ty = self.parse_type()?;
            let end = ty.span();
            expr = AstExpr::Cast {
                expr: Box::new(expr),
                ty,
                span: start.merge(end),
            };
        }
        Ok(expr)
    }

    fn parse_cmp_expr(&mut self) -> Result<AstExpr, ParseError> {
        let mut lhs = self.parse_add_expr()?;
        loop {
            let op = match self.peek_tok() {
                Token::EqEq => AstBinOp::CmpEq,
                Token::NotEq => AstBinOp::CmpNe,
                Token::LtGt => AstBinOp::CmpNe,  // <> as alias for !=
                Token::LAngle => AstBinOp::CmpLt,
                Token::LtEq => AstBinOp::CmpLe,
                Token::RAngle => AstBinOp::CmpGt,
                Token::GtEq => AstBinOp::CmpGe,
                _ => break,
            };
            self.advance();
            let rhs = self.parse_add_expr()?;
            let span = lhs.span().merge(rhs.span());
            lhs = AstExpr::BinOp {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
                span,
            };
        }
        Ok(lhs)
    }

    fn parse_unary(&mut self) -> Result<AstExpr, ParseError> {
        let span = self.current_span();
        if matches!(self.peek_tok(), Token::Minus) {
            self.advance();
            let expr = self.parse_unary()?;
            let end = expr.span();
            return Ok(AstExpr::UnaryOp {
                op: AstUnaryOp::Neg,
                expr: Box::new(expr),
                span: span.merge(end),
            });
        }
        if matches!(self.peek_tok(), Token::Bang) {
            self.advance();
            let expr = self.parse_unary()?;
            let end = expr.span();
            return Ok(AstExpr::UnaryOp {
                op: AstUnaryOp::Not,
                expr: Box::new(expr),
                span: span.merge(end),
            });
        }
        // Handle await expression
        if matches!(self.peek_tok(), Token::Await) {
            self.advance();
            let inner = self.parse_unary()?;
            let end = inner.span();
            return Ok(AstExpr::Await {
                expr: Box::new(inner),
                span: span.merge(end),
            });
        }
        self.parse_primary()
    }

    fn parse_primary(&mut self) -> Result<AstExpr, ParseError> {
        let span = self.current_span();

        let mut expr = match self.peek_tok().clone() {
            Token::Ident(ref name) if name == "handle" && self.is_handle_stmt_at_pos() => {
                self.advance();
                let expr = self.parse_expr()?;
                self.expect(&Token::With)?;
                self.expect(&Token::LBrace)?;
                let mut arms = Vec::new();
                let return_ty = Box::new(AstType::Named("Infer".to_string(), span));
                loop {
                    if matches!(self.peek_tok(), Token::RBrace | Token::Eof) {
                        break;
                    }
                    let arm_start = self.current_span();
                    let effect_name = self.expect_ident()?.name;
                    let mut params = Vec::new();
                    if matches!(self.peek_tok(), Token::LParen) {
                        self.advance();
                        loop {
                            if matches!(self.peek_tok(), Token::RParen | Token::Eof) {
                                break;
                            }
                            let p = self.expect_ident()?;
                            params.push(p);
                            if matches!(self.peek_tok(), Token::Comma) {
                                self.advance();
                                continue;
                            }
                            break;
                        }
                        self.expect(&Token::RParen)?;
                    }
                    let mut resume_param = None;
                    if matches!(self.peek_tok(), Token::Arrow) {
                        self.advance();
                        if matches!(self.peek_tok(), Token::Resume) || matches!(self.peek_tok(), Token::Ident(ref name) if name == "resume") {
                            self.advance();
                            if matches!(self.peek_tok(), Token::LParen) {
                                self.advance();
                                let rp = self.expect_ident()?;
                                self.expect(&Token::RParen)?;
                                resume_param = Some(rp);
                            }
                        } else {
                            return Err(ParseError::UnexpectedToken {
                                expected: "'resume'".to_owned(),
                                found: format!("{}", self.peek_tok()),
                                span: self.current_span(),
                            });
                        }
                    }
                    self.expect(&Token::FatArrow)?;
                    let body = self.parse_expr()?;
                    let arm_span = arm_start.merge(body.span());
                    arms.push(AstHandlerArm {
                        effect_name,
                        params,
                        resume_param,
                        body: Box::new(body),
                        span: arm_span,
                    });
                    if matches!(self.peek_tok(), Token::Comma) {
                        self.advance();
                        continue;
                    }
                }
                let end = self.expect(&Token::RBrace)?;
                AstExpr::Handle {
                    expr: Box::new(expr),
                    arms,
                    return_ty,
                    span: span.merge(end),
                }
            }
            Token::Ident(name) => {
                let ident_span = self.advance().span;
                let ident = Ident {
                    name: name.clone(),
                    span: ident_span,
                };
                // Struct literal: Name { field: expr, ... }
                // Disambiguate from `ident` followed by a block expression by
                // checking: after `{`, the content is either `}` (empty struct)
                // or `Ident :` (field initializer). Any other form is not a
                // struct literal.
                let is_struct_lit = matches!(self.peek_tok(), Token::LBrace)
                    && (matches!(self.peek_next_tok(), Token::RBrace) // Name {}
                        || (matches!(self.peek_next_tok(), Token::Ident(_))
                            && matches!(self.peek_at(2), Token::Colon))); // Name { field: ...}
                if is_struct_lit {
                    self.advance(); // consume '{'
                    let mut fields = Vec::new();
                    while !matches!(self.peek_tok(), Token::RBrace | Token::Eof) {
                        let field_name = self.expect_ident()?;
                        self.expect(&Token::Colon)?;
                        let val = self.parse_expr()?;
                        fields.push((field_name.name, val));
                        if matches!(self.peek_tok(), Token::Comma) {
                            self.advance();
                        }
                    }
                    let end = self.expect(&Token::RBrace)?;
                    AstExpr::StructLit {
                        name,
                        fields,
                        span: ident_span.merge(end),
                    }
                } else if matches!(self.peek_tok(), Token::LParen) {
                    // Function call
                    self.advance(); // consume '('
                    let args = self.parse_call_args()?;
                    let end = self.expect(&Token::RParen)?;
                    AstExpr::Call {
                        callee: ident,
                        args,
                        span: ident_span.merge(end),
                    }
                } else {
                    AstExpr::Ident(ident)
                }
            }

            Token::IntLit(n) => {
                self.advance();
                AstExpr::IntLit { value: n, span }
            }

            Token::FloatLit(v) => {
                self.advance();
                AstExpr::FloatLit { value: v, span }
            }

            Token::BoolLit(b) => {
                self.advance();
                AstExpr::BoolLit { value: b, span }
            }

            Token::StringLit(s) => {
                self.advance();
                AstExpr::StringLit { value: s, span }
            }

            Token::FStringLit(raw) => {
                let raw = raw.clone();
                self.advance();
                self.desugar_fstring(&raw, span)
            }

            Token::LParen => {
                self.advance(); // consume '('
                let first = self.parse_expr()?;
                if matches!(self.peek_tok(), Token::Comma) {
                    // Tuple literal: (expr, expr, ...)
                    let mut elements = vec![first];
                    while matches!(self.peek_tok(), Token::Comma) {
                        self.advance();
                        if matches!(self.peek_tok(), Token::RParen) {
                            break; // trailing comma
                        }
                        elements.push(self.parse_expr()?);
                    }
                    let end = self.expect(&Token::RParen)?;
                    AstExpr::Tuple {
                        elements,
                        span: span.merge(end),
                    }
                } else {
                    // Grouping: (expr)
                    self.expect(&Token::RParen)?;
                    first
                }
            }

            Token::If => {
                self.advance(); // consume 'if'
                // Check for `if let pattern = expr { body }`
                if matches!(self.peek_tok(), Token::Let) {
                    self.advance(); // consume 'let'
                    // Parse the pattern (reuse when-arm sub-pattern logic)
                    let pattern_start = self.current_span();
                    let first_name = self.expect_ident()?.name;
                    let pattern = if first_name == "_" {
                        AstWhenPattern::Wildcard
                    } else if (first_name == "some" || first_name == "ok" || first_name == "err")
                        && matches!(self.peek_tok(), Token::LParen)
                    {
                        self.advance(); // consume '('
                        let binding = if matches!(self.peek_tok(), Token::RParen) {
                            None
                        } else {
                            Some(self.expect_ident()?.name)
                        };
                        self.expect(&Token::RParen)?;
                        if first_name == "some" {
                            AstWhenPattern::OptionSome { binding }
                        } else if first_name == "ok" {
                            AstWhenPattern::ResultOk { binding }
                        } else {
                            AstWhenPattern::ResultErr { binding }
                        }
                    } else if first_name == "none" && !matches!(self.peek_tok(), Token::Dot) {
                        AstWhenPattern::OptionNone
                    } else {
                        return Err(ParseError::UnexpectedToken {
                            expected: "pattern (some(x), none, ok(x), err(e), or _)".to_owned(),
                            found: first_name.clone(),
                            span: self.current_span(),
                        });
                    };
                    let pattern_end = self.current_span();
                    let pat_span = pattern_start.merge(pattern_end);
                    self.expect(&Token::Eq)?; // consume '='
                    let scrutinee = self.parse_expr()?;
                    let mut then_block = self.parse_block()?;
                    // Ensure then_block has a tail expression for block expression wrapping.
                    if then_block.tail.is_none() {
                        then_block.tail = Some(Box::new(AstExpr::IntLit { value: 0, span: then_block.span }));
                    }
                    let (else_block, end_span) = if matches!(self.peek_tok(), Token::Else) {
                        self.advance(); // consume 'else'
                        if matches!(self.peek_tok(), Token::If) {
                            // `else if` → desugar to `else { if ... }`
                            let elif_span_start = self.current_span();
                            let elif_expr = self.parse_primary()?;
                            let elif_span = elif_expr.span();
                            let eb = AstBlock {
                                stmts: vec![],
                                tail: Some(Box::new(elif_expr)),
                                span: elif_span_start.merge(elif_span),
                            };
                            (Some(eb), elif_span_start.merge(elif_span))
                        } else {
                            let eb = self.parse_block()?;
                            let es = eb.span;
                            (Some(eb), es)
                        }
                    } else {
                        (None, then_block.span)
                    };
                    // Desugar to `when scrutinee { pattern => then, _ => else }`
                    let when_span = span.merge(end_span);
                    let mut arms = Vec::new();
                    let then_tail = AstExpr::Block(AstBlock {
                        stmts: vec![],
                        tail: Some(Box::new(AstExpr::Block(then_block))),
                        span: when_span,
                    });
                    arms.push(AstWhenArm {
                        pattern,
                        guard: None,
                        body: Box::new(then_tail),
                        span: pat_span,
                        enum_name: String::new(),
                        variant_name: String::new(),
                    });
                    let else_body = if let Some(mut eb) = else_block {
                        if eb.tail.is_none() {
                            eb.tail = Some(Box::new(AstExpr::IntLit { value: 0, span: eb.span }));
                        }
                        AstExpr::Block(eb)
                    } else {
                        AstExpr::IntLit { value: 0, span: when_span }
                    };
                    arms.push(AstWhenArm {
                        pattern: AstWhenPattern::Wildcard,
                        guard: None,
                        body: Box::new(else_body),
                        span: when_span,
                        enum_name: "_".to_string(),
                        variant_name: "_".to_string(),
                    });
                    AstExpr::When {
                        scrutinee: Box::new(scrutinee),
                        arms,
                        span: when_span,
                    }
                } else {
                    let cond = self.parse_expr()?;
                    let then_block = self.parse_block()?;
                    let (else_block, end_span) = if matches!(self.peek_tok(), Token::Else) {
                        self.advance(); // consume 'else'
                        if matches!(self.peek_tok(), Token::If) {
                            // Desugar `else if cond { .. }` as `else { if cond { .. } }`
                            let elif_span_start = self.current_span();
                            let elif_expr = self.parse_primary()?;
                            let elif_span = elif_expr.span();
                            let eb = AstBlock {
                                stmts: vec![],
                                tail: Some(Box::new(elif_expr)),
                                span: elif_span_start.merge(elif_span),
                            };
                            let es = eb.span;
                            (Some(eb), es)
                        } else {
                            let eb = self.parse_block()?;
                            let es = eb.span;
                            (Some(eb), es)
                        }
                    } else {
                        (None, then_block.span)
                    };
                    AstExpr::If {
                        cond: Box::new(cond),
                        then_block,
                        else_block,
                        span: span.merge(end_span),
                    }
                }
            }

            Token::LBrace => {
                let block = self.parse_block()?;
                AstExpr::Block(block)
            }

            Token::With => {
                self.advance();
                let mut effects = Vec::new();
                loop {
                    let name = self.expect_ident()?.name;
                    effects.push(name);
                    if matches!(self.peek_tok(), Token::Comma) {
                        self.advance();
                        continue;
                    }
                    break;
                }
                let body = self.parse_block()?;
                let end = body.span;
                AstExpr::Mask {
                    effects,
                    body,
                    span: span.merge(end),
                }
            }


            Token::LBracket => {
                // Array literal: [expr, expr, ...]
                self.advance(); // consume '['
                let mut elems = Vec::new();
                if !matches!(self.peek_tok(), Token::RBracket) {
                    elems.push(self.parse_expr()?);
                    while matches!(self.peek_tok(), Token::Comma) {
                        self.advance();
                        if matches!(self.peek_tok(), Token::RBracket) {
                            break;
                        }
                        elems.push(self.parse_expr()?);
                    }
                }
                let end = self.expect(&Token::RBracket)?;
                AstExpr::ArrayLit {
                    elems,
                    span: span.merge(end),
                }
            }

            Token::Pipe => {
                // Lambda: |param: type, ...| body_expr
                self.advance(); // consume opening '|'
                let mut params = Vec::new();
                while !matches!(self.peek_tok(), Token::Pipe | Token::Eof) {
                    let name = self.expect_ident()?;
                    self.expect(&Token::Colon)?;
                    let ty = self.parse_type()?;
                    params.push(AstParam {
                        name,
                        ty,
                        default: None,
                    });
                    if matches!(self.peek_tok(), Token::Comma) {
                        self.advance();
                    }
                }
                self.expect(&Token::Pipe)?; // consume closing '|'
                let body = self.parse_expr()?;
                let end = body.span();
                AstExpr::Lambda {
                    params,
                    body: Box::new(body),
                    span: span.merge(end),
                }
            }

            Token::When => {
                self.advance(); // consume 'when'
                let scrutinee = self.parse_expr()?;
                self.expect(&Token::LBrace)?;
                let mut arms = Vec::new();
                while !matches!(self.peek_tok(), Token::RBrace | Token::Eof) {
                    let arm_start = self.current_span();
                    // Peek BEFORE consuming to handle literal/wildcard patterns.
                    let (pattern, enum_name_leg, variant_name_leg) = match self.peek_tok() {
                        Token::IntLit(n) => {
                            let n = *n;
                            self.advance(); // consume int literal
                                            // Check for inclusive range pattern: lo..=hi
                            if matches!(self.peek_tok(), Token::DotDotEq) {
                                self.advance(); // consume '..='
                                let hi = match self.peek_tok().clone() {
                                    Token::IntLit(h) => {
                                        self.advance();
                                        h
                                    }
                                    _ => {
                                        return Err(ParseError::UnexpectedToken {
                                            expected: "integer for range upper bound".to_owned(),
                                            found: format!("{}", self.peek_tok()),
                                            span: self.current_span(),
                                        })
                                    }
                                };
                                (
                                    AstWhenPattern::Range { lo: n, hi },
                                    "_range".to_string(),
                                    format!("{}..={}", n, hi),
                                )
                            } else {
                                (AstWhenPattern::IntLit(n), "_lit".to_string(), n.to_string())
                            }
                        }
                        Token::BoolLit(b) => {
                            let b = *b;
                            self.advance(); // consume bool literal
                            (
                                AstWhenPattern::BoolLit(b),
                                "_lit".to_string(),
                                b.to_string(),
                            )
                        }
                        Token::StringLit(_) => {
                            let s = if let Token::StringLit(s) = self.peek_tok() {
                                s.clone()
                            } else {
                                unreachable!()
                            };
                            self.advance(); // consume string literal
                            (AstWhenPattern::StringLit(s.clone()), "_lit".to_string(), s)
                        }
                        Token::LParen => {
                            // Tuple pattern: (sub, sub, ...)
                            self.advance(); // consume '('
                            let mut subs = Vec::new();
                            while !matches!(self.peek_tok(), Token::RParen | Token::Eof) {
                                let sub = self.parse_when_sub_pattern()?;
                                subs.push(sub);
                                if matches!(self.peek_tok(), Token::Comma) {
                                    self.advance();
                                }
                            }
                            self.expect(&Token::RParen)?;
                            (
                                AstWhenPattern::Tuple(subs),
                                "_tuple".to_string(),
                                "_tuple".to_string(),
                            )
                        }
                        Token::LBracket => {
                            // Slice pattern: [a, b, ..rest]
                            self.advance(); // consume '['
                            let mut prefix = Vec::new();
                            let mut rest = None;
                            while !matches!(self.peek_tok(), Token::RBracket | Token::Eof) {
                                if matches!(self.peek_tok(), Token::DotDot) {
                                    self.advance(); // consume '..'
                                    if matches!(self.peek_tok(), Token::Ident(_)) {
                                        rest = Some(self.expect_ident()?.name);
                                    }
                                    break;
                                }
                                let sub = self.parse_when_sub_pattern()?;
                                prefix.push(sub);
                                if matches!(self.peek_tok(), Token::Comma) {
                                    self.advance();
                                }
                            }
                            self.expect(&Token::RBracket)?;
                            (
                                AstWhenPattern::Slice { prefix, rest },
                                "_slice".to_string(),
                                "_slice".to_string(),
                            )
                        }
                        _ => {
                            // Peek at ident to determine pattern type.
                            let first_name = self.expect_ident()?.name;
                            if first_name == "_" {
                                // Wildcard pattern.
                                (AstWhenPattern::Wildcard, "_".to_string(), "_".to_string())
                            } else if (first_name == "some"
                                || first_name == "ok"
                                || first_name == "err")
                                && matches!(self.peek_tok(), Token::LParen)
                            {
                                // `some(x)` / `ok(x)` / `err(e)` — consume `(binding)`
                                self.advance(); // consume '('
                                let binding = if matches!(self.peek_tok(), Token::RParen) {
                                    None
                                } else {
                                    Some(self.expect_ident()?.name)
                                };
                                self.expect(&Token::RParen)?;
                                let pat = if first_name == "some" {
                                    AstWhenPattern::OptionSome {
                                        binding: binding.clone(),
                                    }
                                } else if first_name == "ok" {
                                    AstWhenPattern::ResultOk {
                                        binding: binding.clone(),
                                    }
                                } else {
                                    AstWhenPattern::ResultErr {
                                        binding: binding.clone(),
                                    }
                                };
                                (pat, first_name.clone(), binding.unwrap_or_default())
                            } else if first_name == "none" && !matches!(self.peek_tok(), Token::Dot)
                            {
                                // `none` pattern (no dot follows)
                                (
                                    AstWhenPattern::OptionNone,
                                    "none".to_string(),
                                    "none".to_string(),
                                )
                            } else {
                                // `EnumName.Variant` or `EnumName.Variant(a, b, ...)` — enum pattern
                                let mut parts = vec![first_name.clone()];
                                while matches!(self.peek_tok(), Token::Dot) {
                                    self.advance(); // consume '.'
                                    let sub = self.expect_ident()?.name;
                                    parts.push(sub);
                                }
                                if parts.len() < 2 {
                                    return Err(ParseError::UnexpectedToken {
                                        expected: "enum variant name pattern (e.g. Enum.Variant)"
                                            .to_owned(),
                                        found: format!("{}", self.peek_tok()),
                                        span: self.current_span(),
                                    });
                                }
                                let variant_name = parts.pop().unwrap();
                                let enum_name = parts.join("__");
                                // Optionally parse data bindings: `Variant(a, b, ...)`
                                let bindings = if matches!(self.peek_tok(), Token::LParen) {
                                    self.advance(); // consume '('
                                    let mut names = Vec::new();
                                    while !matches!(self.peek_tok(), Token::RParen | Token::Eof) {
                                        names.push(self.expect_ident()?.name);
                                        if matches!(self.peek_tok(), Token::Comma) {
                                            self.advance();
                                        }
                                    }
                                    self.expect(&Token::RParen)?;
                                    names
                                } else {
                                    Vec::new()
                                };
                                let pat = AstWhenPattern::EnumVariant {
                                    enum_name: enum_name.clone(),
                                    variant_name: variant_name.clone(),
                                    bindings,
                                };
                                (pat, enum_name, variant_name)
                            }
                        }
                    };
                        // Support or-patterns: `pat1 | pat2 | ...`
                        let mut patterns = vec![pattern];
                        while matches!(self.peek_tok(), Token::Pipe) {
                            self.advance(); // consume '|'
                            // Parse the next full pattern
                            let next_pat = self.parse_when_pattern()?;
                            patterns.push(next_pat);
                        }
                        // Combine into Or if multiple patterns
                        let pattern = if patterns.len() == 1 {
                            patterns.pop().unwrap()
                        } else {
                            AstWhenPattern::Or(patterns)
                        };
                    // Optional guard: `pattern if expr =>`
                    let guard = if matches!(self.peek_tok(), Token::If) {
                        self.advance(); // consume 'if'
                        Some(Box::new(self.parse_expr()?))
                    } else {
                        None
                    };
                    self.expect(&Token::FatArrow)?;
                    let body = self.parse_expr()?;
                    let arm_end = body.span();
                    // Optional comma between arms
                    if matches!(self.peek_tok(), Token::Comma) {
                        self.advance();
                    }
                    arms.push(AstWhenArm {
                        pattern,
                        guard,
                        enum_name: enum_name_leg,
                        variant_name: variant_name_leg,
                        body: Box::new(body),
                        span: arm_start.merge(arm_end),
                    });
                }
                let end = self.expect(&Token::RBrace)?;
                AstExpr::When {
                    scrutinee: Box::new(scrutinee),
                    arms,
                    span: span.merge(end),
                }
            }

            _ => {
                return Err(ParseError::UnexpectedToken {
                    expected: "expression".to_owned(),
                    found: format!("{}", self.peek_tok()),
                    span,
                });
            }
        };

        // Postfix: index expr[i, j, ...] or field access expr.field
        loop {
            if matches!(self.peek_tok(), Token::LBracket) {
                let start = expr.span();
                self.advance(); // consume '['
                let mut indices = Vec::new();
                if !matches!(self.peek_tok(), Token::RBracket) {
                    indices.push(self.parse_expr()?);
                    while matches!(self.peek_tok(), Token::Comma) {
                        self.advance();
                        if matches!(self.peek_tok(), Token::RBracket) {
                            break;
                        }
                        indices.push(self.parse_expr()?);
                    }
                }
                let end = self.expect(&Token::RBracket)?;
                expr = AstExpr::Index {
                    base: Box::new(expr),
                    indices,
                    span: start.merge(end),
                };
            } else if matches!(self.peek_tok(), Token::Dot) {
                let start = expr.span();
                self.advance(); // consume '.'
                                // Tuple index access: expr.0, expr.1, ...
                if let Token::IntLit(n) = self.peek_tok().clone() {
                    let end = self.advance().span;
                    expr = AstExpr::TupleIndex {
                        base: Box::new(expr),
                        index: n as usize,
                        span: start.merge(end),
                    };
                } else {
                    let field = self.expect_ident()?;
                    // Method call: expr.method(args...)
                    if matches!(self.peek_tok(), Token::LParen) {
                        self.advance(); // consume '('
                        let args = self.parse_call_args()?;
                        let end = self.expect(&Token::RParen)?;
                        expr = AstExpr::MethodCall {
                            base: Box::new(expr),
                            method: field.name,
                            args,
                            span: start.merge(end),
                        };
                    } else {
                        let end = field.span;
                        let is_struct_lit = matches!(self.peek_tok(), Token::LBrace)
                            && (matches!(self.peek_next_tok(), Token::RBrace)
                                || (matches!(self.peek_next_tok(), Token::Ident(_))
                                    && matches!(self.peek_at(2), Token::Colon)));
                        if is_struct_lit {
                            self.advance(); // consume '{'
                            let mut fields = Vec::new();
                            while !matches!(self.peek_tok(), Token::RBrace | Token::Eof) {
                                let field_name = self.expect_ident()?;
                                self.expect(&Token::Colon)?;
                                let val = self.parse_expr()?;
                                fields.push((field_name.name, val));
                                if matches!(self.peek_tok(), Token::Comma) {
                                    self.advance();
                                }
                            }
                            let end_brace = self.expect(&Token::RBrace)?;
                            if let AstExpr::Ident(ref base_ident) = expr {
                                expr = AstExpr::StructLit {
                                    name: format!("{}__{}", base_ident.name, field.name),
                                    fields,
                                    span: start.merge(end_brace),
                                };
                            } else {
                                expr = AstExpr::FieldAccess {
                                    base: Box::new(expr),
                                    field: field.name,
                                    span: start.merge(end),
                                };
                            }
                        } else {
                            expr = AstExpr::FieldAccess {
                                base: Box::new(expr),
                                field: field.name,
                                span: start.merge(end),
                            };
                        }
                    }
                }
            } else if matches!(self.peek_tok(), Token::Question) {
                let end = self.advance().span; // consume '?'
                let start = expr.span();
                expr = AstExpr::Try {
                    expr: Box::new(expr),
                    span: start.merge(end),
                };
            } else {
                break;
            }
        }

        Ok(expr)
    }

    fn parse_while_stmt(&mut self) -> Result<AstStmt, ParseError> {
        let start = self.current_span();
        self.expect(&Token::While)?;
        // Check for `while let pattern = expr { body }`
        if matches!(self.peek_tok(), Token::Let) {
            self.advance(); // consume 'let'
            // Parse the pattern
            let first_name = self.expect_ident()?.name;
            let pattern = if first_name == "_" {
                AstWhenPattern::Wildcard
            } else if (first_name == "some" || first_name == "ok" || first_name == "err")
                && matches!(self.peek_tok(), Token::LParen)
            {
                self.advance(); // consume '('
                let binding = if matches!(self.peek_tok(), Token::RParen) {
                    None
                } else {
                    Some(self.expect_ident()?.name)
                };
                self.expect(&Token::RParen)?;
                if first_name == "some" {
                    AstWhenPattern::OptionSome { binding }
                } else if first_name == "ok" {
                    AstWhenPattern::ResultOk { binding }
                } else {
                    AstWhenPattern::ResultErr { binding }
                }
            } else if first_name == "none" && !matches!(self.peek_tok(), Token::Dot) {
                AstWhenPattern::OptionNone
            } else {
                return Err(ParseError::UnexpectedToken {
                    expected: "pattern (some(x), none, ok(x), err(e), or _)".to_owned(),
                    found: first_name.clone(),
                    span: self.current_span(),
                });
            };
            self.expect(&Token::Eq)?; // consume '='
            let scrutinee = self.parse_expr()?;
            let mut body = self.parse_block()?;
            // Ensure body has a tail expression for the block expression wrapping.
            if body.tail.is_none() {
                body.tail = Some(Box::new(AstExpr::IntLit { value: 0, span: body.span }));
            }
            let span = start.merge(body.span);
            // Desugar to `loop { when scrutinee { pattern => body, _ => { break; 0 } } }`
            let break_stmt = AstStmt::Break { span };
            let break_block = AstBlock {
                stmts: vec![break_stmt],
                tail: Some(Box::new(AstExpr::IntLit { value: 0, span })),
                span,
            };
            let when_arms = vec![
                AstWhenArm {
                    pattern,
                    guard: None,
                    body: Box::new(AstExpr::Block(AstBlock {
                        stmts: vec![],
                        tail: Some(Box::new(AstExpr::Block(body))),
                        span,
                    })),
                    span,
                    enum_name: String::new(),
                    variant_name: String::new(),
                },
                AstWhenArm {
                    pattern: AstWhenPattern::Wildcard,
                    guard: None,
                    body: Box::new(AstExpr::Block(break_block)),
                    span,
                    enum_name: "_".to_string(),
                    variant_name: "_".to_string(),
                },
            ];
            let when_expr = AstExpr::When {
                scrutinee: Box::new(scrutinee),
                arms: when_arms,
                span,
            };
            let loop_body = AstBlock {
                stmts: vec![],
                tail: Some(Box::new(when_expr)),
                span,
            };
            Ok(AstStmt::Loop {
                body: loop_body,
                span,
            })
        } else {
            let cond = self.parse_expr()?;
            let body = self.parse_block()?;
            let span = start.merge(body.span);
            Ok(AstStmt::While {
                cond: Box::new(cond),
                body,
                span,
            })
        }
    }

    fn parse_for_stmt(&mut self) -> Result<AstStmt, ParseError> {
        let start = self.current_span();
        self.expect(&Token::For)?;
        let var = self.expect_ident()?;
        self.expect(&Token::In)?;
        let iter_expr = self.parse_expr()?;
        // If the next token is `..` or `..=`, it's a range loop; otherwise it's a foreach loop.
        let range_inclusive = matches!(self.peek_tok(), Token::DotDotEq);
        if matches!(self.peek_tok(), Token::DotDot) || range_inclusive {
            if range_inclusive {
                self.expect(&Token::DotDotEq)?;
            } else {
                self.expect(&Token::DotDot)?;
            }
            let range_end = self.parse_expr()?;
            let body = self.parse_block()?;
            let span = start.merge(body.span);
            Ok(AstStmt::ForRange {
                var,
                start: Box::new(iter_expr),
                end: Box::new(range_end),
                inclusive: range_inclusive,
                body,
                span,
            })
        } else {
            let body = self.parse_block()?;
            let span = start.merge(body.span);
            Ok(AstStmt::ForEach {
                var,
                iter: Box::new(iter_expr),
                body,
                span,
            })
        }
    }

    fn parse_loop_stmt(&mut self) -> Result<AstStmt, ParseError> {
        let start = self.current_span();
        self.expect(&Token::Loop)?;
        let body = self.parse_block()?;
        let span = start.merge(body.span);
        Ok(AstStmt::Loop { body, span })
    }

    fn parse_spawn_stmt(&mut self) -> Result<AstStmt, ParseError> {
        let start = self.current_span();
        self.expect(&Token::Spawn)?;
        // Check for optional group: `spawn(group_expr) { body }`
        let group = if matches!(self.peek_tok(), Token::LParen) {
            self.expect(&Token::LParen)?;
            let expr = self.parse_expr()?;
            self.expect(&Token::RParen)?;
            Some(Box::new(expr))
        } else {
            None
        };
        let block = self.parse_block()?;
        let span = start.merge(block.span);
        // Collect stmts, and if the block has a tail expression, append it as a statement too.
        let mut body = block.stmts;
        if let Some(tail) = block.tail {
            body.push(AstStmt::Expr(tail));
        }
        Ok(AstStmt::Spawn { body, span, group })
    }

    fn parse_nursery_stmt(&mut self) -> Result<Vec<AstStmt>, ParseError> {
        let start = self.current_span();
        self.advance(); // consume 'nursery'
        let tg_var_name = format!("__nursery_tg_{}", start.start.0);
        let tg_ident = Ident {
            name: tg_var_name,
            span: start,
        };
        let tg_call = AstExpr::Call {
            callee: Ident {
                name: "task_group".into(),
                span: start,
            },
            args: Vec::new(),
            span: start,
        };
        let init_stmt = AstStmt::Let {
            name: tg_ident.clone(),
            ty: None,
            init: Box::new(tg_call),
            is_var: false,
            span: start,
        };
        let block = self.parse_block()?;
        let mut expanded_stmts = vec![init_stmt];
        for mut stmt in block.stmts {
            if let AstStmt::Spawn { ref mut group, .. } = stmt {
                if group.is_none() {
                    *group = Some(Box::new(AstExpr::Ident(tg_ident.clone())));
                }
            }
            expanded_stmts.push(stmt);
        }
        if let Some(tail) = block.tail {
            expanded_stmts.push(AstStmt::Expr(tail));
        }
        let join_call = AstExpr::Call {
            callee: Ident {
                name: "task_group_join".into(),
                span: start,
            },
            args: vec![AstExpr::Ident(tg_ident)],
            span: block.span,
        };
        expanded_stmts.push(AstStmt::Expr(Box::new(join_call)));
        Ok(expanded_stmts)
    }

    /// Parse `with <effect1, effect2, ...> { body }` — effect mask block.
    /// Syntax: `with` ident (',' ident)* '{' block '}'.
    fn parse_with_stmt(&mut self) -> Result<AstStmt, ParseError> {
        let start = self.expect(&Token::With)?;
        // Parse effect list: keyword, identifier, comma, identifier, etc.
        // Special case: `with pure { ... }` means an empty effect row.
        let mut effects = Vec::new();
        loop {
            let name = self.expect_ident()?.name;
            effects.push(name);
            if matches!(self.peek_tok(), Token::Comma) {
                self.advance();
                continue;
            }
            break;
        }
        let body = self.parse_block()?;
        let span = start.merge(body.span);
        Ok(AstStmt::MaskStmt {
            effects,
            body,
            span,
        })
    }

    /// Parse `handle <expr> with { arm1, arm2, ... }` — algebraic-effect handler.
    /// Each arm: `<effect_name>(<params>) -> resume(<resume_param>) => <body>`
    /// The `-> resume(...)` is optional (handler may not resume).
    fn parse_handle_stmt(&mut self) -> Result<AstStmt, ParseError> {
        let start = match self.peek_tok() {
            Token::Ident(name) if name == "handle" => {
                self.advance().span
            }
            _ => {
                return Err(ParseError::UnexpectedToken {
                    expected: "'handle'".to_owned(),
                    found: format!("{}", self.peek_tok()),
                    span: self.current_span(),
                });
            }
        };
        let expr = self.parse_expr()?;
        self.expect(&Token::With)?;
        self.expect(&Token::LBrace)?;
        let mut arms = Vec::new();
        // The return type is inferred for now — allow `: T` suffix later.
        let return_ty = Box::new(AstType::Named("Infer".to_string(), start));
        loop {
            if matches!(self.peek_tok(), Token::RBrace | Token::Eof) {
                break;
            }
            let arm_start = self.current_span();
            let effect_name = self.expect_ident()?.name;
            let mut params = Vec::new();
            if matches!(self.peek_tok(), Token::LParen) {
                self.advance();
                loop {
                    if matches!(self.peek_tok(), Token::RParen | Token::Eof) {
                        break;
                    }
                    let p = self.expect_ident()?;
                    params.push(p);
                    if matches!(self.peek_tok(), Token::Comma) {
                        self.advance();
                        continue;
                    }
                    break;
                }
                self.expect(&Token::RParen)?;
            }
            // Optional `-> resume(name)`.
            let mut resume_param = None;
            if matches!(self.peek_tok(), Token::Arrow) {
                self.advance();
                if matches!(self.peek_tok(), Token::Resume) || matches!(self.peek_tok(), Token::Ident(ref name) if name == "resume") {
                    self.advance();
                    if matches!(self.peek_tok(), Token::LParen) {
                        self.advance();
                        let rp = self.expect_ident()?;
                        self.expect(&Token::RParen)?;
                        resume_param = Some(rp);
                    }
                } else {
                    return Err(ParseError::UnexpectedToken {
                        expected: "'resume'".to_owned(),
                        found: format!("{}", self.peek_tok()),
                        span: self.current_span(),
                    });
                }
            }
            self.expect(&Token::FatArrow)?;
            let body = self.parse_expr()?;
            let span = arm_start.merge(body.span());
            arms.push(AstHandlerArm {
                effect_name,
                params,
                resume_param,
                body: Box::new(body),
                span,
            });
            // Optional comma between arms.
            if matches!(self.peek_tok(), Token::Comma) {
                self.advance();
                continue;
            }
        }
        let end = self.expect(&Token::RBrace)?;
        Ok(AstStmt::HandleStmt {
            expr: Box::new(expr),
            arms,
            return_ty,
            span: start.merge(end),
        })
    }

    fn parse_par_for_stmt(&mut self) -> Result<AstStmt, ParseError> {
        let start = self.current_span();
        self.expect(&Token::Par)?;
        self.expect(&Token::For)?;
        let var = self.expect_ident()?;
        self.expect(&Token::In)?;
        let range_start = self.parse_expr()?;
        let inclusive = matches!(self.peek_tok(), Token::DotDotEq);
        if inclusive {
            self.expect(&Token::DotDotEq)?;
        } else {
            self.expect(&Token::DotDot)?;
        }
        let range_end = self.parse_expr()?;
        let body = self.parse_block()?;
        let span = start.merge(body.span);
        Ok(AstStmt::ParFor {
            var,
            start: Box::new(range_start),
            end: Box::new(range_end),
            inclusive,
            body,
            span,
        })
    }

    fn parse_call_args(&mut self) -> Result<Vec<AstExpr>, ParseError> {
        let mut args = Vec::new();
        if matches!(self.peek_tok(), Token::RParen) {
            return Ok(args);
        }
        args.push(self.parse_expr()?);
        while matches!(self.peek_tok(), Token::Comma) {
            self.advance();
            if matches!(self.peek_tok(), Token::RParen) {
                break;
            }
            args.push(self.parse_expr()?);
        }
        Ok(args)
    }

    /// Parse a sub-pattern inside a tuple pattern: wildcard, int/bool literal, or ident binding.
    fn parse_when_sub_pattern(&mut self) -> Result<AstWhenPattern, ParseError> {
        match self.peek_tok().clone() {
            Token::Ident(ref name) if name == "_" => {
                self.advance();
                Ok(AstWhenPattern::Wildcard)
            }
            Token::Ident(name) => {
                let name = name.clone();
                self.advance();
                // Check for some(x) / ok(x) / err(e)
                if (name == "some" || name == "ok" || name == "err")
                    && matches!(self.peek_tok(), Token::LParen)
                {
                    self.advance(); // consume '('
                    let binding = if matches!(self.peek_tok(), Token::RParen) {
                        None
                    } else {
                        Some(self.expect_ident()?.name)
                    };
                    self.expect(&Token::RParen)?;
                    return if name == "some" {
                        Ok(AstWhenPattern::OptionSome { binding })
                    } else if name == "ok" {
                        Ok(AstWhenPattern::ResultOk { binding })
                    } else {
                        Ok(AstWhenPattern::ResultErr { binding })
                    };
                }
                if name == "none" && !matches!(self.peek_tok(), Token::Dot) {
                    return Ok(AstWhenPattern::OptionNone);
                }
                // EnumName.Variant or binding: collect dot-separated parts
                let mut parts = vec![name.clone()];
                while matches!(self.peek_tok(), Token::Dot) {
                    self.advance();
                    let sub = self.expect_ident()?.name;
                    parts.push(sub);
                }
                if parts.len() >= 2 {
                    // Enum variant pattern: EnumName.VariantName
                    let variant_name = parts.pop().unwrap();
                    let enum_name = parts.join("__");
                    // Parse optional bindings: Variant(a, b, ...)
                    let bindings = if matches!(self.peek_tok(), Token::LParen) {
                        self.advance(); // consume '('
                        let mut names = Vec::new();
                        while !matches!(self.peek_tok(), Token::RParen | Token::Eof) {
                            names.push(self.expect_ident()?.name);
                            if matches!(self.peek_tok(), Token::Comma) {
                                self.advance();
                            }
                        }
                        self.expect(&Token::RParen)?;
                        names
                    } else {
                        Vec::new()
                    };
                    Ok(AstWhenPattern::EnumVariant {
                        enum_name,
                        variant_name,
                        bindings,
                    })
                } else {
                    // Plain identifier binding
                    Ok(AstWhenPattern::EnumVariant {
                        enum_name: String::new(),
                        variant_name: name,
                        bindings: vec![],
                    })
                }
            }
            Token::IntLit(n) => {
                let n = n;
                self.advance();
                // Check for inclusive range pattern: lo..=hi
                if matches!(self.peek_tok(), Token::DotDotEq) {
                    self.advance(); // consume '..='
                    let hi = match self.peek_tok().clone() {
                        Token::IntLit(h) => {
                            self.advance();
                            h
                        }
                        _ => {
                            return Err(ParseError::UnexpectedToken {
                                expected: "integer for range upper bound".to_owned(),
                                found: format!("{}", self.peek_tok()),
                                span: self.current_span(),
                            })
                        }
                    };
                    Ok(AstWhenPattern::Range { lo: n, hi })
                } else {
                    Ok(AstWhenPattern::IntLit(n))
                }
            }
            Token::BoolLit(b) => {
                let b = b;
                self.advance();
                Ok(AstWhenPattern::BoolLit(b))
            }
            Token::StringLit(ref s) => {
                let s = s.clone();
                self.advance();
                Ok(AstWhenPattern::StringLit(s))
            }
            Token::LParen => {
                // Tuple pattern: (sub, sub, ...)
                self.advance();
                let mut subs = Vec::new();
                while !matches!(self.peek_tok(), Token::RParen | Token::Eof) {
                    let sub = self.parse_when_sub_pattern()?;
                    subs.push(sub);
                    if matches!(self.peek_tok(), Token::Comma) {
                        self.advance();
                    }
                }
                self.expect(&Token::RParen)?;
                Ok(AstWhenPattern::Tuple(subs))
            }
            Token::LBracket => {
                // Slice pattern: [a, b, ..rest]
                self.advance();
                let mut prefix = Vec::new();
                let mut rest = None;
                while !matches!(self.peek_tok(), Token::RBracket | Token::Eof) {
                    if matches!(self.peek_tok(), Token::DotDot) {
                        self.advance();
                        if matches!(self.peek_tok(), Token::Ident(_)) {
                            rest = Some(self.expect_ident()?.name);
                        }
                        break;
                    }
                    let sub = self.parse_when_sub_pattern()?;
                    prefix.push(sub);
                    if matches!(self.peek_tok(), Token::Comma) {
                        self.advance();
                    }
                }
                self.expect(&Token::RBracket)?;
                Ok(AstWhenPattern::Slice { prefix, rest })
            }
            _ => Err(ParseError::UnexpectedToken {
                expected: "sub-pattern (wildcard, literal, range, identifier, enum variant, tuple, or slice)".to_owned(),
                found: format!("{}", self.peek_tok()),
                span: self.current_span(),
            }),
        }
    }

    /// Parse a complete when pattern (handles or-patterns at top level).
    fn parse_when_pattern(&mut self) -> Result<AstWhenPattern, ParseError> {
        let mut patterns = vec![self.parse_when_sub_pattern()?];
        while matches!(self.peek_tok(), Token::Pipe) {
            self.advance(); // consume '|'
            let next_pat = self.parse_when_sub_pattern()?;
            patterns.push(next_pat);
        }
        if patterns.len() == 1 {
            Ok(patterns.pop().unwrap())
        } else {
            Ok(AstWhenPattern::Or(patterns))
        }
    }

    /// Desugar `f"Hello {name}! You are {age} years old."` into nested `concat` calls.
    /// Supports full expressions inside `{...}` with optional type ascription `{expr: Type}`.
    /// Each placeholder is wrapped with `to_str(expr)` so any type can be interpolated.
    fn desugar_fstring(&self, raw: &str, span: Span) -> AstExpr {
        // Split raw into alternating text/expr parts.
        #[derive(Debug)]
        enum Part {
            Text(String),
            Expr { expr_str: String, ty_annotation: Option<AstType> },
        }
        let mut parts: Vec<Part> = Vec::new();
        let mut cur = String::new();
        let mut chars = raw.chars().peekable();
        while let Some(c) = chars.next() {
            if c == '{' {
                if !cur.is_empty() {
                    parts.push(Part::Text(cur.clone()));
                    cur.clear();
                }
                let mut expr_content = String::new();
                for ic in chars.by_ref() {
                    if ic == '}' {
                        break;
                    }
                    expr_content.push(ic);
                }
                let expr_content = expr_content.trim();
                if !expr_content.is_empty() {
                    // Check for type ascription: expr:Type
                    let (expr_str, ty_annotation) = if let Some(colon_pos) = expr_content.rfind(':') {
                        // Check if this is a type ascription (not a ternary or similar)
                        let before_colon = &expr_content[..colon_pos].trim();
                        let after_colon = &expr_content[colon_pos + 1..].trim();
                        // Simple heuristic: if after colon looks like a type, treat as annotation
                        if !after_colon.is_empty() && (after_colon.chars().next().unwrap().is_uppercase() || after_colon.starts_with("list<") || after_colon.starts_with("map<") || after_colon.starts_with("option<") || after_colon.starts_with("result<") || after_colon.starts_with("tensor<") || after_colon.starts_with("chan<") || after_colon.starts_with("atomic<") || after_colon.starts_with("mutex<")) {
                            let ty = self.parse_type_annotation(after_colon, span);
                            (before_colon.to_string(), Some(ty))
                        } else {
                            (expr_content.to_string(), None)
                        }
                    } else {
                        (expr_content.to_string(), None)
                    };
                    parts.push(Part::Expr { expr_str, ty_annotation });
                }
            } else {
                cur.push(c);
            }
        }
        if !cur.is_empty() {
            parts.push(Part::Text(cur));
        }

        // Helper: build an AstExpr for a single part.
        let make_part = |p: &Part| -> AstExpr {
            match p {
                Part::Text(s) => AstExpr::StringLit {
                    value: s.clone(),
                    span,
                },
                Part::Expr { expr_str, ty_annotation } => {
                    // Parse the expression string
                    let expr_tokens = crate::parser::lexer::Lexer::new(expr_str).tokenize().unwrap_or_default();
                    let mut expr_parser = Parser::new(&expr_tokens);
                    let mut expr = expr_parser.parse_expr().unwrap_or_else(|_| {
                        // Fallback to identifier if parsing fails
                        AstExpr::Ident(crate::parser::ast::Ident {
                            name: expr_str.clone(),
                            span,
                        })
                    });
                    // Apply type annotation if present (via `to` cast)
                    if let Some(ty) = ty_annotation {
                        expr = AstExpr::Cast {
                            expr: Box::new(expr),
                            ty: ty.clone(),
                            span,
                        };
                    }
                    // Wrap with to_str for interpolation
                    AstExpr::Call {
                        callee: Ident {
                            name: "to_str".into(),
                            span,
                        },
                        args: vec![expr],
                        span,
                    }
                }
            }
        };

        if parts.is_empty() {
            return AstExpr::StringLit {
                value: String::new(),
                span,
            };
        }

        // Build right-to-left concat chain.
        let mut expr = make_part(parts.last().expect("parts is non-empty, checked above"));
        for p in parts[..parts.len() - 1].iter().rev() {
            let left = make_part(p);
            expr = AstExpr::Call {
                callee: Ident {
                    name: "concat".into(),
                    span,
                },
                args: vec![left, expr],
                span,
            };
        }
        expr
    }

    /// Parse a type annotation string (e.g., "i64", "list<f64>") into an AstType.
    fn parse_type_annotation(&self, type_str: &str, span: Span) -> AstType {
        let tokens = crate::parser::lexer::Lexer::new(type_str).tokenize().unwrap_or_default();
        let mut ty_parser = Parser::new(&tokens);
        ty_parser.parse_type().unwrap_or(AstType::Named(type_str.to_string(), span))
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::lexer::Lexer;

    /// Helper: parse source to AST module, expecting no errors.
    fn parse_ok(src: &str) -> AstModule {
        let tokens = Lexer::new(src).tokenize().expect("lex failed");
        let mut parser = Parser::new(&tokens);
        let (module, errors) = parser.parse_module_recovering();
        assert!(errors.is_empty(), "unexpected parse errors: {:?}", errors);
        module
    }

    /// Helper: parse source and expect at least one parse error.
    fn parse_err(src: &str) -> Vec<crate::error::ParseError> {
        let tokens = Lexer::new(src).tokenize().expect("lex failed");
        let mut parser = Parser::new(&tokens);
        let (_module, errors) = parser.parse_module_recovering();
        assert!(!errors.is_empty(), "expected parse errors but got none");
        errors
    }

    // -- Functions --------------------------------------------------------

    #[test]
    fn parse_empty_function() {
        let m = parse_ok("def main() -> i64 { 0 }");
        assert_eq!(m.functions.len(), 1);
        assert_eq!(m.functions[0].name.name, "main");
    }

    #[test]
    fn parse_function_with_params() {
        let m = parse_ok("def add(a: i64, b: i64) -> i64 { a + b }");
        assert_eq!(m.functions.len(), 1);
        assert_eq!(m.functions[0].params.len(), 2);
        assert_eq!(m.functions[0].params[0].name.name, "a");
        assert_eq!(m.functions[0].params[1].name.name, "b");
    }

    #[test]
    fn parse_multiple_functions() {
        let src = r#"
            def foo() -> i64 { 1 }
            def bar() -> i64 { 2 }
            def baz() -> i64 { 3 }
        "#;
        let m = parse_ok(src);
        assert_eq!(m.functions.len(), 3);
    }

    // -- Records and Enums ------------------------------------------------

    #[test]
    fn parse_record() {
        let m = parse_ok("record Point { x: f64, y: f64 }");
        assert_eq!(m.structs.len(), 1);
        assert_eq!(m.structs[0].name.name, "Point");
        assert_eq!(m.structs[0].fields.len(), 2);
    }

    #[test]
    fn parse_enum() {
        let m = parse_ok("choice Color { Red, Green, Blue }");
        assert_eq!(m.enums.len(), 1);
        assert_eq!(m.enums[0].name.name, "Color");
        assert_eq!(m.enums[0].variants.len(), 3);
    }

    // -- Control flow -----------------------------------------------------

    #[test]
    fn parse_if_else() {
        let m = parse_ok("def f(x: i64) -> i64 { if x > 0 { x } else { 0 - x } }");
        assert_eq!(m.functions.len(), 1);
    }

    #[test]
    fn parse_while_loop() {
        let m = parse_ok("def f() -> i64 { var i = 0; while i < 10 { i = i + 1; } i }");
        assert_eq!(m.functions.len(), 1);
    }

    #[test]
    fn parse_for_range() {
        let m = parse_ok("def f() -> i64 { for i in 0..10 { print(to_str(i)); } 0 }");
        assert_eq!(m.functions.len(), 1);
    }

    // -- Bindings ---------------------------------------------------------

    #[test]
    fn parse_val_binding() {
        let m = parse_ok("def f() -> i64 { val x = 42; x }");
        assert_eq!(m.functions.len(), 1);
    }

    #[test]
    fn parse_var_binding() {
        let m = parse_ok("def f() -> i64 { var x = 0; x = 1; x }");
        assert_eq!(m.functions.len(), 1);
    }

    // -- Types ------------------------------------------------------------

    #[test]
    fn parse_tensor_type() {
        let m = parse_ok("def f(t: tensor<f32, [3, 4]>) -> i64 { 0 }");
        assert_eq!(m.functions[0].params.len(), 1);
    }

    #[test]
    fn parse_option_type() {
        let m = parse_ok("def f() -> option<i64> { none }");
        assert_eq!(m.functions.len(), 1);
    }

    #[test]
    fn parse_result_type() {
        let m = parse_ok("def f() -> result<i64, str> { ok(42) }");
        assert_eq!(m.functions.len(), 1);
    }

    #[test]
    fn parse_list_type() {
        let m = parse_ok("def f(l: list<i64>) -> i64 { 0 }");
        assert_eq!(m.functions.len(), 1);
    }

    #[test]
    fn parse_map_type() {
        let m = parse_ok("def f(m: map<str, i64>) -> i64 { 0 }");
        assert_eq!(m.functions.len(), 1);
    }

    // -- Closures ---------------------------------------------------------

    #[test]
    fn parse_closure() {
        let m = parse_ok("def f() -> i64 { val double = |x: i64| x * 2; double(21) }");
        assert_eq!(m.functions.len(), 1);
    }

    // -- Pattern matching -------------------------------------------------

    #[test]
    fn parse_when_expression() {
        let src = r#"
            choice Dir { Up, Down }
            def f(d: Dir) -> i64 {
                when d {
                    Dir.Up   => 1,
                    Dir.Down => 0,
                }
            }
        "#;
        let m = parse_ok(src);
        assert_eq!(m.enums.len(), 1);
        assert_eq!(m.functions.len(), 1);
    }

    // -- Const declarations -----------------------------------------------

    #[test]
    fn parse_const() {
        let m = parse_ok("const PI: f64 = 3.14159");
        assert_eq!(m.consts.len(), 1);
    }

    // -- Error recovery ---------------------------------------------------

    #[test]
    fn parse_missing_return_type() {
        let _ = parse_err("def f() { 0 }");
    }

    #[test]
    fn parse_missing_closing_brace() {
        let _ = parse_err("def f() -> i64 { 0");
    }

    // -- Bring (imports) --------------------------------------------------

    #[test]
    fn parse_bring() {
        let m = parse_ok("bring std.math\ndef f() -> i64 { 0 }");
        assert_eq!(m.brings.len(), 1);
    }

    // -- Traits and impls -------------------------------------------------

    #[test]
    fn parse_trait() {
        let m = parse_ok("trait Printable { def to_string(self: Self) -> str }");
        assert_eq!(m.traits.len(), 1);
    }

    // -- Complex programs -------------------------------------------------

    #[test]
    fn parse_full_program() {
        let src = r#"
            record Point { x: f64, y: f64 }
            choice Shape { Circle, Square }
            const MAX: i64 = 100
            def distance(a: Point, b: Point) -> f64 {
                val dx = b.x - a.x;
                val dy = b.y - a.y;
                sqrt(dx * dx + dy * dy)
            }
            def main() -> i64 {
                val p1 = Point { x: 0.0, y: 0.0 };
                val p2 = Point { x: 3.0, y: 4.0 };
                val d = distance(p1, p2);
                0
            }
        "#;
        let m = parse_ok(src);
        assert_eq!(m.structs.len(), 1);
        assert_eq!(m.enums.len(), 1);
        assert_eq!(m.consts.len(), 1);
        assert_eq!(m.functions.len(), 2);
    }
}
