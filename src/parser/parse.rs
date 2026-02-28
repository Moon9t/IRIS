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
use crate::parser::ast::{
    AstBinOp, AstBlock, AstConst, AstDim, AstEnumDef, AstEnumVariant, AstExpr, AstFieldDef,
    AstFunction, AstImplDef, AstLayer, AstLayerParam, AstModel, AstModelInput, AstModelOutput,
    AstModule, AstParam, AstScalarKind, AstStmt, AstStructDef, AstTraitDef, AstTraitMethod,
    AstType, AstTypeAlias, AstUnaryOp, AstWhenArm, AstWhenPattern, Ident,
};
use crate::parser::lexer::{Span, Spanned, Token};

pub struct Parser<'t> {
    tokens: &'t [Spanned<Token>],
    pos: usize,
}

impl<'t> Parser<'t> {
    pub fn new(tokens: &'t [Spanned<Token>]) -> Self {
        Self { tokens, pos: 0 }
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
        let mut enums = Vec::new();
        let mut structs = Vec::new();
        let mut functions = Vec::new();
        let mut models = Vec::new();
        let mut consts = Vec::new();
        let mut type_aliases = Vec::new();
        let mut traits = Vec::new();
        let mut impls = Vec::new();
        let mut imports = Vec::new();
        while !self.at_eof() {
            match self.peek_tok().clone() {
                Token::Choice => enums.push(self.parse_enum_def()?),
                Token::Record => structs.push(self.parse_struct_def()?),
                Token::Def | Token::Async | Token::At => functions.push(self.parse_fn()?),
                Token::Model => models.push(self.parse_model()?),
                Token::Const => consts.push(self.parse_const_decl()?),
                Token::Type => type_aliases.push(self.parse_type_alias()?),
                Token::Trait => traits.push(self.parse_trait_def()?),
                Token::Impl => impls.push(self.parse_impl_def()?),
                Token::Bring => {
                    self.advance();
                    let mod_name = self.expect_ident()?.name;
                    imports.push(mod_name);
                }
                Token::Pub => {
                    self.advance(); // consume 'pub'
                    match self.peek_tok().clone() {
                        Token::Def | Token::Async => {
                            let mut func = self.parse_fn()?;
                            func.is_pub = true;
                            functions.push(func);
                        }
                        Token::Record => structs.push(self.parse_struct_def()?),
                        Token::Choice => enums.push(self.parse_enum_def()?),
                        Token::Trait => traits.push(self.parse_trait_def()?),
                        _ => return Err(ParseError::UnexpectedToken {
                            expected: "'def', 'record', or 'choice' after 'pub'".to_owned(),
                            found: format!("{}", self.peek_tok()),
                            span: self.current_span(),
                        }),
                    }
                }
                _ => {
                    return Err(ParseError::UnexpectedToken {
                        expected: "'choice', 'record', 'def', 'model', 'const', 'type', 'trait', 'impl', or 'bring'".to_owned(),
                        found: format!("{}", self.peek_tok()),
                        span: self.current_span(),
                    })
                }
            }
        }
        Ok(AstModule {
            enums,
            structs,
            functions,
            models,
            consts,
            type_aliases,
            traits,
            impls,
            imports,
        })
    }

    /// Parses a type name as a plain string (handles keywords like `i64`, `f64`, `bool`, `str`
    /// in addition to bare identifiers). Used for `impl Trait for TypeName`.
    fn parse_type_name_str(&mut self) -> Result<String, ParseError> {
        let name = match self.peek_tok().clone() {
            Token::I64 => { self.advance(); "i64".to_owned() }
            Token::I32 => { self.advance(); "i32".to_owned() }
            Token::F64 => { self.advance(); "f64".to_owned() }
            Token::F32 => { self.advance(); "f32".to_owned() }
            Token::Bool => { self.advance(); "bool".to_owned() }
            Token::Str => { self.advance(); "str".to_owned() }
            Token::Ident(n) => { let n = n.clone(); self.advance(); n }
            _ => return Err(ParseError::UnexpectedToken {
                expected: "type name".to_owned(),
                found: format!("{}", self.peek_tok()),
                span: self.current_span(),
            }),
        };
        Ok(name)
    }

    /// Parses `trait Name { (def method(params) -> type)* }`.
    fn parse_trait_def(&mut self) -> Result<AstTraitDef, ParseError> {
        let start = self.current_span();
        self.expect(&Token::Trait)?;
        let name = self.expect_ident()?;
        self.expect(&Token::LBrace)?;
        let mut methods = Vec::new();
        while !matches!(self.peek_tok(), Token::RBrace | Token::Eof) {
            let m_start = self.current_span();
            self.expect(&Token::Def)?;
            let m_name = self.expect_ident()?;
            self.expect(&Token::LParen)?;
            let mut params = Vec::new();
            while !matches!(self.peek_tok(), Token::RParen | Token::Eof) {
                let pname = self.expect_ident()?;
                self.expect(&Token::Colon)?;
                let pty = self.parse_type()?;
                params.push(AstParam { name: pname, ty: pty, default: None });
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
        Ok(AstTraitDef { name, methods, span: start.merge(end) })
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
        let mut methods = Vec::new();
        while !matches!(self.peek_tok(), Token::RBrace | Token::Eof) {
            methods.push(self.parse_fn()?);
        }
        let end = self.expect(&Token::RBrace)?;
        Ok(AstImplDef { trait_name, type_name, methods, span: start.merge(end) })
    }

    /// Parses `type Name = Type`.
    fn parse_type_alias(&mut self) -> Result<AstTypeAlias, ParseError> {
        let start = self.current_span();
        self.expect(&Token::Type)?;
        let name = self.expect_ident()?.name;
        self.expect(&Token::Eq)?;
        let ty = self.parse_type()?;
        let end = start; // span is approximate — just use the keyword span
        Ok(AstTypeAlias { name, ty, span: start.merge(end) })
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
        })
    }

    fn parse_struct_def(&mut self) -> Result<AstStructDef, ParseError> {
        let start = self.current_span();
        self.expect(&Token::Record)?;
        let name = self.expect_ident()?;
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
            fields,
            span: start.merge(end),
        })
    }

    fn parse_fn(&mut self) -> Result<AstFunction, ParseError> {
        let start = self.current_span();
        // Optional @attr annotations before async/def
        let mut attrs = Vec::new();
        while matches!(self.peek_tok(), Token::At) {
            self.advance(); // consume '@'
            let attr_name = self.expect_ident()?.name;
            attrs.push(attr_name);
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
        let type_params = if matches!(self.peek_tok(), Token::LBracket) {
            self.advance(); // consume '['
            let mut ty_params = Vec::new();
            while !matches!(self.peek_tok(), Token::RBracket | Token::Eof) {
                let tp = self.expect_ident()?;
                ty_params.push(tp.name);
                if matches!(self.peek_tok(), Token::Comma) {
                    self.advance();
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
        let body = self.parse_block()?;
        let span = start.merge(body.span);
        Ok(AstFunction {
            name,
            is_pub: false, // set to true by parse_module when preceded by `pub`
            type_params,
            params,
            return_ty,
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
                Token::Input => inputs.push(self.parse_model_input()?),
                Token::Layer => layers.push(self.parse_layer()?),
                Token::Output => outputs.push(self.parse_model_output()?),
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
        self.expect(&Token::Input)?;
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
        self.expect(&Token::Layer)?;
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
        self.expect(&Token::Output)?;
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
                let len = match self.peek_tok().clone() {
                    Token::IntLit(n) => {
                        self.advance();
                        n as usize
                    }
                    _ => {
                        return Err(ParseError::UnexpectedToken {
                            expected: "integer length for array type".to_owned(),
                            found: format!("{}", self.peek_tok()),
                            span: self.current_span(),
                        })
                    }
                };
                let end = self.expect(&Token::RBracket)?;
                Ok(AstType::Array {
                    elem: Box::new(elem),
                    len,
                    span: span.merge(end),
                })
            }
            Token::Ident(ref name) if name == "u8" => {
                let _ = name.clone(); self.advance();
                Ok(AstType::Scalar(AstScalarKind::U8, span))
            }
            Token::Ident(ref name) if name == "i8" => {
                let _ = name.clone(); self.advance();
                Ok(AstType::Scalar(AstScalarKind::I8, span))
            }
            Token::Ident(ref name) if name == "u32" => {
                let _ = name.clone(); self.advance();
                Ok(AstType::Scalar(AstScalarKind::U32, span))
            }
            Token::Ident(ref name) if name == "u64" => {
                let _ = name.clone(); self.advance();
                Ok(AstType::Scalar(AstScalarKind::U64, span))
            }
            Token::Ident(ref name) if name == "usize" => {
                let _ = name.clone(); self.advance();
                Ok(AstType::Scalar(AstScalarKind::USize, span))
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
                Ok(AstType::Result(Box::new(ok_ty), Box::new(err_ty), span.merge(end)))
            }
            Token::Ident(name) => {
                self.advance();
                Ok(AstType::Named(name, span))
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

            // `val` or `var` binding statement
            if matches!(self.peek_tok(), Token::Val | Token::Var) {
                stmts.push(self.parse_let_stmt()?);
                continue;
            }

            // `while` statement
            if matches!(self.peek_tok(), Token::While) {
                stmts.push(self.parse_while_stmt()?);
                continue;
            }

            // `for` range loop
            if matches!(self.peek_tok(), Token::For) {
                stmts.push(self.parse_for_stmt()?);
                continue;
            }

            // `par for` parallel range loop
            if matches!(self.peek_tok(), Token::Par) {
                stmts.push(self.parse_par_for_stmt()?);
                continue;
            }

            // `spawn { }` concurrent task
            if matches!(self.peek_tok(), Token::Spawn) {
                stmts.push(self.parse_spawn_stmt()?);
                continue;
            }

            // `loop` statement
            if matches!(self.peek_tok(), Token::Loop) {
                stmts.push(self.parse_loop_stmt()?);
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
            if matches!(self.peek_tok(), Token::Eq) {
                // Assignment: lvalue = value
                let start_span = expr.span();
                self.advance(); // consume '='
                let value = self.parse_expr()?;
                let end_span = value.span();
                if matches!(self.peek_tok(), Token::Semi) {
                    self.advance();
                }
                stmts.push(AstStmt::Assign {
                    target: Box::new(expr),
                    value: Box::new(value),
                    span: start_span.merge(end_span),
                });
            } else if matches!(self.peek_tok(), Token::Semi) {
                self.advance(); // consume `;`
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
        self.advance(); // consume 'val' or 'var' (caller already checked)

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
                span: start.merge(end),
            });
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
        let mut lhs = self.parse_add_expr()?;
        loop {
            if !matches!(self.peek_tok(), Token::AmpAmp) {
                break;
            }
            self.advance();
            let rhs = self.parse_add_expr()?;
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
        let mut expr = self.parse_cmp_expr()?;
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
        let mut lhs = self.parse_unary()?;
        loop {
            let op = match self.peek_tok() {
                Token::EqEq => AstBinOp::CmpEq,
                Token::NotEq => AstBinOp::CmpNe,
                Token::LAngle => AstBinOp::CmpLt,
                Token::LtEq => AstBinOp::CmpLe,
                Token::RAngle => AstBinOp::CmpGt,
                Token::GtEq => AstBinOp::CmpGe,
                _ => break,
            };
            self.advance();
            let rhs = self.parse_unary()?;
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
                let cond = self.parse_expr()?;
                let then_block = self.parse_block()?;
                let (else_block, end_span) = if matches!(self.peek_tok(), Token::Else) {
                    self.advance();
                    let eb = self.parse_block()?;
                    let es = eb.span;
                    (Some(eb), es)
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

            Token::LBrace => {
                let block = self.parse_block()?;
                AstExpr::Block(block)
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
                    params.push(AstParam { name, ty, default: None });
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
                    let (pattern, enum_name_leg, variant_name_leg) =
                        match self.peek_tok() {
                            Token::IntLit(n) => {
                                let n = *n;
                                self.advance(); // consume int literal
                                // Check for inclusive range pattern: lo..=hi
                                if matches!(self.peek_tok(), Token::DotDotEq) {
                                    self.advance(); // consume '..='
                                    let hi = match self.peek_tok().clone() {
                                        Token::IntLit(h) => { self.advance(); h }
                                        _ => return Err(ParseError::UnexpectedToken {
                                            expected: "integer for range upper bound".to_owned(),
                                            found: format!("{}", self.peek_tok()),
                                            span: self.current_span(),
                                        }),
                                    };
                                    (AstWhenPattern::Range { lo: n, hi }, "_range".to_string(), format!("{}..={}", n, hi))
                                } else {
                                    (AstWhenPattern::IntLit(n), "_lit".to_string(), n.to_string())
                                }
                            }
                            Token::BoolLit(b) => {
                                let b = *b;
                                self.advance(); // consume bool literal
                                (AstWhenPattern::BoolLit(b), "_lit".to_string(), b.to_string())
                            }
                            Token::StringLit(_) => {
                                let s = if let Token::StringLit(s) = self.peek_tok() {
                                    s.clone()
                                } else { unreachable!() };
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
                                (AstWhenPattern::Tuple(subs), "_tuple".to_string(), "_tuple".to_string())
                            }
                            _ => {
                                // Peek at ident to determine pattern type.
                                let first_name = self.expect_ident()?.name;
                                if first_name == "_" {
                                    // Wildcard pattern.
                                    (AstWhenPattern::Wildcard, "_".to_string(), "_".to_string())
                                } else if (first_name == "some" || first_name == "ok" || first_name == "err")
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
                                        AstWhenPattern::OptionSome { binding: binding.clone() }
                                    } else if first_name == "ok" {
                                        AstWhenPattern::ResultOk { binding: binding.clone() }
                                    } else {
                                        AstWhenPattern::ResultErr { binding: binding.clone() }
                                    };
                                    (pat, first_name.clone(), binding.unwrap_or_default())
                                } else if first_name == "none" && !matches!(self.peek_tok(), Token::Dot) {
                                    // `none` pattern (no dot follows)
                                    (AstWhenPattern::OptionNone, "none".to_string(), "none".to_string())
                                } else {
                                    // `EnumName.Variant` or `EnumName.Variant(a, b, ...)` — enum pattern
                                    self.expect(&Token::Dot)?;
                                    let variant_name = self.expect_ident()?.name;
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
                                        enum_name: first_name.clone(),
                                        variant_name: variant_name.clone(),
                                        bindings,
                                    };
                                    (pat, first_name, variant_name)
                                }
                            }
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
                        expr = AstExpr::FieldAccess {
                            base: Box::new(expr),
                            field: field.name,
                            span: start.merge(end),
                        };
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
        let cond = self.parse_expr()?;
        let body = self.parse_block()?;
        let span = start.merge(body.span);
        Ok(AstStmt::While {
            cond: Box::new(cond),
            body,
            span,
        })
    }

    fn parse_for_stmt(&mut self) -> Result<AstStmt, ParseError> {
        let start = self.current_span();
        self.expect(&Token::For)?;
        let var = self.expect_ident()?;
        self.expect(&Token::In)?;
        let iter_expr = self.parse_expr()?;
        // If the next token is `..`, it's a range loop; otherwise it's a foreach loop.
        if matches!(self.peek_tok(), Token::DotDot) {
            self.expect(&Token::DotDot)?;
            let range_end = self.parse_expr()?;
            let body = self.parse_block()?;
            let span = start.merge(body.span);
            Ok(AstStmt::ForRange {
                var,
                start: Box::new(iter_expr),
                end: Box::new(range_end),
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
        let block = self.parse_block()?;
        let span = start.merge(block.span);
        // Collect stmts, and if the block has a tail expression, append it as a statement too.
        let mut body = block.stmts;
        if let Some(tail) = block.tail {
            body.push(AstStmt::Expr(tail));
        }
        Ok(AstStmt::Spawn { body, span })
    }

    fn parse_par_for_stmt(&mut self) -> Result<AstStmt, ParseError> {
        let start = self.current_span();
        self.expect(&Token::Par)?;
        self.expect(&Token::For)?;
        let var = self.expect_ident()?;
        self.expect(&Token::In)?;
        let range_start = self.parse_expr()?;
        self.expect(&Token::DotDot)?;
        let range_end = self.parse_expr()?;
        let body = self.parse_block()?;
        let span = start.merge(body.span);
        Ok(AstStmt::ParFor {
            var,
            start: Box::new(range_start),
            end: Box::new(range_end),
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
                // Treat as a binding (wildcard-style, name is captured in Wildcard for simplicity
                // — we use a dedicated Binding variant by reusing IntLit with a sentinel? No:
                // we store bindings as Wildcard with a special flag. Actually, just use a new
                // convention: store as EnumVariant with empty enum name to mean "binding".
                // Simplest: sub-patterns that are just identifiers are treated as Wildcard but
                // we need the name for the outer tuple pattern handler to bind them. We'll use
                // a local convention that EnumVariant { enum_name: "", variant_name: name, bindings: [] }
                // means "bind this element to `name`".
                Ok(AstWhenPattern::EnumVariant { enum_name: String::new(), variant_name: name, bindings: vec![] })
            }
            Token::IntLit(_) => {
                let n = if let Token::IntLit(n) = self.peek_tok() { *n } else { unreachable!() };
                self.advance();
                Ok(AstWhenPattern::IntLit(n))
            }
            Token::BoolLit(_) => {
                let b = if let Token::BoolLit(b) = self.peek_tok() { *b } else { unreachable!() };
                self.advance();
                Ok(AstWhenPattern::BoolLit(b))
            }
            Token::StringLit(_) => {
                let s = if let Token::StringLit(s) = self.peek_tok() { s.clone() } else { unreachable!() };
                self.advance();
                Ok(AstWhenPattern::StringLit(s))
            }
            _ => Err(ParseError::UnexpectedToken {
                expected: "sub-pattern (wildcard, literal, or identifier)".to_owned(),
                found: format!("{}", self.peek_tok()),
                span: self.current_span(),
            }),
        }
    }

    /// Desugar `f"Hello {name}! You are {age} years old."` into nested `concat` calls.
    /// Only simple identifiers (no spaces) are supported inside `{...}`.
    /// Each placeholder is wrapped with `to_str(ident)` so any type can be interpolated.
    fn desugar_fstring(&self, raw: &str, span: Span) -> AstExpr {
        // Split raw into alternating text/ident parts.
        enum Part { Text(String), Ident(String) }
        let mut parts: Vec<Part> = Vec::new();
        let mut cur = String::new();
        let mut chars = raw.chars().peekable();
        while let Some(c) = chars.next() {
            if c == '{' {
                if !cur.is_empty() {
                    parts.push(Part::Text(cur.clone()));
                    cur.clear();
                }
                let mut ident = String::new();
                for ic in chars.by_ref() {
                    if ic == '}' { break; }
                    ident.push(ic);
                }
                let ident = ident.trim().to_owned();
                if !ident.is_empty() {
                    parts.push(Part::Ident(ident));
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
                Part::Text(s) => AstExpr::StringLit { value: s.clone(), span },
                Part::Ident(name) => AstExpr::Call {
                    callee: Ident { name: "to_str".into(), span },
                    args: vec![AstExpr::Ident(Ident { name: name.clone(), span })],
                    span,
                },
            }
        };

        if parts.is_empty() {
            return AstExpr::StringLit { value: String::new(), span };
        }

        // Build right-to-left concat chain.
        let mut expr = make_part(parts.last().unwrap());
        for p in parts[..parts.len() - 1].iter().rev() {
            let left = make_part(p);
            expr = AstExpr::Call {
                callee: Ident { name: "concat".into(), span },
                args: vec![left, expr],
                span,
            };
        }
        expr
    }
}
