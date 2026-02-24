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
    AstBinOp, AstBlock, AstDim, AstEnumDef, AstExpr, AstFieldDef, AstFunction, AstLayer,
    AstLayerParam, AstModel, AstModelInput, AstModelOutput, AstModule, AstParam, AstScalarKind,
    AstStmt, AstStructDef, AstType, AstUnaryOp, AstWhenArm, Ident,
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
        while !self.at_eof() {
            match self.peek_tok().clone() {
                Token::Choice => enums.push(self.parse_enum_def()?),
                Token::Record => structs.push(self.parse_struct_def()?),
                Token::Def => functions.push(self.parse_fn()?),
                Token::Model => models.push(self.parse_model()?),
                _ => {
                    return Err(ParseError::UnexpectedToken {
                        expected: "'choice', 'record', 'def', or 'model'".to_owned(),
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
        })
    }

    fn parse_enum_def(&mut self) -> Result<AstEnumDef, ParseError> {
        let start = self.current_span();
        self.expect(&Token::Choice)?;
        let name = self.expect_ident()?;
        self.expect(&Token::LBrace)?;
        let mut variants = Vec::new();
        while !matches!(self.peek_tok(), Token::RBrace | Token::Eof) {
            variants.push(self.expect_ident()?);
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
        self.expect(&Token::Def)?;
        let name = self.expect_ident()?;
        self.expect(&Token::LParen)?;
        let params = self.parse_params()?;
        self.expect(&Token::RParen)?;
        self.expect(&Token::Arrow)?;
        let return_ty = self.parse_type()?;
        let body = self.parse_block()?;
        let span = start.merge(body.span);
        Ok(AstFunction {
            name,
            params,
            return_ty,
            body,
            span,
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
        Ok(AstParam { name, ty })
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
                Ok(AstType::Tuple(elems, span.merge(end)))
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
                    params.push(AstParam { name, ty });
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
                    // Parse `EnumName.Variant => expr`
                    let enum_name = self.expect_ident()?.name;
                    self.expect(&Token::Dot)?;
                    let variant_name = self.expect_ident()?.name;
                    self.expect(&Token::FatArrow)?;
                    let body = self.parse_expr()?;
                    let arm_end = body.span();
                    // Optional comma between arms
                    if matches!(self.peek_tok(), Token::Comma) {
                        self.advance();
                    }
                    arms.push(AstWhenArm {
                        enum_name,
                        variant_name,
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
                    let end = field.span;
                    expr = AstExpr::FieldAccess {
                        base: Box::new(expr),
                        field: field.name,
                        span: start.merge(end),
                    };
                }
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
        let range_start = self.parse_expr()?;
        self.expect(&Token::DotDot)?;
        let range_end = self.parse_expr()?;
        let body = self.parse_block()?;
        let span = start.merge(body.span);
        Ok(AstStmt::ForRange {
            var,
            start: Box::new(range_start),
            end: Box::new(range_end),
            body,
            span,
        })
    }

    fn parse_loop_stmt(&mut self) -> Result<AstStmt, ParseError> {
        let start = self.current_span();
        self.expect(&Token::Loop)?;
        let body = self.parse_block()?;
        let span = start.merge(body.span);
        Ok(AstStmt::Loop { body, span })
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
}
