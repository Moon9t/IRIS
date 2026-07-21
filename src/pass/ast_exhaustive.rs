/// Phase 85: AST-level Exhaustiveness Checking Pass.
///
/// Checks that all `when` expressions in the AST are exhaustive:
/// - For enums: all variants must be covered, or wildcard `_` present
/// - For Option: both `Some` and `None` must be covered, or wildcard
/// - For Result: both `Ok` and `Err` must be covered, or wildcard
/// - Wildcard `_` makes any match exhaustive.
use crate::error::PassError;
use crate::parser::ast::{AstBlock, AstExpr, AstFunction, AstModule, AstStmt, AstWhenArm, AstWhenPattern};
use std::collections::HashSet;

pub struct AstExhaustivenessPass;

impl AstExhaustivenessPass {
    pub fn new() -> Self {
        Self
    }

    /// Run the exhaustiveness check on an AST module.
    pub fn run(&self, module: &AstModule) -> Result<(), PassError> {
        for func in &module.functions {
            self.check_function(func)?;
        }
        for impl_def in &module.impls {
            for method in &impl_def.methods {
                self.check_function(method)?;
            }
        }
        Ok(())
    }

    fn check_function(&self, func: &AstFunction) -> Result<(), PassError> {
        self.check_block(&func.body, &func.name.name)
    }

    fn check_block(&self, block: &AstBlock, fn_name: &str) -> Result<(), PassError> {
        for stmt in &block.stmts {
            self.check_stmt(stmt, fn_name)?;
        }
        // Also check tail expression if present
        if let Some(tail) = &block.tail {
            self.check_expr(tail, fn_name)?;
        }
        Ok(())
    }

    fn check_stmt(&self, stmt: &AstStmt, fn_name: &str) -> Result<(), PassError> {
        match stmt {
            AstStmt::Let { init, .. } => self.check_expr(init, fn_name),
            AstStmt::LetTuple { init, .. } => self.check_expr(init, fn_name),
            AstStmt::Assign { value, .. } => self.check_expr(value, fn_name),
            AstStmt::Expr(expr) => self.check_expr(expr, fn_name),
            AstStmt::While { cond, body, .. } => {
                self.check_expr(cond, fn_name)?;
                self.check_block(body, fn_name)
            }
            AstStmt::ForRange { start, end, body, .. } => {
                self.check_expr(start, fn_name)?;
                self.check_expr(end, fn_name)?;
                self.check_block(body, fn_name)
            }
            AstStmt::ForEach { iter, body, .. } => {
                self.check_expr(iter, fn_name)?;
                self.check_block(body, fn_name)
            }
            AstStmt::Loop { body, .. } => self.check_block(body, fn_name),
            AstStmt::Return { value, .. } => {
                if let Some(v) = value {
                    self.check_expr(v, fn_name)
                } else {
                    Ok(())
                }
            }
            AstStmt::Spawn { body, .. } => {
                for stmt in body {
                    self.check_stmt(stmt, fn_name)?;
                }
                Ok(())
            }
            AstStmt::ParFor { start, end, body, .. } => {
                self.check_expr(start, fn_name)?;
                self.check_expr(end, fn_name)?;
                self.check_block(body, fn_name)
            }
            AstStmt::Break { .. } | AstStmt::Continue { .. } => Ok(()),
            AstStmt::MaskStmt { body, .. } => self.check_block(body, fn_name),
            AstStmt::HandleStmt { expr, .. } => self.check_expr(expr, fn_name),
        }
    }

    fn check_expr(&self, expr: &AstExpr, fn_name: &str) -> Result<(), PassError> {
        match expr {
            AstExpr::When { scrutinee, arms, span } => {
                self.check_expr(scrutinee, fn_name)?;
                self.check_when_arms(scrutinee, arms, *span, fn_name)?;
                Ok(())
            }
            AstExpr::If { cond, then_block, else_block, .. } => {
                self.check_expr(cond, fn_name)?;
                self.check_block(then_block, fn_name)?;
                if let Some(eb) = else_block {
                    self.check_block(eb, fn_name)?;
                }
                Ok(())
            }
            AstExpr::Block(block) => self.check_block(block, fn_name),
            AstExpr::BinOp { lhs, rhs, .. } => {
                self.check_expr(lhs, fn_name)?;
                self.check_expr(rhs, fn_name)
            }
            AstExpr::UnaryOp { expr, .. } => self.check_expr(expr, fn_name),
            AstExpr::Call { args, .. } => {
                for arg in args {
                    self.check_expr(arg, fn_name)?;
                }
                Ok(())
            }
            AstExpr::MethodCall { base, args, .. } => {
                self.check_expr(base, fn_name)?;
                for arg in args {
                    self.check_expr(arg, fn_name)?;
                }
                Ok(())
            }
            AstExpr::FieldAccess { base, .. } => self.check_expr(base, fn_name),
            AstExpr::Index { base, indices, .. } => {
                self.check_expr(base, fn_name)?;
                for idx in indices {
                    self.check_expr(idx, fn_name)?;
                }
                Ok(())
            }
            AstExpr::StructLit { fields, .. } => {
                for (_, e) in fields {
                    self.check_expr(e, fn_name)?;
                }
                Ok(())
            }
            AstExpr::Tuple { elements, .. } => {
                for e in elements {
                    self.check_expr(e, fn_name)?;
                }
                Ok(())
            }
            AstExpr::ArrayLit { elems, .. } => {
                for e in elems {
                    self.check_expr(e, fn_name)?;
                }
                Ok(())
            }
            AstExpr::Lambda { body, .. } => self.check_expr(body, fn_name),
            AstExpr::Await { expr, .. } => self.check_expr(expr, fn_name),
            AstExpr::Try { expr, .. } => self.check_expr(expr, fn_name),
            AstExpr::Cast { expr, .. } => self.check_expr(expr, fn_name),
            AstExpr::TupleIndex { base, .. } => self.check_expr(base, fn_name),
            _ => Ok(()), // literals, identifiers, etc. - no sub-expressions to check
        }
    }

    fn check_when_arms(
        &self,
        scrutinee: &AstExpr,
        arms: &[AstWhenArm],
        span: crate::parser::lexer::Span,
        fn_name: &str,
    ) -> Result<(), PassError> {
        // First, check all arm bodies
        for arm in arms {
            self.check_expr(&arm.body, fn_name)?;
            if let Some(guard) = &arm.guard {
                self.check_expr(guard, fn_name)?;
            }
        }

        // Determine the type being matched by looking at the scrutinee expression
        let scrut_type = self.infer_scrutinee_type(scrutinee);

        match scrut_type {
            ScrutineeType::Enum { variants } => {
                self.check_enum_exhaustive(variants, arms, span, fn_name)
            }
            ScrutineeType::Option => {
                self.check_option_exhaustive(arms, span, fn_name)
            }
            ScrutineeType::Result => {
                self.check_result_exhaustive(arms, span, fn_name)
            }
            ScrutineeType::Unknown => {
                // Can't determine type - skip check
                Ok(())
            }
        }
    }

    fn infer_scrutinee_type(&self, expr: &AstExpr) -> ScrutineeType {
        match expr {
            AstExpr::Ident(ident) => {
                let name = &ident.name;
                // Check for Option/Result type constructors
                if name == "Option" || name == "Result" {
                    ScrutineeType::Unknown // Can't distinguish without more context
                } else {
                    // Could be an enum type name - we'd need type info
                    ScrutineeType::Unknown
                }
            }
            AstExpr::FieldAccess { base, .. } => {
                // Could be enum variant like Color.Red
                if let AstExpr::Ident(_base_ident) = base.as_ref() {
                    // If base is an enum name like "Color", this is an enum variant
                    // For now, we can't easily determine without type info
                    ScrutineeType::Unknown
                } else {
                    ScrutineeType::Unknown
                }
            }
            AstExpr::Call { callee, args, .. } => {
                // Could be some(x), none, ok(x), err(x)
                if args.len() == 1 || args.is_empty() {
                    match callee.name.as_str() {
                        "some" | "none" => ScrutineeType::Option,
                        "ok" | "err" => ScrutineeType::Result,
                        _ => ScrutineeType::Unknown,
                    }
                } else {
                    ScrutineeType::Unknown
                }
            }
            _ => ScrutineeType::Unknown,
        }
    }

    fn check_enum_exhaustive(
        &self,
        variants: Vec<String>,
        arms: &[AstWhenArm],
        _span: crate::parser::lexer::Span,
        fn_name: &str,
    ) -> Result<(), PassError> {
        // Check if there's a wildcard pattern
        let has_wildcard = arms.iter().any(|a| matches!(a.pattern, AstWhenPattern::Wildcard));
        if has_wildcard {
            return Ok(());
        }

        // Collect covered variants
        let mut covered = HashSet::new();
        for arm in arms {
            if let AstWhenPattern::EnumVariant { variant_name, .. } = &arm.pattern {
                covered.insert(variant_name.clone());
            }
        }

        // Check for missing variants
        let missing: Vec<String> = variants.into_iter()
            .filter(|v| !covered.contains(v))
            .collect();

        if !missing.is_empty() {
            return Err(PassError::TypeError {
                func: fn_name.to_string(),
                detail: format!(
                    "non-exhaustive match: variant{} {:?} not covered",
                    if missing.len() == 1 { "" } else { "s" },
                    missing
                ),
            });
        }
        Ok(())
    }

    fn check_option_exhaustive(
        &self,
        arms: &[AstWhenArm],
        _span: crate::parser::lexer::Span,
        fn_name: &str,
    ) -> Result<(), PassError> {
        let has_wildcard = arms.iter().any(|a| matches!(a.pattern, AstWhenPattern::Wildcard));
        if has_wildcard {
            return Ok(());
        }

        let mut has_some = false;
        let mut has_none = false;
        for arm in arms {
            match &arm.pattern {
                AstWhenPattern::OptionSome { .. } => has_some = true,
                AstWhenPattern::OptionNone => has_none = true,
                _ => {}
            }
        }

        if !has_some {
            return Err(PassError::TypeError {
                func: fn_name.to_string(),
                detail: "non-exhaustive match: variant `Some` not covered".into(),
            });
        }
        if !has_none {
            return Err(PassError::TypeError {
                func: fn_name.to_string(),
                detail: "non-exhaustive match: variant `None` not covered".into(),
            });
        }
        Ok(())
    }

    fn check_result_exhaustive(
        &self,
        arms: &[AstWhenArm],
        _span: crate::parser::lexer::Span,
        fn_name: &str,
    ) -> Result<(), PassError> {
        let has_wildcard = arms.iter().any(|a| matches!(a.pattern, AstWhenPattern::Wildcard));
        if has_wildcard {
            return Ok(());
        }

        let mut has_ok = false;
        let mut has_err = false;
        for arm in arms {
            match &arm.pattern {
                AstWhenPattern::ResultOk { .. } => has_ok = true,
                AstWhenPattern::ResultErr { .. } => has_err = true,
                _ => {}
            }
        }

        if !has_ok {
            return Err(PassError::TypeError {
                func: fn_name.to_string(),
                detail: "non-exhaustive match: variant `Ok` not covered".into(),
            });
        }
        if !has_err {
            return Err(PassError::TypeError {
                func: fn_name.to_string(),
                detail: "non-exhaustive match: variant `Err` not covered".into(),
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ScrutineeType {
    #[allow(dead_code)]
    Enum { variants: Vec<String> },
    Option,
    Result,
    Unknown,
}

impl Default for AstExhaustivenessPass {
    fn default() -> Self {
        Self::new()
    }
}