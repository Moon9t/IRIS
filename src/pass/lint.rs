//! Dead variable warnings: detect `val x = expr` bindings that are never used.

use crate::parser::ast::{AstBlock, AstExpr, AstFunction, AstModule, AstStmt, AstWhenArm};
use crate::parser::lexer::Span;

/// A compiler warning (non-fatal diagnostic).
#[derive(Debug, Clone)]
pub struct IrWarning {
    /// Name of the function containing the warning.
    pub func: String,
    /// Human-readable warning message.
    pub message: String,
    /// Optional byte span for the warning location.
    pub span: Option<Span>,
}

/// Analyze an AST module and return dead variable warnings.
pub fn find_unused_vars(module: &AstModule) -> Vec<IrWarning> {
    let mut warnings = Vec::new();
    for func in &module.functions {
        collect_unused_in_function(func, &mut warnings);
    }
    warnings
}

fn collect_unused_in_function(func: &AstFunction, warnings: &mut Vec<IrWarning>) {
    let declared: Vec<(String, Option<Span>)> = func.body.stmts.iter()
        .filter_map(|s| {
            if let AstStmt::Let { name, span, .. } = s { Some((name.name.clone(), Some(*span))) } else { None }
        })
        .collect();

    for (name, span) in declared {
        // Skip names starting with '_' — convention for intentionally unused.
        if name.starts_with('_') { continue; }
        // Check if this name is referenced anywhere in the block.
        if !block_uses_name(&func.body, &name) {
            warnings.push(IrWarning {
                func: func.name.name.clone(),
                message: format!("variable '{}' is assigned but never used", name),
                span,
            });
        }
    }
}

/// Returns true if `name` appears as an `AstExpr::Ident` anywhere in the block.
fn block_uses_name(block: &AstBlock, name: &str) -> bool {
    for stmt in &block.stmts {
        if stmt_uses_name(stmt, name) { return true; }
    }
    if let Some(tail) = &block.tail {
        if expr_uses_name(tail, name) { return true; }
    }
    false
}

fn stmt_uses_name(stmt: &AstStmt, name: &str) -> bool {
    match stmt {
        AstStmt::Let { init, .. } => expr_uses_name(init, name),
        AstStmt::Expr(e) => expr_uses_name(e, name),
        AstStmt::While { cond, body, .. } => {
            expr_uses_name(cond, name) || block_uses_name(body, name)
        }
        AstStmt::Loop { body, .. } => block_uses_name(body, name),
        AstStmt::ForRange { start, end, body, .. } => {
            expr_uses_name(start, name) || expr_uses_name(end, name) || block_uses_name(body, name)
        }
        AstStmt::ForEach { iter, body, .. } => {
            expr_uses_name(iter, name) || block_uses_name(body, name)
        }
        AstStmt::Assign { target, value, .. } => {
            expr_uses_name(target, name) || expr_uses_name(value, name)
        }
        AstStmt::LetTuple { init, .. } => expr_uses_name(init, name),
        AstStmt::Return { value, .. } => {
            value.as_ref().map_or(false, |e| expr_uses_name(e, name))
        }
        AstStmt::Spawn { body, .. } => body.iter().any(|s| stmt_uses_name(s, name)),
        AstStmt::ParFor { start, end, body, .. } => {
            expr_uses_name(start, name) || expr_uses_name(end, name) || block_uses_name(body, name)
        }
        AstStmt::Break { .. } | AstStmt::Continue { .. } => false,
    }
}

fn expr_uses_name(expr: &AstExpr, name: &str) -> bool {
    match expr {
        AstExpr::Ident(ident) => ident.name == name,
        AstExpr::IntLit { .. } | AstExpr::FloatLit { .. } | AstExpr::BoolLit { .. }
        | AstExpr::StringLit { .. } => false,
        AstExpr::BinOp { lhs, rhs, .. } => {
            expr_uses_name(lhs, name) || expr_uses_name(rhs, name)
        }
        AstExpr::UnaryOp { expr, .. } | AstExpr::Cast { expr, .. }
        | AstExpr::Await { expr, .. } | AstExpr::Try { expr, .. } => expr_uses_name(expr, name),
        AstExpr::Call { callee, args, .. } => {
            callee.name == name || args.iter().any(|a| expr_uses_name(a, name))
        }
        AstExpr::If { cond, then_block, else_block, .. } => {
            expr_uses_name(cond, name)
            || block_uses_name(then_block, name)
            || else_block.as_ref().map_or(false, |b| block_uses_name(b, name))
        }
        AstExpr::Block(b) => block_uses_name(b, name),
        AstExpr::When { scrutinee, arms, .. } => {
            expr_uses_name(scrutinee, name) || arms.iter().any(|a| arm_uses_name(a, name))
        }
        AstExpr::FieldAccess { base, .. } | AstExpr::TupleIndex { base, .. }
        | AstExpr::Index { base, .. } => expr_uses_name(base, name),
        AstExpr::ArrayLit { elems, .. } => elems.iter().any(|e| expr_uses_name(e, name)),
        AstExpr::Tuple { elements, .. } => elements.iter().any(|e| expr_uses_name(e, name)),
        AstExpr::Lambda { body, .. } => expr_uses_name(body, name),
        AstExpr::StructLit { fields, .. } => fields.iter().any(|(_, v)| expr_uses_name(v, name)),
        AstExpr::MethodCall { base, args, .. } => {
            expr_uses_name(base, name) || args.iter().any(|a| expr_uses_name(a, name))
        }
    }
}

fn arm_uses_name(arm: &AstWhenArm, name: &str) -> bool {
    arm.guard.as_ref().map_or(false, |g| expr_uses_name(g, name))
    || expr_uses_name(&arm.body, name)
}
