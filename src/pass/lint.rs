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

use crate::error::PassError;
use crate::ir::module::IrModule;
use crate::pass::Pass;

/// IR-level Lint Pass for static analysis (e.g., unused variables).
pub struct IrLintPass;

impl Pass for IrLintPass {
    fn name(&self) -> &'static str {
        "lint"
    }

    fn run(&mut self, _module: &mut IrModule) -> Result<(), PassError> {
        // IR-level linting logic can be added here.
        // Currently, AST-level linting is primarily used.
        Ok(())
    }
}

/// Analyze an AST module and return dead variable warnings.
pub fn find_unused_vars(module: &AstModule) -> Vec<IrWarning> {
    let mut warnings = Vec::new();
    for func in &module.functions {
        collect_unused_in_function(func, &mut warnings);
        check_potential_infinite_loops(func, &mut warnings);
    }
    warnings
}

fn collect_unused_in_function(func: &AstFunction, warnings: &mut Vec<IrWarning>) {
    let declared: Vec<(String, Option<Span>)> = func
        .body
        .stmts
        .iter()
        .filter_map(|s| {
            if let AstStmt::Let { name, span, .. } = s {
                Some((name.name.clone(), Some(*span)))
            } else {
                None
            }
        })
        .collect();

    for (name, span) in declared {
        // Skip names starting with '_' — convention for intentionally unused.
        if name.starts_with('_') {
            continue;
        }
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
        if stmt_uses_name(stmt, name) {
            return true;
        }
    }
    if let Some(tail) = &block.tail {
        if expr_uses_name(tail, name) {
            return true;
        }
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
        AstStmt::ForRange {
            start, end, body, ..
        } => {
            expr_uses_name(start, name) || expr_uses_name(end, name) || block_uses_name(body, name)
        }
        AstStmt::ForEach { iter, body, .. } => {
            expr_uses_name(iter, name) || block_uses_name(body, name)
        }
        AstStmt::Assign { target, value, .. } => {
            expr_uses_name(target, name) || expr_uses_name(value, name)
        }
        AstStmt::LetTuple { init, .. } => expr_uses_name(init, name),
        AstStmt::Return { value, .. } => value.as_ref().is_some_and(|e| expr_uses_name(e, name)),
        AstStmt::Spawn { body, .. } => body.iter().any(|s| stmt_uses_name(s, name)),
        AstStmt::ParFor {
            start, end, body, ..
        } => {
            expr_uses_name(start, name) || expr_uses_name(end, name) || block_uses_name(body, name)
        }
        AstStmt::Break { .. } | AstStmt::Continue { .. } => false,
    }
}

fn expr_uses_name(expr: &AstExpr, name: &str) -> bool {
    match expr {
        AstExpr::Ident(ident) => ident.name == name,
        AstExpr::IntLit { .. }
        | AstExpr::FloatLit { .. }
        | AstExpr::BoolLit { .. }
        | AstExpr::StringLit { .. } => false,
        AstExpr::BinOp { lhs, rhs, .. } => expr_uses_name(lhs, name) || expr_uses_name(rhs, name),
        AstExpr::UnaryOp { expr, .. }
        | AstExpr::Cast { expr, .. }
        | AstExpr::Await { expr, .. }
        | AstExpr::Try { expr, .. } => expr_uses_name(expr, name),
        AstExpr::Call { callee, args, .. } => {
            callee.name == name || args.iter().any(|a| expr_uses_name(a, name))
        }
        AstExpr::If {
            cond,
            then_block,
            else_block,
            ..
        } => {
            expr_uses_name(cond, name)
                || block_uses_name(then_block, name)
                || else_block
                    .as_ref()
                    .is_some_and(|b| block_uses_name(b, name))
        }
        AstExpr::Block(b) => block_uses_name(b, name),
        AstExpr::When {
            scrutinee, arms, ..
        } => expr_uses_name(scrutinee, name) || arms.iter().any(|a| arm_uses_name(a, name)),
        AstExpr::FieldAccess { base, .. }
        | AstExpr::TupleIndex { base, .. }
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
    arm.guard.as_ref().is_some_and(|g| expr_uses_name(g, name)) || expr_uses_name(&arm.body, name)
}

// ---------------------------------------------------------------------------
// Infinite loop detection
// ---------------------------------------------------------------------------

fn check_potential_infinite_loops(func: &AstFunction, warnings: &mut Vec<IrWarning>) {
    check_infinite_loops_in_block(&func.body, &func.name.name, warnings);
}

fn check_infinite_loops_in_block(block: &AstBlock, func_name: &str, warnings: &mut Vec<IrWarning>) {
    for stmt in &block.stmts {
        match stmt {
            AstStmt::While { cond, body, span } => {
                let cond_vars = collect_idents_in_expr(cond);
                if !cond_vars.is_empty() {
                    let mutated = cond_vars.iter().any(|v| body_assigns_var(body, v));
                    let exits = body_has_exit(body);
                    if !mutated && !exits {
                        warnings.push(IrWarning {
                            func: func_name.to_string(),
                            message: format!(
                                "possible infinite loop: '{}' is never modified in the loop body",
                                cond_vars.join("', '")
                            ),
                            span: Some(*span),
                        });
                    }
                }
                check_infinite_loops_in_block(body, func_name, warnings);
            }
            AstStmt::ForRange { body, .. }
            | AstStmt::ForEach { body, .. }
            | AstStmt::ParFor { body, .. }
            | AstStmt::Loop { body, .. } => {
                check_infinite_loops_in_block(body, func_name, warnings);
            }
            _ => {}
        }
    }
}

/// Collect all simple identifier names referenced in an expression.
fn collect_idents_in_expr(expr: &AstExpr) -> Vec<String> {
    let mut out = Vec::new();
    collect_idents_recursive(expr, &mut out);
    out
}

fn collect_idents_recursive(expr: &AstExpr, out: &mut Vec<String>) {
    match expr {
        AstExpr::Ident(id) => out.push(id.name.clone()),
        AstExpr::BinOp { lhs, rhs, .. } => {
            collect_idents_recursive(lhs, out);
            collect_idents_recursive(rhs, out);
        }
        AstExpr::UnaryOp { expr, .. }
        | AstExpr::Cast { expr, .. }
        | AstExpr::Await { expr, .. }
        | AstExpr::Try { expr, .. } => collect_idents_recursive(expr, out),
        AstExpr::Call { args, .. } => args.iter().for_each(|a| collect_idents_recursive(a, out)),
        AstExpr::MethodCall { base, args, .. } => {
            collect_idents_recursive(base, out);
            args.iter().for_each(|a| collect_idents_recursive(a, out));
        }
        AstExpr::FieldAccess { base, .. }
        | AstExpr::TupleIndex { base, .. }
        | AstExpr::Index { base, .. } => collect_idents_recursive(base, out),
        _ => {}
    }
}

/// Returns true if `name` is assigned (mutated) anywhere in `block`.
fn body_assigns_var(block: &AstBlock, name: &str) -> bool {
    block.stmts.iter().any(|s| stmt_assigns_var(s, name))
}

fn stmt_assigns_var(stmt: &AstStmt, name: &str) -> bool {
    match stmt {
        AstStmt::Assign { target, .. } => expr_uses_name(target, name),
        AstStmt::Let { name: n, .. } => n.name == name,
        AstStmt::While { body, .. }
        | AstStmt::Loop { body, .. }
        | AstStmt::ForRange { body, .. }
        | AstStmt::ForEach { body, .. }
        | AstStmt::ParFor { body, .. } => body_assigns_var(body, name),
        AstStmt::Spawn { body, .. } => body.iter().any(|s| stmt_assigns_var(s, name)),
        // Assignments can live inside if/else or block expressions.
        AstStmt::Expr(e) => expr_assigns_var(e, name),
        _ => false,
    }
}

/// Returns true if `expr` contains an assignment to `name` (recursing into
/// if/else branches and blocks, which is where conditional mutations live).
fn expr_assigns_var(expr: &AstExpr, name: &str) -> bool {
    match expr {
        AstExpr::If {
            then_block,
            else_block,
            ..
        } => {
            body_assigns_var(then_block, name)
                || else_block
                    .as_ref()
                    .is_some_and(|b| body_assigns_var(b, name))
        }
        AstExpr::Block(b) => body_assigns_var(b, name),
        AstExpr::When { arms, .. } => arms.iter().any(|a| expr_assigns_var(&a.body, name)),
        _ => false,
    }
}

/// Returns true if `block` contains a `break` or `return` at the top level
/// (not inside a nested loop, which would break the inner loop instead).
fn body_has_exit(block: &AstBlock) -> bool {
    block.stmts.iter().any(stmt_has_exit)
}

fn stmt_has_exit(stmt: &AstStmt) -> bool {
    match stmt {
        AstStmt::Break { .. } | AstStmt::Return { .. } => true,
        // An if-expression may contain a break — check both branches.
        AstStmt::Expr(e) => expr_has_exit(e),
        // Nested loops own their own break — don't propagate.
        AstStmt::While { .. }
        | AstStmt::Loop { .. }
        | AstStmt::ForRange { .. }
        | AstStmt::ForEach { .. }
        | AstStmt::ParFor { .. } => false,
        _ => false,
    }
}

fn expr_has_exit(expr: &AstExpr) -> bool {
    match expr {
        AstExpr::If {
            then_block,
            else_block,
            ..
        } => body_has_exit(then_block) || else_block.as_ref().is_some_and(body_has_exit),
        AstExpr::Block(b) => body_has_exit(b),
        _ => false,
    }
}
