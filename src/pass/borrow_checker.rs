use crate::error::describe_location;
use crate::parser::ast::*;
use crate::parser::lexer::Span;

#[derive(Debug, Clone)]
struct Borrow {
    #[allow(dead_code)]
    var_name: String,
    is_mut: bool,
    scope_depth: usize,
    span: Span,
}

#[derive(Debug, Clone)]
pub enum BorrowError {
    MutBorrowWhileBorrowed {
        var: String,
        borrow_span: Span,
        mut_span: Span,
    },
    BorrowWhileMutBorrowed {
        var: String,
        borrow_span: Span,
        mut_span: Span,
    },
    MutateWhileBorrowed {
        var: String,
        borrow_span: Span,
        mutate_span: Span,
    },
    MoveWhileBorrowed {
        var: String,
        borrow_span: Span,
        move_span: Span,
    },
    UseAfterMove {
        var: String,
        move_span: Span,
        use_span: Span,
    },
    BorrowAfterMove {
        var: String,
        move_span: Span,
        borrow_span: Span,
    },
    MoveAfterMove {
        var: String,
        first_move: Span,
        second_move: Span,
    },
}

impl BorrowError {
    fn var_name(&self) -> &str {
        match self {
            BorrowError::MutBorrowWhileBorrowed { var, .. }
            | BorrowError::BorrowWhileMutBorrowed { var, .. }
            | BorrowError::MutateWhileBorrowed { var, .. }
            | BorrowError::MoveWhileBorrowed { var, .. }
            | BorrowError::UseAfterMove { var, .. }
            | BorrowError::BorrowAfterMove { var, .. }
            | BorrowError::MoveAfterMove { var, .. } => var,
        }
    }

    fn conflicting_span(&self) -> Span {
        match self {
            BorrowError::MutBorrowWhileBorrowed { borrow_span, .. }
            | BorrowError::BorrowWhileMutBorrowed { borrow_span, .. }
            | BorrowError::MutateWhileBorrowed { borrow_span, .. }
            | BorrowError::MoveWhileBorrowed { borrow_span, .. } => *borrow_span,
            BorrowError::UseAfterMove { move_span, .. }
            | BorrowError::BorrowAfterMove { move_span, .. } => *move_span,
            BorrowError::MoveAfterMove { first_move, .. } => *first_move,
        }
    }

    #[allow(dead_code)]
    fn offending_span(&self) -> Span {
        match self {
            BorrowError::MutBorrowWhileBorrowed { mut_span, .. } => *mut_span,
            BorrowError::BorrowWhileMutBorrowed { mut_span, .. } => *mut_span,
            BorrowError::MutateWhileBorrowed { mutate_span, .. } => *mutate_span,
            BorrowError::MoveWhileBorrowed { move_span, .. } => *move_span,
            BorrowError::UseAfterMove { use_span, .. } => *use_span,
            BorrowError::BorrowAfterMove { borrow_span, .. } => *borrow_span,
            BorrowError::MoveAfterMove { second_move, .. } => *second_move,
        }
    }
}

impl std::fmt::Display for BorrowError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BorrowError::MutBorrowWhileBorrowed { var, .. } => {
                write!(
                    f,
                    "cannot borrow `{}` as mutable because it is already borrowed as immutable",
                    var
                )
            }
            BorrowError::BorrowWhileMutBorrowed { var, .. } => {
                write!(
                    f,
                    "cannot borrow `{}` as immutable because it is already borrowed as mutable",
                    var
                )
            }
            BorrowError::MutateWhileBorrowed { var, .. } => {
                write!(
                    f,
                    "cannot mutate `{}` because it is currently borrowed",
                    var
                )
            }
            BorrowError::MoveWhileBorrowed { var, .. } => {
                write!(
                    f,
                    "cannot move `{}` because it is currently borrowed",
                    var
                )
            }
            BorrowError::UseAfterMove { var, .. } => {
                write!(
                    f,
                    "use of moved value `{}`",
                    var
                )
            }
            BorrowError::BorrowAfterMove { var, .. } => {
                write!(
                    f,
                    "cannot borrow `{}` because it has been moved",
                    var
                )
            }
            BorrowError::MoveAfterMove { var, .. } => {
                write!(
                    f,
                    "cannot move `{}` because it has already been moved",
                    var
                )
            }
        }
    }
}

pub struct BorrowChecker {
    borrows: std::collections::HashMap<String, Vec<Borrow>>,
    moved: std::collections::HashSet<String>,
    scope_depth: usize,
    errors: Vec<(BorrowError, Span)>,
    has_ref_types: bool,
    source: Option<String>,
}

impl BorrowChecker {
    pub fn new() -> Self {
        Self {
            borrows: std::collections::HashMap::new(),
            moved: std::collections::HashSet::new(),
            scope_depth: 0,
            errors: Vec::new(),
            has_ref_types: false,
            source: None,
        }
    }

    pub fn with_source(mut self, source: String) -> Self {
        self.source = Some(source);
        self
    }

    pub fn check_module(&mut self, ast: &AstModule) {
        for func in &ast.functions {
            self.check_function(func);
        }
    }

    pub fn errors(&self) -> &[(BorrowError, Span)] {
        &self.errors
    }

    pub fn has_ref_types(&self) -> bool {
        self.has_ref_types
    }

    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    fn loc(&self, byte: u32) -> String {
        describe_location(self.source.as_deref(), byte)
    }

    fn source_line(&self, span: Span) -> Option<String> {
        let src = self.source.as_ref()?;
        let start = span.start.0 as usize;
        let end = std::cmp::min(span.end.0 as usize, src.len());
        if start >= src.len() {
            return None;
        }
        let line_start = src[..start].rfind('\n').map(|i| i + 1).unwrap_or(0);
        let line_end = src[end..].find('\n').map(|i| end + i).unwrap_or(src.len());
        let line_text = src[line_start..line_end].trim_end();
        if line_text.is_empty() {
            return None;
        }
        let col = start - line_start;
        let len = end - start;
        let line_num = src[..start].bytes().filter(|&b| b == b'\n').count() + 1;
        let underline = format!("{: >width$}{:^^len$}", "", "", width = col, len = len.max(1));
        Some(format!(
            "  --> {}:{}\n       |\n {: >4} | {}\n       | {}",
            line_num,
            col + 1,
            line_num,
            line_text,
            underline
        ))
    }

    fn format_error(&self, err: &BorrowError, offending: Span) -> String {
        let var = err.var_name();
        let conflicting = err.conflicting_span();

        let primary_msg = format!("{}", err);
        let primary_loc = self.loc(offending.start.0);
        let conflicting_loc = self.loc(conflicting.start.0);

        let mut out = format!("[borrow error] {} (at {})", primary_msg, primary_loc);

        if conflicting_loc != primary_loc {
            let kind = match err {
                BorrowError::MutBorrowWhileBorrowed { .. } => "immutable borrow",
                BorrowError::BorrowWhileMutBorrowed { .. } => "mutable borrow",
                BorrowError::MutateWhileBorrowed { .. } => "first borrow",
                BorrowError::MoveWhileBorrowed { .. } => "first borrow",
                BorrowError::UseAfterMove { .. } => "move",
                BorrowError::BorrowAfterMove { .. } => "move",
                BorrowError::MoveAfterMove { .. } => "first move",
            };
            out.push_str(&format!(
                "\n  note: {} of `{}` created here (at {})",
                kind, var, conflicting_loc
            ));
        }

        if let Some(ctx) = self.source_line(conflicting) {
            out.push_str(&format!("\n{}", ctx));
        }
        if let Some(ctx) = self.source_line(offending) {
            out.push_str(&format!("\n{}", ctx));
        }

        out
    }

    pub fn print_errors(&self) {
        for (err, span) in &self.errors {
            eprintln!("{}", self.format_error(err, *span));
        }
    }

    fn check_function(&mut self, func: &AstFunction) {
        self.scope_depth = 0;
        self.borrows.clear();
        self.moved.clear();

        for param in &func.params {
            self.scan_type_for_refs(&param.ty);
            self.add_var(&param.name.name);
        }

        self.scan_type_for_refs(&func.return_ty);
        self.check_block(&func.body);
    }

    fn scan_type_for_refs(&mut self, ty: &AstType) {
        match ty {
            AstType::Ref(inner, _) | AstType::RefMut(inner, _) => {
                self.has_ref_types = true;
                self.scan_type_for_refs(inner);
            }
            AstType::Option(inner, _) | AstType::Chan(inner, _) | AstType::Atomic(inner, _)
            | AstType::Mutex(inner, _) | AstType::Grad(inner, _) | AstType::Sparse(inner, _)
            | AstType::List(inner, _) | AstType::WeakRef(inner, _) => {
                self.scan_type_for_refs(inner)
            }
            AstType::Result(ok, err, _) => {
                self.scan_type_for_refs(ok);
                self.scan_type_for_refs(err);
            }
            AstType::Map(k, v, _) => {
                self.scan_type_for_refs(k);
                self.scan_type_for_refs(v);
            }
            AstType::Tuple(elems, _) => {
                for e in elems {
                    self.scan_type_for_refs(e);
                }
            }
            AstType::Fn { params, ret, .. } => {
                for p in params {
                    self.scan_type_for_refs(p);
                }
                self.scan_type_for_refs(ret);
            }
            AstType::Array { elem, .. } => self.scan_type_for_refs(elem),
            AstType::Generic { args, .. } => {
                for a in args {
                    self.scan_type_for_refs(a);
                }
            }
            _ => {}
        }
    }

    fn check_block(&mut self, block: &AstBlock) {
        self.scope_depth += 1;
        for stmt in &block.stmts {
            self.check_stmt(stmt);
        }
        if let Some(result) = &block.tail {
            self.check_expr(result);
        }
        self.scope_depth -= 1;
        self.clear_scope_borrows();
    }

    fn check_stmt(&mut self, stmt: &AstStmt) {
        match stmt {
            AstStmt::Let { name, init, ty, .. } => {
                if let Some(t) = ty {
                    self.scan_type_for_refs(t);
                }
                self.check_expr(init);
                self.add_var(&name.name);
            }
            AstStmt::Expr(e) => {
                self.check_expr(e);
            }
            AstStmt::Assign { target, value, .. } => {
                self.check_expr(value);
                if let AstExpr::Ident(ident) = target.as_ref() {
                    self.check_use_after_move(&ident.name, target.span());
                    if let Some(borrows) = self.borrows.get(&ident.name) {
                        if !borrows.is_empty() {
                            self.errors.push((
                                BorrowError::MutateWhileBorrowed {
                                    var: ident.name.clone(),
                                    borrow_span: borrows[0].span,
                                    mutate_span: target.span(),
                                },
                                target.span(),
                            ));
                        }
                    }
                }
            }
            AstStmt::Return { value, .. } => {
                if let Some(v) = value {
                    self.check_expr(v);
                }
            }
            AstStmt::While { cond, body, .. } => {
                self.check_expr(cond);
                self.check_block(body);
            }
            AstStmt::Loop { body, .. } => {
                self.check_block(body);
            }
            AstStmt::ForRange { start, end, body, .. } => {
                self.check_expr(start);
                self.check_expr(end);
                self.check_block(body);
            }
            AstStmt::ForEach { iter, body, .. } => {
                self.check_expr(iter);
                self.check_block(body);
            }
            AstStmt::ParFor { start, end, body, .. } => {
                self.check_expr(start);
                self.check_expr(end);
                self.check_block(body);
            }
            AstStmt::Spawn { body, group, .. } => {
                for s in body {
                    self.check_stmt(s);
                }
                if let Some(g) = group {
                    self.check_expr(g);
                }
            }
            AstStmt::LetTuple { init, .. } => {
                self.check_expr(init);
            }
            AstStmt::MaskStmt { body, .. } => {
                self.check_block(body);
            }
            AstStmt::HandleStmt { expr, arms, .. } => {
                self.check_expr(expr);
                for arm in arms {
                    self.check_expr(&arm.body);
                }
            }
            AstStmt::Break { .. } | AstStmt::Continue { .. } => {}
            AstStmt::Defer { expr, .. } => {
                self.check_expr(expr);
            }
            AstStmt::Yield { expr, .. } => {
                self.check_expr(expr);
            }
            AstStmt::Select { arms, default, .. } => {
                for arm in arms {
                    self.check_expr(&arm.channel);
                    self.check_block(&arm.body);
                }
                if let Some(d) = default {
                    self.check_block(d);
                }
            }
        }
    }

    fn check_expr(&mut self, expr: &AstExpr) {
        match expr {
            AstExpr::Ref { expr: inner, span } => {
                self.has_ref_types = true;
                if let AstExpr::Ident(ident) = inner.as_ref() {
                    self.check_borrow_after_move(&ident.name, *span);
                    if let Some(borrows) = self.borrows.get(&ident.name) {
                        if borrows.iter().any(|b| b.is_mut) {
                            self.errors.push((
                                BorrowError::BorrowWhileMutBorrowed {
                                    var: ident.name.clone(),
                                    borrow_span: borrows.iter().find(|b| b.is_mut).unwrap().span,
                                    mut_span: *span,
                                },
                                *span,
                            ));
                        }
                    }
                    self.add_borrow(ident.name.clone(), false, *span);
                } else {
                    self.check_expr(inner);
                }
            }
            AstExpr::RefMut { expr: inner, span } => {
                self.has_ref_types = true;
                if let AstExpr::Ident(ident) = inner.as_ref() {
                    self.check_borrow_after_move(&ident.name, *span);
                    if let Some(borrows) = self.borrows.get(&ident.name) {
                        if !borrows.is_empty() {
                            self.errors.push((
                                BorrowError::MutBorrowWhileBorrowed {
                                    var: ident.name.clone(),
                                    borrow_span: borrows[0].span,
                                    mut_span: *span,
                                },
                                *span,
                            ));
                        }
                    }
                    self.add_borrow(ident.name.clone(), true, *span);
                } else {
                    self.check_expr(inner);
                }
            }
            AstExpr::Deref { expr: inner, .. } => {
                self.check_expr(inner);
            }
            AstExpr::Move { expr: inner, span } => {
                // Check for borrow conflicts first
                if let AstExpr::Ident(ident) = inner.as_ref() {
                    if let Some(borrows) = self.borrows.get(&ident.name) {
                        if !borrows.is_empty() {
                            self.errors.push((
                                BorrowError::MoveWhileBorrowed {
                                    var: ident.name.clone(),
                                    borrow_span: borrows[0].span,
                                    move_span: *span,
                                },
                                *span,
                            ));
                        }
                    }
                    self.mark_moved(&ident.name, *span);
                } else {
                    self.check_expr(inner);
                }
            }
            AstExpr::Ident(ident) => {
                self.check_use_after_move(&ident.name, ident.span);
            }
            AstExpr::IntLit { .. } | AstExpr::FloatLit { .. }
            | AstExpr::BoolLit { .. } | AstExpr::StringLit { .. } => {}
            AstExpr::BinOp { lhs, rhs, .. } => {
                self.check_expr(lhs);
                self.check_expr(rhs);
            }
            AstExpr::UnaryOp { expr, .. } => {
                self.check_expr(expr);
            }
            AstExpr::Call { args, named_args, .. } => {
                for arg in args {
                    self.check_expr(arg);
                }
                for (_, arg) in named_args {
                    self.check_expr(arg);
                }
            }
            AstExpr::MethodCall { base, args, .. } => {
                self.check_expr(base);
                for arg in args {
                    self.check_expr(arg);
                }
            }
            AstExpr::If {
                cond,
                then_block,
                else_block,
                ..
            } => {
                self.check_expr(cond);
                self.check_block(then_block);
                if let Some(eb) = else_block {
                    self.check_block(eb);
                }
            }
            AstExpr::Block(block) => {
                self.check_block(block);
            }
            AstExpr::Lambda { body, .. } => {
                self.check_expr(body);
            }
            AstExpr::Index { base, indices, .. } => {
                self.check_expr(base);
                for idx in indices {
                    self.check_expr(idx);
                }
            }
            AstExpr::Cast { expr, .. } => {
                self.check_expr(expr);
            }
            AstExpr::StructLit {
                fields,
                spread,
                ..
            } => {
                for (_, v) in fields {
                    self.check_expr(v);
                }
                if let Some(s) = spread {
                    self.check_expr(s);
                }
            }
            AstExpr::FieldAccess { base, .. } => {
                self.check_expr(base);
            }
            AstExpr::When {
                scrutinee, arms, ..
            } => {
                self.check_expr(scrutinee);
                for arm in arms {
                    self.check_expr(&arm.body);
                }
            }
            AstExpr::Tuple { elements, .. } => {
                for e in elements {
                    self.check_expr(e);
                }
            }
            AstExpr::TupleIndex { base, .. } => {
                self.check_expr(base);
            }
            AstExpr::ArrayLit { elems, .. } => {
                for e in elems {
                    self.check_expr(e);
                }
            }
            AstExpr::MapLiteral { entries, .. } => {
                for (k, v) in entries {
                    self.check_expr(k);
                    self.check_expr(v);
                }
            }
            AstExpr::Await { expr, .. } => {
                self.check_expr(expr);
            }
            AstExpr::Try { expr, .. } => {
                self.check_expr(expr);
            }
            AstExpr::Mask { body, .. } => {
                self.check_block(body);
            }
            AstExpr::Handle { expr, arms, .. } => {
                self.check_expr(expr);
                for arm in arms {
                    self.check_expr(&arm.body);
                }
            }
            AstExpr::NullCoal { expr, default, .. } => {
                self.check_expr(expr);
                self.check_expr(default);
            }
            AstExpr::TryCatch { body, catch_body, .. } => {
                self.check_expr(body);
                self.check_expr(catch_body);
            }
            AstExpr::Raise { args, .. } => {
                for a in args {
                    self.check_expr(a);
                }
            }
            AstExpr::Unsafe { body, .. } => {
                self.check_expr(body);
            }
            AstExpr::Splat { expr, .. } => {
                self.check_expr(expr);
            }
            AstExpr::MacroCall { args, .. } => {
                for a in args {
                    self.check_expr(a);
                }
            }
        }
    }

    fn add_var(&mut self, name: &str) {
        self.borrows.entry(name.to_string()).or_default();
    }

    fn add_borrow(&mut self, var_name: String, is_mut: bool, span: Span) {
        self.borrows
            .entry(var_name.clone())
            .or_default()
            .push(Borrow {
                var_name,
                is_mut,
                scope_depth: self.scope_depth,
                span,
            });
    }

    fn clear_scope_borrows(&mut self) {
        for borrows in self.borrows.values_mut() {
            borrows.retain(|b| b.scope_depth < self.scope_depth);
        }
    }

    #[allow(dead_code)]
    fn is_moved(&self, name: &str) -> bool {
        self.moved.contains(name)
    }

    fn mark_moved(&mut self, name: &str, span: Span) {
        if self.moved.contains(name) {
            self.errors.push((
                BorrowError::MoveAfterMove {
                    var: name.to_string(),
                    first_move: span,
                    second_move: span,
                },
                span,
            ));
        } else {
            self.moved.insert(name.to_string());
        }
    }

    fn check_use_after_move(&mut self, name: &str, span: Span) {
        if self.moved.contains(name) {
            self.errors.push((
                BorrowError::UseAfterMove {
                    var: name.to_string(),
                    move_span: span,
                    use_span: span,
                },
                span,
            ));
        }
    }

    fn check_borrow_after_move(&mut self, name: &str, span: Span) {
        if self.moved.contains(name) {
            self.errors.push((
                BorrowError::BorrowAfterMove {
                    var: name.to_string(),
                    move_span: span,
                    borrow_span: span,
                },
                span,
            ));
        }
    }
}
