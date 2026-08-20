// Effect inference and verification pass.
//
// Walks the AST bottom-up, computing an effect row for every function
// (the union of callee effect rows), and verifies at each call site
// that the caller's effect row covers the callee's.
//
// Backward-compat: functions without an explicit `effect` clause are
// auto-promoted to their inferred effect row (no error). Strict mode
// (`--strict-effects`) requires explicit clauses on effectful functions.

use crate::parser::ast::*;
use std::collections::{HashMap, HashSet};
use super::effect_registry::{EffectRegistry, EffectRow};

pub struct EffectChecker {
    pub registry: EffectRegistry,
    pub inferred: HashMap<String, EffectRow>,
    pub strict: bool,
    pub errors: Vec<String>,
}

impl EffectChecker {
    pub fn new(strict: bool) -> Self {
        Self {
            registry: EffectRegistry::new(),
            inferred: HashMap::new(),
            strict,
            errors: Vec::new(),
        }
    }

    /// Run the pass on an AST module. Collects errors instead of returning early.
    pub fn run(&mut self, ast: &AstModule) {
        let call_graph = self.build_call_graph(ast);
        let order = self.topological_sort(&call_graph);

        let mut in_progress: HashSet<String> = HashSet::new();
        for name in &order {
            self.infer_function(ast, name, &call_graph, &mut in_progress);
        }
        // Second pass: revisit any function still missing (recursion loop).
        for name in call_graph.keys() {
            if !self.inferred.contains_key(name) {
                self.infer_function(ast, name, &call_graph, &mut HashSet::new());
            }
        }

        // Verify call sites.
        for func in &ast.functions {
            self.verify_call_sites(&func.name.name, &func.body);
        }
        for impl_def in &ast.impls {
            for method in &impl_def.methods {
                let mangled = Self::mangle(impl_def, method);
                self.verify_call_sites(&mangled, &method.body);
            }
        }
    }

    fn build_call_graph(&self, ast: &AstModule) -> HashMap<String, HashSet<String>> {
        let mut graph: HashMap<String, HashSet<String>> = HashMap::new();
        for func in &ast.functions {
            let mut callees = HashSet::new();
            self.collect_callees_block(&func.body, &mut callees);
            graph.insert(func.name.name.clone(), callees);
        }
        for impl_def in &ast.impls {
            for method in &impl_def.methods {
                let mangled = Self::mangle(impl_def, method);
                let mut callees = HashSet::new();
                self.collect_callees_block(&method.body, &mut callees);
                graph.insert(mangled, callees.clone());
                graph.insert(method.name.name.clone(), callees);
            }
        }
        graph
    }

    fn collect_callees_block(&self, body: &AstBlock, callees: &mut HashSet<String>) {
        for stmt in &body.stmts {
            self.collect_callees_stmt(stmt, callees);
        }
        if let Some(expr) = &body.tail {
            self.collect_callees_expr(expr, callees);
        }
    }

    fn collect_callees_stmt(&self, stmt: &AstStmt, callees: &mut HashSet<String>) {
        match stmt {
            AstStmt::Let { init, .. } => self.collect_callees_expr(init, callees),
            AstStmt::Expr(e) => self.collect_callees_expr(e, callees),
            AstStmt::While { cond, body, .. } => {
                self.collect_callees_expr(cond, callees);
                self.collect_callees_block(body, callees);
            }
            AstStmt::Loop { body, .. } => self.collect_callees_block(body, callees),
            AstStmt::Break { .. } | AstStmt::Continue { .. } => {}
            AstStmt::ForRange { start, end, body, .. } => {
                self.collect_callees_expr(start, callees);
                self.collect_callees_expr(end, callees);
                self.collect_callees_block(body, callees);
            }
            AstStmt::ForEach { iter, body, .. } => {
                self.collect_callees_expr(iter, callees);
                self.collect_callees_block(body, callees);
            }
            AstStmt::Assign { value, .. } => self.collect_callees_expr(value, callees),
            AstStmt::LetTuple { init, .. } => self.collect_callees_expr(init, callees),
            AstStmt::Return { value, .. } => {
                if let Some(v) = value {
                    self.collect_callees_expr(v, callees);
                }
            }
            AstStmt::Spawn { body, .. } => {
                for s in body {
                    self.collect_callees_stmt(s, callees);
                }
            }
            AstStmt::MaskStmt { body, .. } => {
                self.collect_callees_block(body, callees);
            }
            AstStmt::HandleStmt { expr, arms, .. } => {
                self.collect_callees_expr(expr, callees);
                for arm in arms {
                    self.collect_callees_expr(&arm.body, callees);
                }
            }
            AstStmt::ParFor { start, end, body, .. } => {
                self.collect_callees_expr(start, callees);
                self.collect_callees_expr(end, callees);
                self.collect_callees_block(body, callees);
            }
            AstStmt::Defer { expr, .. } => self.collect_callees_expr(expr, callees),
            AstStmt::Yield { expr, .. } => self.collect_callees_expr(expr, callees),
            AstStmt::Select { arms, default, .. } => {
                for arm in arms {
                    self.collect_callees_expr(&arm.channel, callees);
                    self.collect_callees_block(&arm.body, callees);
                }
                if let Some(d) = default {
                    self.collect_callees_block(d, callees);
                }
            }
        }
    }

    fn collect_callees_expr(&self, expr: &AstExpr, callees: &mut HashSet<String>) {
        match expr {
            AstExpr::Call { callee, args, .. } => {
                callees.insert(callee.name.clone());
                for a in args {
                    self.collect_callees_expr(a, callees);
                }
            }
            AstExpr::MethodCall { base, args, .. } => {
                // Look up by method name only; impl resolution happens at monomorphization.
                self.collect_callees_expr(base, callees);
                for a in args {
                    self.collect_callees_expr(a, callees);
                }
            }
            AstExpr::BinOp { lhs, rhs, .. } => {
                self.collect_callees_expr(lhs, callees);
                self.collect_callees_expr(rhs, callees);
            }
            AstExpr::UnaryOp { expr, .. } => self.collect_callees_expr(expr, callees),
            AstExpr::If { cond, then_block, else_block, .. } => {
                self.collect_callees_expr(cond, callees);
                self.collect_callees_block(then_block, callees);
                if let Some(eb) = else_block {
                    self.collect_callees_block(eb, callees);
                }
            }
            AstExpr::Block(b) => self.collect_callees_block(b, callees),
            AstExpr::Mask { body, .. } => self.collect_callees_block(body, callees),
            AstExpr::Handle { expr, arms, .. } => {
                self.collect_callees_expr(expr, callees);
                for arm in arms {
                    self.collect_callees_expr(&arm.body, callees);
                }
            }
            AstExpr::Lambda { body, .. } => self.collect_callees_expr(body, callees),
            AstExpr::Tuple { elements, .. } => {
                for e in elements {
                    self.collect_callees_expr(e, callees);
                }
            }
            AstExpr::ArrayLit { elems, .. } => {
                for e in elems {
                    self.collect_callees_expr(e, callees);
                }
            }
            AstExpr::FieldAccess { base, .. } => self.collect_callees_expr(base, callees),
            AstExpr::Index { base, indices, .. } => {
                self.collect_callees_expr(base, callees);
                for i in indices {
                    self.collect_callees_expr(i, callees);
                }
            }
            AstExpr::When { scrutinee, arms, .. } => {
                self.collect_callees_expr(scrutinee, callees);
                for arm in arms {
                    if let Some(g) = &arm.guard {
                        self.collect_callees_expr(g, callees);
                    }
                    self.collect_callees_expr(&arm.body, callees);
                }
            }
            AstExpr::Cast { expr, .. } => self.collect_callees_expr(expr, callees),
            AstExpr::StructLit { fields, spread, .. } => {
                for (_, v) in fields {
                    self.collect_callees_expr(v, callees);
                }
                if let Some(s) = spread {
                    self.collect_callees_expr(s, callees);
                }
            }
            AstExpr::Try { expr, .. } => self.collect_callees_expr(expr, callees),
            AstExpr::Await { expr, .. } => self.collect_callees_expr(expr, callees),
            AstExpr::Ident(_) | AstExpr::IntLit { .. } | AstExpr::FloatLit { .. } |
            AstExpr::BoolLit { .. } | AstExpr::StringLit { .. } |
            AstExpr::TupleIndex { .. } => {}
            AstExpr::NullCoal { expr, default, .. } => {
                self.collect_callees_expr(expr, callees);
                self.collect_callees_expr(default, callees);
            }
            AstExpr::MapLiteral { entries, .. } => {
                for (k, v) in entries {
                    self.collect_callees_expr(k, callees);
                    self.collect_callees_expr(v, callees);
                }
            }
            AstExpr::Ref { expr, .. }
            | AstExpr::RefMut { expr, .. }
            | AstExpr::Deref { expr, .. }
            | AstExpr::Move { expr, .. } => {
                self.collect_callees_expr(expr, callees);
            }
            AstExpr::Unsafe { body, .. } => {
                self.collect_callees_expr(body, callees);
            }
            AstExpr::Splat { expr, .. } => {
                self.collect_callees_expr(expr, callees);
            }
            AstExpr::TryCatch { body, catch_body, .. } => {
                self.collect_callees_expr(body, callees);
                self.collect_callees_expr(catch_body, callees);
            }
            AstExpr::Raise { args, .. } => {
                for a in args {
                    self.collect_callees_expr(a, callees);
                }
            }
            AstExpr::MacroCall { args, .. } => {
                for a in args {
                    self.collect_callees_expr(a, callees);
                }
            }
        }
    }

    /// Iterative post-order DFS for topological sort. Leaves come first
    /// (we want bottom-up inference).
    fn topological_sort(&self, graph: &HashMap<String, HashSet<String>>) -> Vec<String> {
        let mut visited: HashSet<String> = HashSet::new();
        let mut stack: HashSet<String> = HashSet::new();
        let mut order: Vec<String> = Vec::new();
        for name in graph.keys() {
            self.dfs_postorder(name, graph, &mut visited, &mut stack, &mut order);
        }
        order
    }

    fn dfs_postorder(
        &self,
        name: &str,
        graph: &HashMap<String, HashSet<String>>,
        visited: &mut HashSet<String>,
        stack: &mut HashSet<String>,
        order: &mut Vec<String>,
    ) {
        if visited.contains(name) || stack.contains(name) {
            return;
        }
        stack.insert(name.to_string());
        if let Some(callees) = graph.get(name) {
            for c in callees {
                self.dfs_postorder(c, graph, visited, stack, order);
            }
        }
        stack.remove(name);
        visited.insert(name.to_string());
        order.push(name.to_string());
    }

    fn infer_function(
        &mut self,
        ast: &AstModule,
        name: &str,
        call_graph: &HashMap<String, HashSet<String>>,
        _in_progress: &mut HashSet<String>,
    ) {
        let func = self.find_function(ast, name);

        if let Some(func) = func {
            // Parse declared effects: support `effect E` for effect vars.
            let mut effect_vars: Vec<String> = Vec::new();
            let mut concrete_effects: Vec<String> = Vec::new();
            for e in &func.effects {
                if e.chars().all(|c| c.is_ascii_uppercase() || c == '_' || c.is_ascii_digit()) {
                    effect_vars.push(e.clone());
                } else {
                    concrete_effects.push(e.clone());
                }
            }
            let declared = EffectRow::from_parts(concrete_effects.clone(), effect_vars.clone());
            // `from_callees` is what the body actually does, tracked apart from
            // what the signature claims. `inferred` still starts from `declared`
            // so an effect performed only on a conditional path still propagates
            // to callers — but seeding *both* from `declared` made the relation
            // `declared ⊇ inferred` true by construction, which is why no
            // subsumption violation could ever be detected.
            let mut from_callees = EffectRow::pure();
            let mut inferred = EffectRow::from_parts(concrete_effects.clone(), effect_vars.clone());
            if let Some(callees) = call_graph.get(name) {
                for callee in callees {
                    if let Some(row) = self.registry.lookup(callee) {
                        // Instantiate callee's effect vars with the current context.
                        let row = row.instantiate(&declared);
                        from_callees = from_callees.union(&row);
                        inferred = inferred.union(&row);
                    } else if let Some(row) = self.inferred.get(callee) {
                        let row = row.instantiate(&declared);
                        from_callees = from_callees.union(&row);
                        inferred = inferred.union(&row);
                    }
                }
            }

            self.inferred.insert(name.to_string(), inferred.clone());

            // Warn on unused declared effects (concrete only). Compared against
            // `from_callees`, not `inferred` — the latter contains the declared
            // row itself, so this warning could never fire. Strict mode only, to
            // keep default output unchanged.
            if self.strict && !concrete_effects.is_empty() {
                let unused: Vec<String> = concrete_effects
                    .iter()
                    .filter(|e| !from_callees.effects.contains(*e))
                    .cloned()
                    .collect();
                if !unused.is_empty() {
                    self.errors.push(format!(
                        "warning: function `{}` declares effect{} `{}` that the body doesn't use",
                        name,
                        if unused.len() == 1 { "" } else { "s" },
                        unused.join(", ")
                    ));
                }
            }

            // Strict mode: require explicit clause on effectful functions.
            if self.strict && func.effects.is_empty() && !inferred.is_pure() {
                self.errors.push(format!(
                    "error[E0301]: [effect check] function `{}` has effect{} `{}` from callees but no explicit `effect` clause (strict mode)",
                    name,
                    if inferred.effects.len() == 1 { "" } else { "s" },
                    inferred.display()
                ));
            }

            // Strict mode: the declared row must *cover* what the body does.
            //
            // Without this, an `effect` clause is documentation rather than a
            // proof: `def f() -> i64 effect throw { list_get(xs, 0) }` allocates
            // while declaring only `throw`, and used to pass silently. A row
            // variable (`effect E`) is row-polymorphic and absorbs anything, so
            // it is exempt.
            if self.strict && !func.effects.is_empty() && declared.vars.is_empty() {
                let missing: Vec<String> = from_callees
                    .effects
                    .iter()
                    .filter(|e| !declared.effects.contains(*e))
                    .cloned()
                    .collect();
                if !missing.is_empty() {
                    // E0303, not E0302 — E0302 is the call-site check below,
                    // which asks a different question (does the *caller* cover
                    // the callee). This one asks whether a function's own
                    // clause covers its own body.
                    self.errors.push(format!(
                        "error[E0303]: [effect check] function `{}` performs effect{} `{}` not covered by its `effect` clause `{}` (strict mode)",
                        name,
                        if missing.len() == 1 { "" } else { "s" },
                        missing.join(", "),
                        declared.display()
                    ));
                }
            }
        } else {
            // External / builtin.
            if let Some(row) = self.registry.lookup(name) {
                self.inferred.insert(name.to_string(), row.clone());
            } else {
                self.inferred.insert(name.to_string(), EffectRow::pure());
            }
        }
    }

    /// The name an impl method is known by.
    ///
    /// Built in exactly one place and used everywhere, because the alternative
    /// -- reconstructing it by splitting on `__` -- stops working the moment a
    /// module prefix is present, and a module prefix is itself joined with
    /// `__`. `container__Sized__list__size` split into
    /// ("container", "Sized", "list__size"), matched no impl, and the method's
    /// declared effects were therefore never read: every trait method in a
    /// brought module was treated as `pure`. See known-issues #39.
    fn mangle(impl_def: &AstImplDef, method: &AstFunction) -> String {
        if impl_def.trait_name.is_empty() {
            format!("{}__{}", impl_def.type_name, method.name.name)
        } else {
            format!(
                "{}__{}__{}",
                impl_def.trait_name, impl_def.type_name, method.name.name
            )
        }
    }

    fn find_function<'a>(&self, ast: &'a AstModule, name: &str) -> Option<&'a AstFunction> {
        if let Some(f) = ast.functions.iter().find(|f| f.name.name == name) {
            return Some(f);
        }
        for impl_def in &ast.impls {
            for method in &impl_def.methods {
                if Self::mangle(impl_def, method) == name {
                    return Some(method);
                }
            }
        }
        None
    }

    fn verify_call_sites(&mut self, caller_name: &str, body: &AstBlock) {
        let caller_row = self
            .inferred
            .get(caller_name)
            .cloned()
            .unwrap_or_else(EffectRow::pure);
        for stmt in &body.stmts {
            self.verify_stmt(caller_name, &caller_row, stmt);
        }
        if let Some(expr) = &body.tail {
            self.verify_expr(caller_name, &caller_row, expr);
        }
    }

    fn verify_stmt(&mut self, caller: &str, caller_row: &EffectRow, stmt: &AstStmt) {
        match stmt {
            AstStmt::Let { init, .. } => self.verify_expr(caller, caller_row, init),
            AstStmt::Expr(e) => self.verify_expr(caller, caller_row, e),
            AstStmt::While { cond, body, .. } => {
                self.verify_expr(caller, caller_row, cond);
                self.verify_call_sites(caller, body);
            }
            AstStmt::Loop { body, .. } => self.verify_call_sites(caller, body),
            AstStmt::Break { .. } | AstStmt::Continue { .. } => {}
            AstStmt::ForRange { start, end, body, .. } => {
                self.verify_expr(caller, caller_row, start);
                self.verify_expr(caller, caller_row, end);
                self.verify_call_sites(caller, body);
            }
            AstStmt::ForEach { iter, body, .. } => {
                self.verify_expr(caller, caller_row, iter);
                self.verify_call_sites(caller, body);
            }
            AstStmt::Assign { value, .. } => self.verify_expr(caller, caller_row, value),
            AstStmt::LetTuple { init, .. } => self.verify_expr(caller, caller_row, init),
            AstStmt::Return { value, .. } => {
                if let Some(v) = value {
                    self.verify_expr(caller, caller_row, v);
                }
            }
            AstStmt::Spawn { body, .. } => {
                for s in body {
                    self.verify_stmt(caller, caller_row, s);
                }
            }
            AstStmt::ParFor { start, end, body, .. } => {
                self.verify_expr(caller, caller_row, start);
                self.verify_expr(caller, caller_row, end);
                self.verify_call_sites(caller, body);
            }
            AstStmt::MaskStmt { effects, body, .. } => {
                let masked_row = EffectRow::new(effects.clone());
                let inner_row = caller_row.intersect(&masked_row);
                for s in &body.stmts {
                    self.verify_stmt(caller, &inner_row, s);
                }
                if let Some(e) = &body.tail {
                    self.verify_expr(caller, &inner_row, e);
                }
            }
            AstStmt::HandleStmt { expr, arms, .. } => {
                let handled_effects: Vec<String> = arms.iter().map(|a| a.effect_name.clone()).collect();
                let inner_row = caller_row.union(&EffectRow::new(handled_effects));
                self.verify_expr(caller, &inner_row, expr);
                for arm in arms {
                    self.verify_expr(caller, caller_row, &arm.body);
                }
            }
            AstStmt::Defer { expr, .. } => self.verify_expr(caller, caller_row, expr),
            AstStmt::Yield { expr, .. } => self.verify_expr(caller, caller_row, expr),
            AstStmt::Select { arms, default, .. } => {
                for arm in arms {
                    self.verify_expr(caller, caller_row, &arm.channel);
                    for s in &arm.body.stmts {
                        self.verify_stmt(caller, caller_row, s);
                    }
                    if let Some(e) = &arm.body.tail {
                        self.verify_expr(caller, caller_row, e);
                    }
                }
                if let Some(d) = default {
                    for s in &d.stmts {
                        self.verify_stmt(caller, caller_row, s);
                    }
                    if let Some(e) = &d.tail {
                        self.verify_expr(caller, caller_row, e);
                    }
                }
            }
        }
    }

    fn verify_expr(&mut self, caller: &str, caller_row: &EffectRow, expr: &AstExpr) {
        match expr {
            AstExpr::Call { callee, args, .. } => {
                let mut callee_row = self
                    .inferred
                    .get(&callee.name)
                    .cloned()
                    .or_else(|| self.registry.lookup(&callee.name).cloned())
                    .unwrap_or_else(EffectRow::pure);
                // Instantiate effect row variables with caller's row.
                callee_row = callee_row.instantiate(caller_row);
                if !callee_row.subset(caller_row) {
                    let missing: Vec<String> = callee_row
                        .effects
                        .iter()
                        .filter(|e| !caller_row.contains(e))
                        .cloned()
                        .collect();
                    self.errors.push(format!(
                        "error[E0302]: [effect check] function `{}` requires effect{} `{}` but caller `{}` has effects `{}`",
                        callee.name,
                        if missing.len() == 1 { "" } else { "s" },
                        missing.join(", "),
                        caller,
                        caller_row.display()
                    ));
                }
                for a in args {
                    self.verify_expr(caller, caller_row, a);
                }
            }
            AstExpr::MethodCall { base, args, .. } => {
                self.verify_expr(caller, caller_row, base);
                for a in args {
                    self.verify_expr(caller, caller_row, a);
                }
            }
            AstExpr::BinOp { lhs, rhs, .. } => {
                self.verify_expr(caller, caller_row, lhs);
                self.verify_expr(caller, caller_row, rhs);
            }
            AstExpr::UnaryOp { expr, .. } => self.verify_expr(caller, caller_row, expr),
            AstExpr::If { cond, then_block, else_block, .. } => {
                self.verify_expr(caller, caller_row, cond);
                self.verify_call_sites(caller, then_block);
                if let Some(eb) = else_block {
                    self.verify_call_sites(caller, eb);
                }
            }
            AstExpr::Block(b) => self.verify_call_sites(caller, b),
            AstExpr::Lambda { body, .. } => self.verify_expr(caller, caller_row, body),
            AstExpr::Tuple { elements, .. } => {
                for e in elements {
                    self.verify_expr(caller, caller_row, e);
                }
            }
            AstExpr::ArrayLit { elems, .. } => {
                for e in elems {
                    self.verify_expr(caller, caller_row, e);
                }
            }
            AstExpr::FieldAccess { base, .. } => self.verify_expr(caller, caller_row, base),
            AstExpr::Index { base, indices, .. } => {
                self.verify_expr(caller, caller_row, base);
                for i in indices {
                    self.verify_expr(caller, caller_row, i);
                }
            }
            AstExpr::When { scrutinee, arms, .. } => {
                self.verify_expr(caller, caller_row, scrutinee);
                for arm in arms {
                    if let Some(g) = &arm.guard {
                        self.verify_expr(caller, caller_row, g);
                    }
                    self.verify_expr(caller, caller_row, &arm.body);
                }
            }
            AstExpr::Cast { expr, .. } => self.verify_expr(caller, caller_row, expr),
            AstExpr::StructLit { fields, spread, .. } => {
                for (_, v) in fields {
                    self.verify_expr(caller, caller_row, v);
                }
                if let Some(s) = spread {
                    self.verify_expr(caller, caller_row, s);
                }
            }
            AstExpr::Try { expr, .. } => self.verify_expr(caller, caller_row, expr),
            AstExpr::Await { expr, .. } => self.verify_expr(caller, caller_row, expr),
            AstExpr::TupleIndex { base, .. } => self.verify_expr(caller, caller_row, base),
            AstExpr::Ident(_) | AstExpr::IntLit { .. } | AstExpr::FloatLit { .. } |
            AstExpr::BoolLit { .. } | AstExpr::StringLit { .. } => {}
            AstExpr::Mask { effects, body, .. } => {
                let masked_row = EffectRow::new(effects.clone());
                let inner_row = caller_row.intersect(&masked_row);
                for s in &body.stmts {
                    self.verify_stmt(caller, &inner_row, s);
                }
                if let Some(e) = &body.tail {
                    self.verify_expr(caller, &inner_row, e);
                }
            }
            AstExpr::Handle { expr, arms, .. } => {
                let handled_effects: Vec<String> = arms.iter().map(|a| a.effect_name.clone()).collect();
                let inner_row = caller_row.union(&EffectRow::new(handled_effects));
                self.verify_expr(caller, &inner_row, expr);
                for arm in arms {
                    self.verify_expr(caller, caller_row, &arm.body);
                }
            }
            AstExpr::NullCoal { expr, default, .. } => {
                self.verify_expr(caller, caller_row, expr);
                self.verify_expr(caller, caller_row, default);
            }
            AstExpr::MapLiteral { entries, .. } => {
                for (k, v) in entries {
                    self.verify_expr(caller, caller_row, k);
                    self.verify_expr(caller, caller_row, v);
                }
            }
            AstExpr::Ref { expr, .. }
            | AstExpr::RefMut { expr, .. }
            | AstExpr::Deref { expr, .. }
            | AstExpr::Move { expr, .. } => {
                self.verify_expr(caller, caller_row, expr);
            }
            AstExpr::Unsafe { body, .. } => {
                self.verify_expr(caller, caller_row, body);
            }
            AstExpr::Splat { expr, .. } => {
                self.verify_expr(caller, caller_row, expr);
            }
            AstExpr::TryCatch { body, catch_body, .. } => {
                self.verify_expr(caller, caller_row, body);
                self.verify_expr(caller, caller_row, catch_body);
            }
            AstExpr::Raise { args, .. } => {
                for a in args {
                    self.verify_expr(caller, caller_row, a);
                }
            }
            AstExpr::MacroCall { args, .. } => {
                for a in args {
                    self.verify_expr(caller, caller_row, a);
                }
            }
        }
    }
}
