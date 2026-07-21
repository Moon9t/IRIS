//! File-based module compiler with bring resolution.
//!
//! [`FileCompiler`] resolves `bring "path.iris"` and `bring std.name`
//! declarations by reading files from disk (and the embedded stdlib).
//! It performs BFS resolution with cycle detection.
//!
//! Supports **incremental compilation** via [`crate::cache::BuildCache`]:
//! files whose content hash has not changed since the last build are
//! skipped during re-parsing.

use std::collections::{HashSet, VecDeque};
use std::path::{Path, PathBuf};

use crate::cache::BuildCache;
use crate::error::Error;
use crate::parser::ast::{
    AstBlock, AstExpr, AstModule, AstStmt, AstType, AstWhenPattern, BringPath,
};
use crate::parser::lexer::Lexer;
use crate::parser::parse::Parser;

/// Compiles `.iris` files from disk, resolving all `bring` declarations.
pub struct FileCompiler {
    /// Extra search directories for bring resolution (beyond the file's directory).
    search_paths: Vec<PathBuf>,
    /// Incremental build cache.
    cache: BuildCache,
}

impl FileCompiler {
    pub fn new() -> Self {
        // Try to locate a project root from CWD.
        let cache = if let Ok(cwd) = std::env::current_dir() {
            BuildCache::open(&cwd)
        } else {
            BuildCache::disabled()
        };
        Self {
            search_paths: Vec::new(),
            cache,
        }
    }

    /// Create a compiler with an explicit cache.
    pub fn with_cache(cache: BuildCache) -> Self {
        Self {
            search_paths: Vec::new(),
            cache,
        }
    }

    pub fn with_search_paths(paths: Vec<PathBuf>) -> Self {
        Self {
            search_paths: paths,
            cache: BuildCache::disabled(),
        }
    }

    /// Set extra search paths (used by LSP / builder to find packages/libraries).
    pub fn set_search_paths(&mut self, paths: Vec<PathBuf>) {
        self.search_paths = paths;
    }

    /// Compile a file on disk, BFS-resolving all brings and returning the full merged AST.
    pub fn compile_file_to_ast(
        &self,
        path: &Path,
        extra_paths: &[&Path],
    ) -> Result<AstModule, Error> {
        let src = std::fs::read_to_string(path)?;
        self.compile_file_to_ast_with_text(path, &src, extra_paths)
    }

    /// Like [`compile_file_to_ast`] but uses the provided `source` text for the
    /// main file instead of reading from disk.  Brings are still resolved from
    /// disk relative to `path`'s directory.
    pub fn compile_file_to_ast_with_text(
        &self,
        path: &Path,
        source: &str,
        extra_paths: &[&Path],
    ) -> Result<AstModule, Error> {
        let canonical = path.canonicalize().map_err(Error::Io)?;
        let base_dir = canonical.parent().unwrap_or(Path::new(".")).to_path_buf();

        let mut search: Vec<PathBuf> = vec![base_dir.clone()];
        search.extend(extra_paths.iter().map(|p| p.to_path_buf()));
        search.extend(self.search_paths.iter().cloned());

        let main_ast = self.parse_source(source)?;

        self.resolve_brings(main_ast, &canonical, &base_dir, &search)
    }

    /// Disable the incremental cache for this compiler instance.
    pub fn disable_cache(&mut self) {
        self.cache = BuildCache::disabled();
    }

    /// Add an extra search path for bring resolution.
    pub fn add_search_path(&mut self, path: PathBuf) {
        self.search_paths.push(path);
    }

    /// Flush the build cache manifest to disk.
    pub fn flush_cache(&mut self) {
        self.cache.flush();
    }

    /// BFS-resolve all `bring` declarations and merge dependencies into `main_ast`.
    fn resolve_brings(
        &self,
        main_ast: AstModule,
        canonical: &Path,
        base_dir: &Path,
        search: &[PathBuf],
    ) -> Result<AstModule, Error> {
        let mut merged = main_ast;
        let mut visited: HashSet<PathBuf> = HashSet::new();
        visited.insert(canonical.to_path_buf());

        let mut queue: VecDeque<(BringPath, PathBuf)> = VecDeque::new();
        for bring in &merged.brings.clone() {
            queue.push_back((bring.path.clone(), base_dir.to_path_buf()));
        }

        while let Some((bring_path, from_dir)) = queue.pop_front() {
            match &bring_path {
                BringPath::File(rel_path) => {
                    // Resolve relative to `from_dir`, then search_paths.
                    let resolved = self.resolve_file_path(rel_path, &from_dir, search)?;
                    if !visited.contains(&resolved) {
                        visited.insert(resolved.clone());
                        let dep_src = std::fs::read_to_string(&resolved)?;
                        let mut dep_ast = self.parse_source(&dep_src)?;
                        let dep_dir = resolved.parent().unwrap_or(Path::new(".")).to_path_buf();
                        for dep_bring in &dep_ast.brings {
                            queue.push_back((dep_bring.path.clone(), dep_dir.clone()));
                        }
                        let stem = resolved
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .unwrap_or("module")
                            .replace(['.', '-'], "_");
                        mangle_module_symbols(&mut dep_ast, &stem);
                        self.merge_dep(&mut merged, dep_ast);
                    }
                }
                BringPath::Stdlib(name) => {
                    let key = format!("__stdlib:{}", name);
                    let key_path = PathBuf::from(&key);
                    if !visited.contains(&key_path) {
                        visited.insert(key_path);
                        if let Some(src) = crate::stdlib::stdlib_source(name) {
                            let mut dep_ast = self.parse_source(src)?;
                            for dep_bring in &dep_ast.brings {
                                queue.push_back((dep_bring.path.clone(), base_dir.to_path_buf()));
                            }
                            let mod_name = name.replace(['.', '-'], "_");
                            mangle_module_symbols(&mut dep_ast, &mod_name);
                            self.merge_dep(&mut merged, dep_ast);
                        }
                    }
                }
            }
        }

        Ok(merged)
    }

    fn parse_source(&self, src: &str) -> Result<AstModule, Error> {
        let tokens = Lexer::new(src).tokenize()?;
        let mut parser = Parser::new(&tokens);
        let (module, errors) = parser.parse_module_recovering();
        if errors.is_empty() {
            return Ok(module);
        }
        for e in &errors {
            eprintln!("\x1b[1;31merror\x1b[0m: {}", e);
        }
        if errors.len() > 1 {
            eprintln!(
                "\x1b[1;31merror\x1b[0m: aborting due to {} parse error(s)",
                errors.len()
            );
        }
        Err(Error::Parse(
            errors
                .into_iter()
                .next()
                .expect("errors is non-empty, checked above"),
        ))
    }

    fn resolve_file_path(
        &self,
        rel_path: &str,
        from_dir: &Path,
        search: &[PathBuf],
    ) -> Result<PathBuf, Error> {
        let find_entry_point = |dir: &Path| -> Option<PathBuf> {
            if dir.is_file() {
                return Some(dir.to_path_buf());
            }
            if dir.is_dir() {
                let lib = dir.join("lib.iris");
                if lib.exists() {
                    return Some(lib);
                }
                let main = dir.join("main.iris");
                if main.exists() {
                    return Some(main);
                }
                let name = dir.file_name().and_then(|s| s.to_str()).unwrap_or("lib");
                let named = dir.join(format!("{}.iris", name));
                if named.exists() {
                    return Some(named);
                }
            }
            None
        };

        if let Some(res) = find_entry_point(&from_dir.join(rel_path)) {
            return res.canonicalize().map_err(Error::Io);
        }

        let mut cur = Some(from_dir.to_path_buf());
        while let Some(dir) = cur {
            let dep_dir = dir.join(".iris").join("deps").join(rel_path);
            if let Some(res) = find_entry_point(&dep_dir) {
                return res.canonicalize().map_err(Error::Io);
            }
            cur = dir.parent().map(|p| p.to_path_buf());
        }

        for dir in search {
            if let Some(res) = find_entry_point(&dir.join(rel_path)) {
                return res.canonicalize().map_err(Error::Io);
            }
        }

        Err(Error::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("cannot find brought module: {}", rel_path),
        )))
    }

    fn merge_dep(&self, main_ast: &mut AstModule, dep: AstModule) {
        main_ast.extern_fns.extend(dep.extern_fns);
        main_ast.functions.extend(dep.functions);
        main_ast.structs.extend(dep.structs);
        main_ast.enums.extend(dep.enums);
        main_ast.consts.extend(dep.consts);
        main_ast.type_aliases.extend(dep.type_aliases);
        main_ast.traits.extend(dep.traits);
        main_ast.impls.extend(dep.impls);
        main_ast.models.extend(dep.models);
    }
}

// ---------------------------------------------------------------------------
// Module Namespace Mangling AST Rewriter
// ---------------------------------------------------------------------------

fn collect_local_symbols(ast: &AstModule) -> HashSet<String> {
    let mut symbols = HashSet::new();
    for f in &ast.functions {
        symbols.insert(f.name.name.clone());
    }
    for s in &ast.structs {
        symbols.insert(s.name.name.clone());
    }
    for e in &ast.enums {
        symbols.insert(e.name.name.clone());
    }
    for c in &ast.consts {
        symbols.insert(c.name.name.clone());
    }
    for ta in &ast.type_aliases {
        symbols.insert(ta.name.clone());
    }
    for t in &ast.traits {
        symbols.insert(t.name.name.clone());
    }
    for m in &ast.models {
        symbols.insert(m.name.name.clone());
    }
    symbols
}

fn rewrite_type(ty: &mut AstType, symbols: &HashSet<String>, prefix: &str) {
    match ty {
        AstType::Scalar(_, _) => {}
        AstType::Tensor { .. } => {}
        AstType::Named(ref mut name, _) => {
            if symbols.contains(name) {
                *name = format!("{}__{}", prefix, name);
            }
        }
        AstType::Tuple(ref mut tys, _) => {
            for t in tys {
                rewrite_type(t, symbols, prefix);
            }
        }
        AstType::Array { ref mut elem, .. } => {
            rewrite_type(elem, symbols, prefix);
        }
        AstType::Option(ref mut elem, _) => {
            rewrite_type(elem, symbols, prefix);
        }
        AstType::Result(ref mut ok, ref mut err, _) => {
            rewrite_type(ok, symbols, prefix);
            rewrite_type(err, symbols, prefix);
        }
        AstType::Chan(ref mut elem, _) => {
            rewrite_type(elem, symbols, prefix);
        }
        AstType::Atomic(ref mut elem, _) => {
            rewrite_type(elem, symbols, prefix);
        }
        AstType::Mutex(ref mut elem, _) => {
            rewrite_type(elem, symbols, prefix);
        }
        AstType::Grad(ref mut elem, _) => {
            rewrite_type(elem, symbols, prefix);
        }
        AstType::Sparse(ref mut elem, _) => {
            rewrite_type(elem, symbols, prefix);
        }
        AstType::List(ref mut elem, _) => {
            rewrite_type(elem, symbols, prefix);
        }
        AstType::Map(ref mut k, ref mut v, _) => {
            rewrite_type(k, symbols, prefix);
            rewrite_type(v, symbols, prefix);
        }
        AstType::Generic { ref mut args, .. } => {
            for arg in args {
                rewrite_type(arg, symbols, prefix);
            }
        }
        AstType::Fn {
            ref mut params,
            ref mut ret,
            ..
        } => {
            for p in params {
                rewrite_type(p, symbols, prefix);
            }
            rewrite_type(ret, symbols, prefix);
        }
        AstType::ConstInt(_, _) => {}
        AstType::AssocType { .. } => {}
        AstType::WeakRef(ref mut elem, _) => {
            rewrite_type(elem, symbols, prefix);
        }
        AstType::DynTrait { .. } => {}
        AstType::MaskEffectType { .. } => {}
    }
}

fn rewrite_expr(expr: &mut AstExpr, symbols: &HashSet<String>, prefix: &str) {
    match expr {
        AstExpr::Ident(ref mut ident) => {
            if symbols.contains(&ident.name) {
                ident.name = format!("{}__{}", prefix, ident.name);
            }
        }
        AstExpr::IntLit { .. }
        | AstExpr::FloatLit { .. }
        | AstExpr::BoolLit { .. }
        | AstExpr::StringLit { .. } => {}
        AstExpr::BinOp {
            ref mut lhs,
            ref mut rhs,
            ..
        } => {
            rewrite_expr(lhs, symbols, prefix);
            rewrite_expr(rhs, symbols, prefix);
        }
        AstExpr::Call {
            ref mut callee,
            ref mut args,
            ..
        } => {
            if symbols.contains(&callee.name) {
                callee.name = format!("{}__{}", prefix, callee.name);
            }
            for arg in args {
                rewrite_expr(arg, symbols, prefix);
            }
        }
        AstExpr::UnaryOp { ref mut expr, .. } => {
            rewrite_expr(expr, symbols, prefix);
        }
        AstExpr::If {
            ref mut cond,
            ref mut then_block,
            ref mut else_block,
            ..
        } => {
            rewrite_expr(cond, symbols, prefix);
            rewrite_block(then_block, symbols, prefix);
            if let Some(ref mut else_b) = else_block {
                rewrite_block(else_b, symbols, prefix);
            }
        }
        AstExpr::Block(ref mut block) => {
            rewrite_block(block, symbols, prefix);
        }
        AstExpr::Index {
            ref mut base,
            ref mut indices,
            ..
        } => {
            rewrite_expr(base, symbols, prefix);
            for idx in indices {
                rewrite_expr(idx, symbols, prefix);
            }
        }
        AstExpr::Cast {
            ref mut expr,
            ref mut ty,
            ..
        } => {
            rewrite_expr(expr, symbols, prefix);
            rewrite_type(ty, symbols, prefix);
        }
        AstExpr::StructLit {
            ref mut name,
            ref mut fields,
            ..
        } => {
            if symbols.contains(name) {
                *name = format!("{}__{}", prefix, name);
            }
            for (_, ref mut val) in fields {
                rewrite_expr(val, symbols, prefix);
            }
        }
        AstExpr::FieldAccess { ref mut base, .. } => {
            rewrite_expr(base, symbols, prefix);
        }
        AstExpr::When {
            ref mut scrutinee,
            ref mut arms,
            ..
        } => {
            rewrite_expr(scrutinee, symbols, prefix);
            for arm in arms {
                rewrite_when_pattern(&mut arm.pattern, symbols, prefix);
                if let Some(ref mut guard) = arm.guard {
                    rewrite_expr(guard, symbols, prefix);
                }
                rewrite_expr(&mut arm.body, symbols, prefix);
            }
        }
        AstExpr::Tuple {
            ref mut elements, ..
        } => {
            for elem in elements {
                rewrite_expr(elem, symbols, prefix);
            }
        }
        AstExpr::TupleIndex { ref mut base, .. } => {
            rewrite_expr(base, symbols, prefix);
        }
        AstExpr::ArrayLit { ref mut elems, .. } => {
            for elem in elems {
                rewrite_expr(elem, symbols, prefix);
            }
        }
        AstExpr::Lambda {
            ref mut params,
            ref mut body,
            ..
        } => {
            for param in params {
                rewrite_type(&mut param.ty, symbols, prefix);
                if let Some(ref mut def_val) = param.default {
                    rewrite_expr(def_val, symbols, prefix);
                }
            }
            rewrite_expr(body, symbols, prefix);
        }
        AstExpr::Await { ref mut expr, .. } => {
            rewrite_expr(expr, symbols, prefix);
        }
        AstExpr::Try { ref mut expr, .. } => {
            rewrite_expr(expr, symbols, prefix);
        }
        AstExpr::MethodCall {
            ref mut base,
            ref mut args,
            ..
        } => {
            rewrite_expr(base, symbols, prefix);
            for arg in args {
                rewrite_expr(arg, symbols, prefix);
            }
        }
        AstExpr::Mask { ref mut body, .. } => {
            rewrite_block(body, symbols, prefix);
        }
        AstExpr::Handle {
            ref mut expr,
            ref mut arms,
            ..
        } => {
            rewrite_expr(expr, symbols, prefix);
            for arm in arms {
                rewrite_expr(&mut arm.body, symbols, prefix);
            }
        }
    }
}

fn rewrite_when_pattern(pat: &mut AstWhenPattern, symbols: &HashSet<String>, prefix: &str) {
    match pat {
        AstWhenPattern::EnumVariant {
            ref mut enum_name, ..
        } => {
            if symbols.contains(enum_name) {
                *enum_name = format!("{}__{}", prefix, enum_name);
            }
        }
        AstWhenPattern::OptionSome { .. } => {}
        AstWhenPattern::OptionNone => {}
        AstWhenPattern::ResultOk { .. } => {}
        AstWhenPattern::ResultErr { .. } => {}
        AstWhenPattern::Wildcard => {}
        AstWhenPattern::IntLit(_) => {}
        AstWhenPattern::BoolLit(_) => {}
        AstWhenPattern::StringLit(_) => {}
        AstWhenPattern::Tuple(ref mut pats) => {
            for p in pats {
                rewrite_when_pattern(p, symbols, prefix);
            }
        }
        AstWhenPattern::Range { .. } => {}
        AstWhenPattern::Or(ref mut pats) => {
            for p in pats {
                rewrite_when_pattern(p, symbols, prefix);
            }
        }
        AstWhenPattern::Slice { prefix: ref mut prefix_pats, .. } => {
            for p in prefix_pats {
                rewrite_when_pattern(p, symbols, prefix);
            }
        }
    }
}

fn rewrite_block(block: &mut AstBlock, symbols: &HashSet<String>, prefix: &str) {
    for stmt in &mut block.stmts {
        rewrite_stmt(stmt, symbols, prefix);
    }
    if let Some(ref mut tail) = block.tail {
        rewrite_expr(tail, symbols, prefix);
    }
}

fn rewrite_stmt(stmt: &mut AstStmt, symbols: &HashSet<String>, prefix: &str) {
    match stmt {
        AstStmt::Let {
            ref mut ty,
            ref mut init,
            ..
        } => {
            if let Some(ref mut t) = ty {
                rewrite_type(t, symbols, prefix);
            }
            rewrite_expr(init, symbols, prefix);
        }
        AstStmt::Expr(ref mut expr) => {
            rewrite_expr(expr, symbols, prefix);
        }
        AstStmt::While {
            ref mut cond,
            ref mut body,
            ..
        } => {
            rewrite_expr(cond, symbols, prefix);
            rewrite_block(body, symbols, prefix);
        }
        AstStmt::Loop { ref mut body, .. } => {
            rewrite_block(body, symbols, prefix);
        }
        AstStmt::Break { .. } | AstStmt::Continue { .. } => {}
        AstStmt::ForRange {
            ref mut start,
            ref mut end,
            ref mut body,
            ..
        } => {
            rewrite_expr(start, symbols, prefix);
            rewrite_expr(end, symbols, prefix);
            rewrite_block(body, symbols, prefix);
        }
        AstStmt::Assign {
            ref mut target,
            ref mut value,
            ..
        } => {
            rewrite_expr(target, symbols, prefix);
            rewrite_expr(value, symbols, prefix);
        }
        AstStmt::LetTuple { ref mut init, .. } => {
            rewrite_expr(init, symbols, prefix);
        }
        AstStmt::Return { ref mut value, .. } => {
            if let Some(ref mut val) = value {
                rewrite_expr(val, symbols, prefix);
            }
        }
        AstStmt::Spawn { ref mut body, .. } => {
            for s in body {
                rewrite_stmt(s, symbols, prefix);
            }
        }
        AstStmt::ParFor {
            ref mut start,
            ref mut end,
            ref mut body,
            ..
        } => {
            rewrite_expr(start, symbols, prefix);
            rewrite_expr(end, symbols, prefix);
            rewrite_block(body, symbols, prefix);
        }
        AstStmt::ForEach {
            ref mut iter,
            ref mut body,
            ..
        } => {
            rewrite_expr(iter, symbols, prefix);
            rewrite_block(body, symbols, prefix);
        }
        AstStmt::MaskStmt { ref mut body, .. } => {
            rewrite_block(body, symbols, prefix);
        }
        AstStmt::HandleStmt {
            ref mut expr,
            ref mut arms,
            ..
        } => {
            rewrite_expr(expr, symbols, prefix);
            for arm in arms {
                rewrite_expr(&mut arm.body, symbols, prefix);
            }
        }
    }
}

pub(crate) fn mangle_module_symbols(ast: &mut AstModule, mod_name: &str) {
    let symbols = collect_local_symbols(ast);
    if symbols.is_empty() {
        return;
    }

    // 1. Mangle the definitions themselves
    for f in &mut ast.functions {
        if symbols.contains(&f.name.name) {
            f.name.name = format!("{}__{}", mod_name, f.name.name);
        }
        rewrite_block(&mut f.body, &symbols, mod_name);
        for p in &mut f.params {
            rewrite_type(&mut p.ty, &symbols, mod_name);
            if let Some(ref mut def_val) = p.default {
                rewrite_expr(def_val, &symbols, mod_name);
            }
        }
        rewrite_type(&mut f.return_ty, &symbols, mod_name);
    }

    for s in &mut ast.structs {
        if symbols.contains(&s.name.name) {
            s.name.name = format!("{}__{}", mod_name, s.name.name);
        }
        for f in &mut s.fields {
            rewrite_type(&mut f.ty, &symbols, mod_name);
        }
    }

    for e in &mut ast.enums {
        if symbols.contains(&e.name.name) {
            e.name.name = format!("{}__{}", mod_name, e.name.name);
        }
        for v in &mut e.variants {
            for f in &mut v.fields {
                rewrite_type(f, &symbols, mod_name);
            }
        }
    }

    for c in &mut ast.consts {
        if symbols.contains(&c.name.name) {
            c.name.name = format!("{}__{}", mod_name, c.name.name);
        }
        if let Some(ref mut t) = c.ty {
            rewrite_type(t, &symbols, mod_name);
        }
        rewrite_expr(&mut c.value, &symbols, mod_name);
    }

    for ta in &mut ast.type_aliases {
        if symbols.contains(&ta.name) {
            ta.name = format!("{}__{}", mod_name, ta.name);
        }
        rewrite_type(&mut ta.ty, &symbols, mod_name);
    }

    for t in &mut ast.traits {
        if symbols.contains(&t.name.name) {
            t.name.name = format!("{}__{}", mod_name, t.name.name);
        }
        for m in &mut t.methods {
            for p in &mut m.params {
                rewrite_type(&mut p.ty, &symbols, mod_name);
            }
            rewrite_type(&mut m.return_ty, &symbols, mod_name);
        }
    }

    for m in &mut ast.models {
        if symbols.contains(&m.name.name) {
            m.name.name = format!("{}__{}", mod_name, m.name.name);
        }
    }

    for i in &mut ast.impls {
        if symbols.contains(&i.trait_name) {
            i.trait_name = format!("{}__{}", mod_name, i.trait_name);
        }
        if symbols.contains(&i.type_name) {
            i.type_name = format!("{}__{}", mod_name, i.type_name);
        }
        for f in &mut i.methods {
            if symbols.contains(&f.name.name) {
                f.name.name = format!("{}__{}", mod_name, f.name.name);
            }
            rewrite_block(&mut f.body, &symbols, mod_name);
            for p in &mut f.params {
                rewrite_type(&mut p.ty, &symbols, mod_name);
                if let Some(ref mut def_val) = p.default {
                    rewrite_expr(def_val, &symbols, mod_name);
                }
            }
            rewrite_type(&mut f.return_ty, &symbols, mod_name);
        }
    }
}

impl Default for FileCompiler {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for FileCompiler {
    fn drop(&mut self) {
        self.cache.flush();
    }
}
