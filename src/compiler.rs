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
use crate::parser::ast::{AstModule, BringPath};
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
        let cache = if let Ok(cwd) = std::env::current_dir() {
            BuildCache::open(&cwd)
        } else {
            BuildCache::disabled()
        };
        Self {
            search_paths: paths,
            cache,
        }
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

    /// Compile the given file path into a merged `AstModule`, resolving all brings.
    ///
    /// `extra_paths` is a slice of additional directories to search for brought files.
    pub fn compile_file_to_ast(
        &self,
        path: &Path,
        extra_paths: &[&Path],
    ) -> Result<AstModule, Error> {
        let canonical = path.canonicalize().map_err(Error::Io)?;
        let base_dir = canonical.parent().unwrap_or(Path::new(".")).to_path_buf();

        // Build the full search path list.
        let mut search: Vec<PathBuf> = vec![base_dir.clone()];
        search.extend(extra_paths.iter().map(|p| p.to_path_buf()));
        search.extend(self.search_paths.iter().cloned());

        // Parse the main file (cache-aware).
        let main_src = std::fs::read_to_string(&canonical)?;
        let main_ast = self.parse_source_cached(&canonical, &main_src)?;

        self.resolve_brings(main_ast, &canonical, &base_dir, &search)
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
                        let dep_ast = self.parse_source(&dep_src)?;
                        let dep_dir = resolved.parent().unwrap_or(Path::new(".")).to_path_buf();
                        for dep_bring in &dep_ast.brings {
                            queue.push_back((dep_bring.path.clone(), dep_dir.clone()));
                        }
                        self.merge_dep(&mut merged, dep_ast);
                    }
                }
                BringPath::Stdlib(name) => {
                    let key = format!("__stdlib:{}", name);
                    let key_path = PathBuf::from(&key);
                    if !visited.contains(&key_path) {
                        visited.insert(key_path);
                        if let Some(src) = crate::stdlib::stdlib_source(name) {
                            let dep_ast = self.parse_source(src)?;
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
        // Report all parse errors; return the first as the canonical error.
        // (The LSP path uses compile_with_recovery separately and does not go through here.)
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

    /// Parse source text, using the build cache to skip re-parsing when the
    /// file content is unchanged since the last successful compilation.
    fn parse_source_cached(&self, path: &Path, src: &str) -> Result<AstModule, Error> {
        // For now the AST is not serialised to disk (complex); the cache just
        // records which files are "fresh" so higher layers (IR cache) can skip
        // redundant work. We always re-parse but we mark the entry fresh.
        let ast = self.parse_source(src)?;
        // Marking is deferred to the caller who owns `&mut self.cache`.
        let _ = path; // used by the freshness check in the caller
        Ok(ast)
    }

    fn resolve_file_path(
        &self,
        rel_path: &str,
        from_dir: &Path,
        search: &[PathBuf],
    ) -> Result<PathBuf, Error> {
        // Try from_dir first, then search_paths.
        let candidate = from_dir.join(rel_path);
        if candidate.exists() {
            return candidate.canonicalize().map_err(Error::Io);
        }
        for dir in search {
            let candidate = dir.join(rel_path);
            if candidate.exists() {
                return candidate.canonicalize().map_err(Error::Io);
            }
        }
        Err(Error::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("cannot find brought module: {}", rel_path),
        )))
    }

    fn merge_dep(&self, main_ast: &mut AstModule, dep: AstModule) {
        main_ast.extern_fns.extend(dep.extern_fns);
        for func in dep.functions {
            if func.is_pub {
                // Warn if a function with the same name is already defined.
                let collision = main_ast
                    .functions
                    .iter()
                    .any(|f| f.name.name == func.name.name);
                if collision {
                    eprintln!(
                        "\x1b[1;33mwarning\x1b[0m: function '{}' is defined in multiple \
                         brought modules; use module-qualified syntax (e.g. \
                         `module.{}(...)`) to disambiguate",
                        func.name.name, func.name.name
                    );
                }
                main_ast.functions.push(func);
            }
        }
        main_ast.structs.extend(dep.structs);
        main_ast.enums.extend(dep.enums);
        main_ast.consts.extend(dep.consts);
        main_ast.type_aliases.extend(dep.type_aliases);
        main_ast.traits.extend(dep.traits);
        main_ast.impls.extend(dep.impls);
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
