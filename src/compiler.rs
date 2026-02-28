//! File-based module compiler with bring resolution.
//!
//! [`FileCompiler`] resolves `bring "path.iris"` and `bring std.name`
//! declarations by reading files from disk (and the embedded stdlib).
//! It performs BFS resolution with cycle detection.

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};

use crate::error::Error;
use crate::parser::ast::{AstModule, BringPath};
use crate::parser::lexer::Lexer;
use crate::parser::parse::Parser;

/// Compiles `.iris` files from disk, resolving all `bring` declarations.
pub struct FileCompiler {
    /// Extra search directories for bring resolution (beyond the file's directory).
    search_paths: Vec<PathBuf>,
}

impl FileCompiler {
    pub fn new() -> Self {
        Self { search_paths: Vec::new() }
    }

    pub fn with_search_paths(paths: Vec<PathBuf>) -> Self {
        Self { search_paths: paths }
    }

    /// Add an extra search path for bring resolution.
    pub fn add_search_path(&mut self, path: PathBuf) {
        self.search_paths.push(path);
    }

    /// Compile the given file path into a merged `AstModule`, resolving all brings.
    ///
    /// `extra_paths` is a slice of additional directories to search for brought files.
    pub fn compile_file_to_ast(
        &self,
        path: &Path,
        extra_paths: &[&Path],
    ) -> Result<AstModule, Error> {
        let canonical = path.canonicalize()
            .map_err(|e| Error::Io(e))?;
        let base_dir = canonical.parent()
            .unwrap_or(Path::new("."))
            .to_path_buf();

        // Build the full search path list.
        let mut search: Vec<PathBuf> = vec![base_dir.clone()];
        search.extend(extra_paths.iter().map(|p| p.to_path_buf()));
        search.extend(self.search_paths.iter().cloned());

        // Parse the main file.
        let main_src = std::fs::read_to_string(&canonical)?;
        let main_ast = self.parse_source(&main_src)?;

        // BFS merge.
        let mut merged = main_ast;
        let mut visited: HashSet<PathBuf> = HashSet::new();
        visited.insert(canonical.clone());

        let mut queue: VecDeque<(BringPath, PathBuf)> = VecDeque::new();
        for bring in &merged.brings.clone() {
            queue.push_back((bring.path.clone(), base_dir.clone()));
        }

        while let Some((bring_path, from_dir)) = queue.pop_front() {
            match &bring_path {
                BringPath::File(rel_path) => {
                    // Resolve relative to `from_dir`, then search_paths.
                    let resolved = self.resolve_file_path(rel_path, &from_dir, &search)?;
                    if !visited.contains(&resolved) {
                        visited.insert(resolved.clone());
                        // Check for circular dependency.
                        let dep_src = std::fs::read_to_string(&resolved)?;
                        let dep_ast = self.parse_source(&dep_src)?;
                        let dep_dir = resolved.parent()
                            .unwrap_or(Path::new("."))
                            .to_path_buf();
                        // Enqueue dep's own brings.
                        for dep_bring in &dep_ast.brings {
                            queue.push_back((dep_bring.path.clone(), dep_dir.clone()));
                        }
                        // Merge pub items.
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
        Ok(Parser::new(&tokens).parse_module()?)
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
        for func in dep.functions {
            if func.is_pub {
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
    fn default() -> Self { Self::new() }
}
