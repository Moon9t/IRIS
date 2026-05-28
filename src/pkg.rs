//! IRIS Package Manager (`iris pkg`).
//!
//! Supports **local path** and **git** dependencies only.
//! No central registry — packages are referenced by path or git URL.
//!
//! ## Manifest format (`iris.toml`)
//!
//! ```toml
//! [package]
//! name    = "my-project"
//! version = "0.1.0"
//! entry   = "main.iris"
//!
//! [dependencies]
//! utils = { path = "../shared/utils" }
//! web   = { git = "https://github.com/user/iris-web.git" }
//! auth  = { git = "https://github.com/user/iris-auth.git", tag  = "v1.2.0" }
//! core  = { git = "https://github.com/user/iris-core.git", rev  = "a1b2c3d" }
//! dev   = { git = "https://github.com/user/iris-dev.git",  branch = "main" }
//! ```
//!
//! ## Lock file (`iris.lock`)
//!
//! Auto-generated next to `iris.toml`. Commit it to source control for
//! reproducible builds. Records the exact git commit for each git dependency.
//!
//! ## Commands
//!
//! - `iris pkg init`                  — create a new `iris.toml`
//! - `iris pkg add <n> --path <p>`    — add a local path dependency
//! - `iris pkg add <n> --git <url>`   — add a git dependency (latest)
//! - `iris pkg add <n> --git <url> --tag <t>`    — pin to a git tag
//! - `iris pkg add <n> --git <url> --rev <sha>`  — pin to a commit SHA
//! - `iris pkg add <n> --git <url> --branch <b>` — track a branch
//! - `iris pkg remove <name>`         — remove a dependency
//! - `iris pkg install`               — fetch / sync all deps into `.iris/deps/`
//! - `iris pkg update [name]`         — update git deps to latest matching ref
//! - `iris pkg list`                  — list current dependencies
//! - `iris pkg build`                 — install deps + build entry binary
//! - `iris pkg run`                   — build + run
//! - `iris pkg check`                 — verify all deps are installed

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;

// ── Manifest types ────────────────────────────────────────────────────────────

/// Parsed `iris.toml` manifest.
#[derive(Debug, Clone)]
pub struct Manifest {
    pub name: String,
    pub version: String,
    pub entry: String,
    pub description: String,
    pub license: String,
    pub repository: String,
    pub deps: BTreeMap<String, Dep>,
}

/// A single dependency — path or git only.
#[derive(Debug, Clone)]
pub enum Dep {
    /// `name = { path = "..." }`
    Path(String),
    /// `name = { git = "...", [branch = "..."], [tag = "..."], [rev = "..."] }`
    Git {
        url: String,
        /// Track a specific branch (default: repo default branch).
        branch: Option<String>,
        /// Pin to a git tag (e.g. "v1.2.0").
        tag: Option<String>,
        /// Pin to an exact commit SHA.
        rev: Option<String>,
    },
}

impl fmt::Display for Dep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Dep::Path(p) => write!(f, "{{ path = \"{}\" }}", p),
            Dep::Git {
                url,
                branch,
                tag,
                rev,
            } => {
                write!(f, "{{ git = \"{}\"", url)?;
                if let Some(b) = branch {
                    write!(f, ", branch = \"{}\"", b)?;
                }
                if let Some(t) = tag {
                    write!(f, ", tag = \"{}\"", t)?;
                }
                if let Some(r) = rev {
                    write!(f, ", rev = \"{}\"", r)?;
                }
                write!(f, " }}")
            }
        }
    }
}

// ── Serde mapping structs for TOML ────────────────────────────────────────────

#[derive(Serialize, Deserialize, Debug, Clone)]
struct TomlManifest {
    package: TomlPackage,
    #[serde(default)]
    dependencies: BTreeMap<String, TomlDep>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct TomlPackage {
    name: String,
    version: String,
    entry: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    license: Option<String>,
    #[serde(default)]
    repository: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
enum TomlDep {
    Simple(String),
    Detailed(DetailedDep),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct DetailedDep {
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    git: Option<String>,
    #[serde(default)]
    branch: Option<String>,
    #[serde(default)]
    tag: Option<String>,
    #[serde(default)]
    rev: Option<String>,
}

impl Manifest {
    /// Parse a TOML manifest using the standard `toml` crate.
    pub fn parse(src: &str) -> Result<Self, String> {
        let toml_manifest: TomlManifest =
            toml::from_str(src).map_err(|e| format!("failed to parse iris.toml: {}", e))?;

        let mut deps = BTreeMap::new();
        for (name, toml_dep) in toml_manifest.dependencies {
            let dep = match toml_dep {
                TomlDep::Simple(s) => {
                    if s.starts_with("http://")
                        || s.starts_with("https://")
                        || s.starts_with("git@")
                    {
                        Dep::Git {
                            url: s,
                            branch: None,
                            tag: None,
                            rev: None,
                        }
                    } else {
                        Dep::Path(s)
                    }
                }
                TomlDep::Detailed(d) => {
                    if let Some(p) = d.path {
                        Dep::Path(p)
                    } else if let Some(git) = d.git {
                        Dep::Git {
                            url: git,
                            branch: d.branch,
                            tag: d.tag,
                            rev: d.rev,
                        }
                    } else {
                        return Err(format!(
                            "dependency '{}' must specify either 'path' or 'git'",
                            name
                        ));
                    }
                }
            };
            deps.insert(name, dep);
        }

        Ok(Manifest {
            name: toml_manifest.package.name,
            version: toml_manifest.package.version,
            entry: toml_manifest.package.entry,
            description: toml_manifest.package.description.unwrap_or_default(),
            license: toml_manifest.package.license.unwrap_or_default(),
            repository: toml_manifest.package.repository.unwrap_or_default(),
            deps,
        })
    }

    /// Serialize back to TOML text.
    pub fn to_toml(&self) -> String {
        let mut toml_deps = BTreeMap::new();
        for (name, dep) in &self.deps {
            let toml_dep = match dep {
                Dep::Path(p) => TomlDep::Detailed(DetailedDep {
                    path: Some(p.clone()),
                    git: None,
                    branch: None,
                    tag: None,
                    rev: None,
                }),
                Dep::Git {
                    url,
                    branch,
                    tag,
                    rev,
                } => TomlDep::Detailed(DetailedDep {
                    path: None,
                    git: Some(url.clone()),
                    branch: branch.clone(),
                    tag: tag.clone(),
                    rev: rev.clone(),
                }),
            };
            toml_deps.insert(name.clone(), toml_dep);
        }

        let toml_manifest = TomlManifest {
            package: TomlPackage {
                name: self.name.clone(),
                version: self.version.clone(),
                entry: self.entry.clone(),
                description: if self.description.is_empty() {
                    None
                } else {
                    Some(self.description.clone())
                },
                license: if self.license.is_empty() {
                    None
                } else {
                    Some(self.license.clone())
                },
                repository: if self.repository.is_empty() {
                    None
                } else {
                    Some(self.repository.clone())
                },
            },
            dependencies: toml_deps,
        };

        toml::to_string_pretty(&toml_manifest).unwrap_or_else(|_| {
            let mut out = String::new();
            out.push_str("[package]\n");
            out.push_str(&format!("name    = \"{}\"\n", self.name));
            out.push_str(&format!("version = \"{}\"\n", self.version));
            out.push_str(&format!("entry   = \"{}\"\n", self.entry));
            out
        })
    }
}

// ── Lock file ─────────────────────────────────────────────────────────────────

/// One entry in `iris.lock`.
#[derive(Debug, Clone)]
pub struct LockEntry {
    /// "path" or "git"
    pub kind: String,
    /// Canonical source: absolute path or git URL.
    pub source: String,
    /// For git: the resolved commit SHA (40 hex chars).
    pub commit: Option<String>,
}

/// The `iris.lock` file — maps dep name → locked entry.
#[derive(Debug, Default)]
pub struct LockFile {
    pub entries: BTreeMap<String, LockEntry>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
struct TomlLockFile {
    #[serde(default)]
    dep: BTreeMap<String, TomlLockEntry>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct TomlLockEntry {
    kind: String,
    source: String,
    #[serde(default)]
    commit: Option<String>,
}

impl LockFile {
    /// Parse from the TOML content of `iris.lock`.
    pub fn parse(src: &str) -> Self {
        let lock: TomlLockFile = toml::from_str(src).unwrap_or_default();
        let mut entries = BTreeMap::new();
        for (name, entry) in lock.dep {
            entries.insert(
                name,
                LockEntry {
                    kind: entry.kind,
                    source: entry.source,
                    commit: entry.commit,
                },
            );
        }
        LockFile { entries }
    }

    /// Serialize to TOML text.
    pub fn to_text(&self) -> String {
        let mut toml_dep = BTreeMap::new();
        for (name, entry) in &self.entries {
            toml_dep.insert(
                name.clone(),
                TomlLockEntry {
                    kind: entry.kind.clone(),
                    source: entry.source.clone(),
                    commit: entry.commit.clone(),
                },
            );
        }
        let lock = TomlLockFile { dep: toml_dep };
        let mut out = String::from(
            "# iris.lock — generated by `iris pkg install`. Commit to version control.\n\
             # Do not edit manually.\n\n",
        );
        if let Ok(toml_str) = toml::to_string_pretty(&lock) {
            out.push_str(&toml_str);
        }
        out
    }
}

// ── Manifest / lock file I/O ──────────────────────────────────────────────────

fn find_manifest(start_dir: &Path) -> Option<PathBuf> {
    let mut dir = start_dir.to_path_buf();
    loop {
        let candidate = dir.join("iris.toml");
        if candidate.exists() {
            return Some(candidate);
        }
        if !dir.pop() {
            return None;
        }
    }
}

fn load_manifest() -> Result<(PathBuf, Manifest), String> {
    let cwd = std::env::current_dir().map_err(|e| format!("cannot read cwd: {}", e))?;
    let path = find_manifest(&cwd)
        .ok_or_else(|| "no iris.toml found (run `iris pkg init` to create one)".to_string())?;
    let text =
        fs::read_to_string(&path).map_err(|e| format!("cannot read {}: {}", path.display(), e))?;
    let manifest = Manifest::parse(&text)?;
    Ok((path, manifest))
}

fn save_manifest(path: &Path, manifest: &Manifest) -> Result<(), String> {
    fs::write(path, manifest.to_toml())
        .map_err(|e| format!("cannot write {}: {}", path.display(), e))
}

fn lock_path(manifest_path: &Path) -> PathBuf {
    manifest_path.with_file_name("iris.lock")
}

fn load_lock(manifest_path: &Path) -> LockFile {
    let lp = lock_path(manifest_path);
    fs::read_to_string(&lp)
        .map(|s| LockFile::parse(&s))
        .unwrap_or_default()
}

fn save_lock(manifest_path: &Path, lock: &LockFile) -> Result<(), String> {
    let lp = lock_path(manifest_path);
    fs::write(&lp, lock.to_text()).map_err(|e| format!("cannot write iris.lock: {}", e))
}

// ── Commands ──────────────────────────────────────────────────────────────────

/// `iris pkg init` — create a new project.
pub fn cmd_init() -> Result<(), String> {
    let cwd = std::env::current_dir().map_err(|e| format!("cannot read cwd: {}", e))?;
    let manifest_path = cwd.join("iris.toml");

    if manifest_path.exists() {
        return Err("iris.toml already exists in this directory".into());
    }

    let dir_name = cwd
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("my-project")
        .to_string();

    let manifest = Manifest {
        name: dir_name.clone(),
        version: "0.1.0".into(),
        entry: "main.iris".into(),
        description: String::new(),
        license: String::new(),
        repository: String::new(),
        deps: BTreeMap::new(),
    };

    save_manifest(&manifest_path, &manifest)?;

    let main_path = cwd.join("main.iris");
    if !main_path.exists() {
        fs::write(
            &main_path,
            format!(
                "// {} — entry point\n\ndef main() -> i64 {{\n    print(\"Hello from {}!\");\n    0\n}}\n",
                dir_name, dir_name
            ),
        ).map_err(|e| format!("cannot write main.iris: {}", e))?;
    }

    fs::create_dir_all(cwd.join(".iris")).map_err(|e| format!("cannot create .iris/: {}", e))?;

    eprintln!(
        "initialized IRIS project '{}' in {}",
        dir_name,
        cwd.display()
    );
    Ok(())
}

/// `iris pkg add <name> --path <p>` or `--git <url> [--tag t | --rev r | --branch b]`.
pub fn cmd_add(name: &str, dep: Dep) -> Result<(), String> {
    let (path, mut manifest) = load_manifest()?;
    manifest.deps.insert(name.to_string(), dep.clone());
    save_manifest(&path, &manifest)?;
    eprintln!("added dependency '{}' = {}", name, dep);
    Ok(())
}

/// `iris pkg remove <name>`.
pub fn cmd_remove(name: &str) -> Result<(), String> {
    let (path, mut manifest) = load_manifest()?;
    if manifest.deps.remove(name).is_none() {
        return Err(format!("dependency '{}' not found in iris.toml", name));
    }
    // Also remove from lock file.
    let mut lock = load_lock(&path);
    lock.entries.remove(name);
    save_manifest(&path, &manifest)?;
    save_lock(&path, &lock)?;
    eprintln!("removed dependency '{}'", name);
    Ok(())
}

/// `iris pkg list` — print all dependencies.
pub fn cmd_list() -> Result<(), String> {
    let (manifest_path, manifest) = load_manifest()?;
    let lock = load_lock(&manifest_path);
    eprintln!("{} v{}", manifest.name, manifest.version);
    if manifest.deps.is_empty() {
        eprintln!("  (no dependencies)");
    } else {
        for (name, dep) in &manifest.deps {
            let locked = lock.entries.get(name);
            let pin = match locked {
                Some(e) if e.commit.is_some() => format!(
                    " [{}]",
                    &e.commit.as_deref().unwrap_or("")
                        [..8.min(e.commit.as_deref().unwrap_or("").len())]
                ),
                _ => String::new(),
            };
            eprintln!("  {} = {}{}", name, dep, pin);
        }
    }
    Ok(())
}

/// `iris pkg check` — verify all deps are installed.
pub fn cmd_check() -> Result<(), String> {
    let (manifest_path, manifest) = load_manifest()?;
    let project_dir = manifest_path
        .parent()
        .ok_or("cannot determine project directory")?;
    let deps_dir = project_dir.join(".iris").join("deps");
    let mut missing = Vec::new();
    for name in manifest.deps.keys() {
        let target = deps_dir.join(name);
        if !target.exists() {
            missing.push(name.clone());
        }
    }
    if missing.is_empty() {
        eprintln!("all {} dependencies installed", manifest.deps.len());
        Ok(())
    } else {
        for m in &missing {
            eprintln!("  missing: {}", m);
        }
        Err(format!(
            "{} dependency/ies missing — run `iris pkg install`",
            missing.len()
        ))
    }
}

/// `iris pkg install` — fetch/sync all deps (including transitive) into `.iris/deps/`.
pub fn cmd_install() -> Result<(), String> {
    let (manifest_path, manifest) = load_manifest()?;
    let project_dir = manifest_path
        .parent()
        .ok_or("cannot determine project directory")?;
    let deps_dir = project_dir.join(".iris").join("deps");

    fs::create_dir_all(&deps_dir).map_err(|e| format!("cannot create .iris/deps/: {}", e))?;

    if manifest.deps.is_empty() {
        eprintln!("no dependencies to install");
        return Ok(());
    }

    let mut lock = load_lock(&manifest_path);

    // BFS queue of (dep_name, dep, project_dir) triples.
    // We track installed names globally to avoid cycles.
    let mut queue: Vec<(String, Dep, PathBuf)> = manifest
        .deps
        .iter()
        .map(|(n, d)| (n.clone(), d.clone(), project_dir.to_path_buf()))
        .collect();

    let mut installed: std::collections::HashSet<String> = std::collections::HashSet::new();

    while let Some((name, dep, from_dir)) = queue.pop() {
        if installed.contains(&name) {
            continue;
        }
        installed.insert(name.clone());

        let target = deps_dir.join(&name);

        // `transitive_from_dir` is where relative paths in this dep's iris.toml
        // should be resolved from — the *original* source, not the installed copy.
        let (lock_entry, transitive_from_dir) = match &dep {
            Dep::Path(rel) => {
                let source = from_dir.join(rel);
                let source = source.canonicalize().unwrap_or_else(|_| source.clone());
                if !source.exists() {
                    return Err(format!(
                        "dependency '{}': path '{}' does not exist",
                        name,
                        source.display()
                    ));
                }
                install_path_dep(&source, &target, &name)?;
                let entry = LockEntry {
                    kind: "path".into(),
                    source: source.to_string_lossy().into_owned(),
                    commit: None,
                };
                // Transitive deps in libA/iris.toml are relative to libA's source dir.
                (entry, source.clone())
            }
            Dep::Git {
                url,
                branch,
                tag,
                rev,
            } => {
                let commit = install_git_dep(
                    url,
                    branch.as_deref(),
                    tag.as_deref(),
                    rev.as_deref(),
                    &target,
                    &name,
                )?;
                let entry = LockEntry {
                    kind: "git".into(),
                    source: url.clone(),
                    commit: Some(commit),
                };
                // Transitive deps in a git repo's iris.toml are relative to the
                // checked-out repo root (the installed target).
                (entry, target.clone())
            }
        };

        lock.entries.insert(name.clone(), lock_entry);

        // Check if this dep has its own iris.toml (transitive deps).
        // Read from the *original source* to get correct paths.
        let sub_manifest_path = transitive_from_dir.join("iris.toml");
        if sub_manifest_path.exists() {
            if let Ok(text) = fs::read_to_string(&sub_manifest_path) {
                if let Ok(sub_manifest) = Manifest::parse(&text) {
                    for (sub_name, sub_dep) in sub_manifest.deps {
                        if !installed.contains(&sub_name) {
                            queue.push((sub_name, sub_dep, transitive_from_dir.clone()));
                        }
                    }
                }
            }
        }
    }

    save_lock(&manifest_path, &lock)?;
    eprintln!("installed {} dependencies", installed.len());
    Ok(())
}

/// `iris pkg update [name]` — update git deps to latest matching ref.
pub fn cmd_update(only: Option<&str>) -> Result<(), String> {
    let (manifest_path, manifest) = load_manifest()?;
    let project_dir = manifest_path
        .parent()
        .ok_or("cannot determine project directory")?;
    let deps_dir = project_dir.join(".iris").join("deps");
    let mut lock = load_lock(&manifest_path);
    let mut updated = 0usize;

    for (name, dep) in &manifest.deps {
        if let Some(filter) = only {
            if name != filter {
                continue;
            }
        }
        match dep {
            Dep::Git {
                url,
                branch,
                tag,
                rev,
            } => {
                let target = deps_dir.join(name);
                if !target.exists() {
                    eprintln!(
                        "  {} — not installed, skipping (run `iris pkg install`)",
                        name
                    );
                    continue;
                }
                eprintln!("  {} — updating {}", name, url);
                let commit = git_pull_or_fetch(
                    &target,
                    url,
                    branch.as_deref(),
                    tag.as_deref(),
                    rev.as_deref(),
                    name,
                )?;
                lock.entries.insert(
                    name.clone(),
                    LockEntry {
                        kind: "git".into(),
                        source: url.clone(),
                        commit: Some(commit),
                    },
                );
                updated += 1;
            }
            Dep::Path(_) => {
                eprintln!("  {} — path dep, nothing to update", name);
            }
        }
    }

    save_lock(&manifest_path, &lock)?;
    eprintln!("updated {} git dependencies", updated);
    Ok(())
}

// ── Install helpers ───────────────────────────────────────────────────────────

/// Install a local path dep via symlink (Unix) or recursive copy (Windows).
fn install_path_dep(source: &Path, target: &Path, name: &str) -> Result<(), String> {
    if target.exists() {
        remove_dir_all_safe(target)?;
    }

    #[cfg(unix)]
    {
        if std::os::unix::fs::symlink(source, target).is_ok() {
            eprintln!("  {} → {} (symlink)", name, source.display());
            return Ok(());
        }
    }

    #[cfg(windows)]
    {
        if std::os::windows::fs::symlink_dir(source, target).is_ok() {
            eprintln!("  {} → {} (junction)", name, source.display());
            return Ok(());
        }
    }

    copy_dir_recursive(source, target)
        .map_err(|e| format!("dependency '{}': copy failed: {}", name, e))?;
    eprintln!("  {} → {} (copied)", name, source.display());
    Ok(())
}

/// Clone or fetch a git dependency. Returns the resolved commit SHA.
fn install_git_dep(
    url: &str,
    branch: Option<&str>,
    tag: Option<&str>,
    rev: Option<&str>,
    target: &Path,
    name: &str,
) -> Result<String, String> {
    if target.join(".git").exists() {
        // Already cloned — update to the pinned ref.
        git_pull_or_fetch(target, url, branch, tag, rev, name)
    } else {
        git_clone(url, branch, tag, rev, target, name)
    }
}

/// Fresh `git clone` into `target`. Returns resolved commit SHA.
fn git_clone(
    url: &str,
    branch: Option<&str>,
    tag: Option<&str>,
    rev: Option<&str>,
    target: &Path,
    name: &str,
) -> Result<String, String> {
    eprintln!("  {} — cloning {} ...", name, url);
    if target.exists() {
        remove_dir_all_safe(target)?;
    }

    // Choose which ref to clone.
    let ref_arg: Option<&str> = tag.or(branch);
    let mut cmd = Command::new("git");
    cmd.arg("clone").arg("--depth").arg("1");
    if let Some(r) = ref_arg {
        cmd.arg("--branch").arg(r);
    }
    cmd.arg(url).arg(target.as_os_str());

    let status = cmd
        .status()
        .map_err(|e| format!("dependency '{}': git clone failed: {}", name, e))?;
    if !status.success() {
        return Err(format!("dependency '{}': git clone failed", name));
    }

    // If a specific rev was requested, check it out.
    if let Some(r) = rev {
        git_checkout(target, r, name)?;
    }

    git_head_commit(target, name)
}

/// `git pull --ff-only` (or fetch + reset for pinned refs). Returns new commit SHA.
fn git_pull_or_fetch(
    target: &Path,
    url: &str,
    branch: Option<&str>,
    tag: Option<&str>,
    rev: Option<&str>,
    name: &str,
) -> Result<String, String> {
    // Ensure remote origin is set to the correct URL.
    let _ = Command::new("git")
        .args(["remote", "set-url", "origin", url])
        .current_dir(target)
        .status();

    if let Some(r) = rev {
        // Fetch and checkout exact SHA.
        let _ = Command::new("git")
            .args(["fetch", "--depth", "1", "origin", r])
            .current_dir(target)
            .status();
        git_checkout(target, r, name)?;
    } else if let Some(t) = tag {
        // Fetch the tag.
        let _ = Command::new("git")
            .args([
                "fetch",
                "--depth",
                "1",
                "origin",
                &format!("refs/tags/{}", t),
            ])
            .current_dir(target)
            .status();
        git_checkout(target, t, name)?;
    } else {
        // Pull the branch (or default).
        let mut pull = Command::new("git");
        pull.arg("pull").arg("--ff-only");
        if let Some(b) = branch {
            pull.arg("origin").arg(b);
        }
        let status = pull
            .current_dir(target)
            .status()
            .map_err(|e| format!("dependency '{}': git pull failed: {}", name, e))?;
        if !status.success() {
            return Err(format!("dependency '{}': git pull failed", name));
        }
    }

    git_head_commit(target, name)
}

/// Run `git checkout <ref>` in `target`.
fn git_checkout(target: &Path, git_ref: &str, name: &str) -> Result<(), String> {
    let status = Command::new("git")
        .args(["checkout", git_ref])
        .current_dir(target)
        .status()
        .map_err(|e| format!("dependency '{}': git checkout failed: {}", name, e))?;
    if !status.success() {
        return Err(format!(
            "dependency '{}': git checkout '{}' failed",
            name, git_ref
        ));
    }
    Ok(())
}

/// Return the current HEAD commit SHA (40 chars) for the repo at `target`.
fn git_head_commit(target: &Path, name: &str) -> Result<String, String> {
    let out = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(target)
        .output()
        .map_err(|e| format!("dependency '{}': git rev-parse failed: {}", name, e))?;
    if !out.status.success() {
        return Err(format!("dependency '{}': git rev-parse HEAD failed", name));
    }
    let sha = String::from_utf8_lossy(&out.stdout).trim().to_string();
    eprintln!("  {} → {}", name, &sha[..sha.len().min(12)]);
    Ok(sha)
}

fn copy_dir_recursive(src: &Path, dst: &Path) -> io::Result<()> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let ty = entry.file_type()?;
        let dest = dst.join(entry.file_name());
        if ty.is_dir() {
            copy_dir_recursive(&entry.path(), &dest)?;
        } else {
            fs::copy(entry.path(), dest)?;
        }
    }
    Ok(())
}

fn remove_dir_all_safe(path: &Path) -> Result<(), String> {
    fs::remove_dir_all(path).map_err(|e| format!("cannot remove {}: {}", path.display(), e))
}

// ── Build / run ───────────────────────────────────────────────────────────────

/// `iris pkg build` / `iris pkg run` — install deps then compile the entry point.
pub fn cmd_build(run_after: bool) -> Result<(), String> {
    let (manifest_path, manifest) = load_manifest()?;
    let project_dir = manifest_path
        .parent()
        .ok_or("cannot determine project directory")?;

    cmd_install()?;

    let entry_path = project_dir.join(&manifest.entry);
    if !entry_path.exists() {
        return Err(format!(
            "entry file '{}' not found (set `entry` in [package])",
            entry_path.display()
        ));
    }

    // Add each installed dep directory to the search path.
    let deps_dir = project_dir.join(".iris").join("deps");
    let mut extra_paths: Vec<PathBuf> = Vec::new();
    if deps_dir.exists() {
        for entry in fs::read_dir(&deps_dir)
            .map_err(|e| e.to_string())?
            .flatten()
        {
            if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                extra_paths.push(entry.path());
            }
        }
    }

    let extra_refs: Vec<&Path> = extra_paths.iter().map(|p| p.as_path()).collect();
    let compiler = crate::FileCompiler::new();
    let main_ast = compiler
        .compile_file_to_ast(&entry_path, &extra_refs)
        .map_err(|e| format!("{}", e))?;

    let module_name = entry_path
        .file_stem()
        .and_then(|n| n.to_str())
        .unwrap_or("main");
    let ir =
        crate::compile_ast_to_module(&main_ast, module_name, None).map_err(|e| format!("{}", e))?;

    let output_name = format!("{}{}", manifest.name, std::env::consts::EXE_SUFFIX);
    let output_path = project_dir.join(&output_name);
    crate::codegen::build_binary(&ir, &output_path).map_err(|e| format!("{}", e))?;
    eprintln!("wrote binary: {}", output_path.display());

    if run_after {
        let run_path = fs::canonicalize(&output_path).unwrap_or_else(|_| output_path.clone());
        let status = Command::new(&run_path)
            .current_dir(project_dir)
            .status()
            .map_err(|e| format!("cannot run binary: {}", e))?;
        std::process::exit(status.code().unwrap_or(1));
    }

    Ok(())
}

// ── CLI dispatcher ────────────────────────────────────────────────────────────

/// Parse `iris pkg <subcmd> [args...]` and dispatch.
pub fn run_pkg_command(args: &[String]) -> Result<(), String> {
    let sub = args.get(2).map(|s| s.as_str()).unwrap_or("help");

    match sub {
        "init" => cmd_init(),

        "add" => {
            let name = args.get(3)
                .ok_or("usage: iris pkg add <name> --path <p> | --git <url> [--tag t | --rev r | --branch b]")?;

            // Collect all flags.
            let mut path_val: Option<String> = None;
            let mut git_val: Option<String> = None;
            let mut tag_val: Option<String> = None;
            let mut rev_val: Option<String> = None;
            let mut branch_val: Option<String> = None;

            let mut i = 4usize;
            while i < args.len() {
                match args[i].as_str() {
                    "--path" => {
                        i += 1;
                        path_val = args.get(i).cloned();
                    }
                    "--git" => {
                        i += 1;
                        git_val = args.get(i).cloned();
                    }
                    "--tag" => {
                        i += 1;
                        tag_val = args.get(i).cloned();
                    }
                    "--rev" => {
                        i += 1;
                        rev_val = args.get(i).cloned();
                    }
                    "--branch" => {
                        i += 1;
                        branch_val = args.get(i).cloned();
                    }
                    other => return Err(format!("unknown flag: {}", other)),
                }
                i += 1;
            }

            let dep = if let Some(p) = path_val {
                Dep::Path(p)
            } else if let Some(url) = git_val {
                Dep::Git {
                    url,
                    branch: branch_val,
                    tag: tag_val,
                    rev: rev_val,
                }
            } else {
                return Err("usage: iris pkg add <name> --path <p> | --git <url>".into());
            };

            cmd_add(name, dep)
        }

        "remove" | "rm" => {
            let name = args.get(3).ok_or("usage: iris pkg remove <name>")?;
            cmd_remove(name)
        }

        "install" | "i" => cmd_install(),

        "update" | "u" => {
            let only = args.get(3).map(|s| s.as_str());
            cmd_update(only)
        }

        "list" | "ls" => cmd_list(),

        "check" => cmd_check(),

        "build" | "b" => cmd_build(false),

        "run" | "r" => cmd_build(true),

        "help" | "--help" | "-h" => {
            eprintln!("{}", pkg_help_text());
            Ok(())
        }

        other => Err(format!(
            "unknown pkg subcommand: '{}'\n\n{}",
            other,
            pkg_help_text()
        )),
    }
}

fn pkg_help_text() -> &'static str {
    "IRIS Package Manager (local/git)\n\
     \n\
     Usage: iris pkg <command> [args...]\n\
     \n\
     Commands:\n\
       init                                    Create a new iris.toml\n\
       add <name> --path <path>                Add a local path dependency\n\
       add <name> --git <url>                  Add a git dependency (default branch)\n\
       add <name> --git <url> --tag <tag>      Pin to a git tag  (e.g. v1.2.0)\n\
       add <name> --git <url> --rev <sha>      Pin to a commit SHA\n\
       add <name> --git <url> --branch <name>  Track a specific branch\n\
       remove <name>                           Remove a dependency\n\
       install                                 Fetch/sync all deps into .iris/deps/\n\
       update [name]                           Update git deps to latest matching ref\n\
       list                                    List dependencies (with lock info)\n\
       check                                   Verify all deps are installed\n\
       build                                   Install deps and build the project binary\n\
       run                                     Install deps, build, and run the project\n\
       help                                    Show this help message\n\
     \n\
     Aliases: rm=remove, i=install, u=update, ls=list, b=build, r=run\n\
     \n\
     Lock file:\n\
       iris.lock is auto-generated next to iris.toml.\n\
       Commit it to source control for reproducible builds.\n\
     \n\
     Transitive dependencies:\n\
       If an installed dep has its own iris.toml, its dependencies\n\
       are automatically installed into the same .iris/deps/ directory.\n"
}
