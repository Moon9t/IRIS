//! IRIS Package Manager (`iris pkg`).
//!
//! Supports local path, git, and registry-based dependencies.
//!
//! ## Manifest format (`iris.toml`)
//!
//! ```toml
//! [package]
//! name    = "my-project"
//! version = "0.1.0"
//! entry   = "main.iris"
//! # Optional: default registry for unqualified registry deps
//! registry = "https://iris-pkg.example.com"
//!
//! [dependencies]
//! utils = { path = "../shared/utils" }
//! web   = { git = "https://github.com/user/iris-web.git" }
//! auth  = { git = "https://github.com/user/iris-auth.git", tag  = "v1.2.0" }
//! core  = { git = "https://github.com/user/iris-core.git", rev  = "a1b2c3d" }
//! dev   = { git = "https://github.com/user/iris-dev.git",  branch = "main" }
//! json  = { registry = "json", version = "^1.0.0" }
//! ```
//!
//! ## Lock file (`iris.lock`)
//!
//! Auto-generated next to `iris.toml`. Commit it to source control for
//! reproducible builds. Records git commit SHAs and content checksums.
//!
//! ## Commands
//!
//! - `iris pkg init`                  — create a new `iris.toml`
//! - `iris pkg add <n> --path <p>`    — add a local path dependency
//! - `iris pkg add <n> --git <url>`   — add a git dependency
//! - `iris pkg add <n> --registry <pkg> --version <req>` — registry dep
//! - `iris pkg remove <name>`         — remove a dependency
//! - `iris pkg install [--offline]`   — fetch/sync all deps into `.iris/deps/`
//! - `iris pkg update [name]`         — update deps to latest matching ref
//! - `iris pkg list`                  — list current dependencies
//! - `iris pkg check`                 — verify all deps are installed
//! - `iris pkg build [--offline]`     — install deps + build entry binary
//! - `iris pkg run [--offline]`       — build + run
//! - `iris pkg vendor`                — download all deps for offline builds
//!
//! ## Registry format
//!
//! A registry is a JSON index at `<registry>/index.json`:
//! ```json
//! {
//!   "packages": {
//!     "json": {
//!       "versions": {
//!         "1.0.0": {
//!           "url": "https://example.com/pkgs/json-1.0.0.tar.gz",
//!           "checksum": "sha256:..."
//!         }
//!       }
//!     }
//!   }
//! }
//! ```

use semver::{Version, VersionReq};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;

const VENDOR_DIR: &str = ".iris/vendor";

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
    /// Default registry URL for registry deps (optional).
    pub registry: Option<String>,
    pub deps: BTreeMap<String, Dep>,
}

/// A dependency source.
#[derive(Debug, Clone)]
pub enum Dep {
    /// `name = { path = "..." }`
    Path(String),
    /// `name = { git = "...", [branch = "..."], [tag = "..."], [rev = "..."] }`
    Git {
        url: String,
        branch: Option<String>,
        tag: Option<String>,
        rev: Option<String>,
    },
    /// `name = { registry = "...", version = "..." }`
    Registry {
        /// Package name in the registry (may differ from the dep key).
        package: String,
        /// Semver version requirement, e.g. "^1.0.0", "~2.3", ">=0.5".
        version_req: String,
    },
}

impl fmt::Display for Dep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Dep::Path(p) => write!(f, "{{ path = \"{}\" }}", p),
            Dep::Git { url, branch, tag, rev } => {
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
            Dep::Registry { package, version_req } => {
                write!(f, "{{ registry = \"{}\", version = \"{}\" }}", package, version_req)
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
    #[serde(default)]
    registry: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
enum TomlDep {
    Simple(String),
    Detailed(DetailedDep),
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
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
    /// Registry package name.
    #[serde(default)]
    registry: Option<String>,
    /// Semver requirement (used with `registry`).
    #[serde(default)]
    version: Option<String>,
}

impl Manifest {
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
                        Dep::Git { url: s, branch: None, tag: None, rev: None }
                    } else {
                        Dep::Path(s)
                    }
                }
                TomlDep::Detailed(d) => {
                    if let Some(p) = d.path {
                        Dep::Path(p)
                    } else if let Some(git) = d.git {
                        Dep::Git { url: git, branch: d.branch, tag: d.tag, rev: d.rev }
                    } else if let Some(pkg) = d.registry {
                        Dep::Registry {
                            package: pkg,
                            version_req: d.version.unwrap_or_else(|| "*".into()),
                        }
                    } else {
                        return Err(format!(
                            "dependency '{}' must specify 'path', 'git', or 'registry'",
                            name
                        ));
                    }
                }
            };
            deps.insert(name, dep);
        }

        let pkg = toml_manifest.package;
        Ok(Manifest {
            name: pkg.name,
            version: pkg.version,
            entry: pkg.entry,
            description: pkg.description.unwrap_or_default(),
            license: pkg.license.unwrap_or_default(),
            repository: pkg.repository.unwrap_or_default(),
            registry: pkg.registry,
            deps,
        })
    }

    pub fn to_toml(&self) -> String {
        let mut toml_deps = BTreeMap::new();
        for (name, dep) in &self.deps {
            let toml_dep = match dep {
                Dep::Path(p) => TomlDep::Detailed(DetailedDep {
                    path: Some(p.clone()),
                    ..Default::default()
                }),
                Dep::Git { url, branch, tag, rev } => TomlDep::Detailed(DetailedDep {
                    path: None,
                    git: Some(url.clone()),
                    branch: branch.clone(),
                    tag: tag.clone(),
                    rev: rev.clone(),
                    ..Default::default()
                }),
                Dep::Registry { package, version_req } => TomlDep::Detailed(DetailedDep {
                    registry: Some(package.clone()),
                    version: Some(version_req.clone()),
                    ..Default::default()
                }),
            };
            toml_deps.insert(name.clone(), toml_dep);
        }

        let toml_manifest = TomlManifest {
            package: TomlPackage {
                name: self.name.clone(),
                version: self.version.clone(),
                entry: self.entry.clone(),
                description: if self.description.is_empty() { None } else { Some(self.description.clone()) },
                license: if self.license.is_empty() { None } else { Some(self.license.clone()) },
                repository: if self.repository.is_empty() { None } else { Some(self.repository.clone()) },
                registry: self.registry.clone(),
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

#[derive(Debug, Clone)]
pub struct LockEntry {
    pub kind: String,
    pub source: String,
    pub commit: Option<String>,
    /// SHA-256 hex checksum of the installed package directory (computed as
    /// the hash of a canonical listing of all files).
    pub checksum: Option<String>,
}

#[derive(Debug, Default)]
pub struct LockFile {
    pub entries: BTreeMap<String, LockEntry>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
struct TomlLockFile {
    #[serde(default)]
    dep: BTreeMap<String, TomlLockEntry>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
struct TomlLockEntry {
    kind: String,
    source: String,
    #[serde(default)]
    commit: Option<String>,
    #[serde(default)]
    checksum: Option<String>,
}

impl LockFile {
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
                    checksum: entry.checksum,
                },
            );
        }
        LockFile { entries }
    }

    pub fn to_text(&self) -> String {
        let mut toml_dep = BTreeMap::new();
        for (name, entry) in &self.entries {
            toml_dep.insert(
                name.clone(),
                TomlLockEntry {
                    kind: entry.kind.clone(),
                    source: entry.source.clone(),
                    commit: entry.commit.clone(),
                    checksum: entry.checksum.clone(),
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
    let text = fs::read_to_string(&path).map_err(|e| format!("cannot read {}: {}", path.display(), e))?;
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
    fs::read_to_string(&lp).map(|s| LockFile::parse(&s)).unwrap_or_default()
}

fn save_lock(manifest_path: &Path, lock: &LockFile) -> Result<(), String> {
    let lp = lock_path(manifest_path);
    fs::write(&lp, lock.to_text()).map_err(|e| format!("cannot write iris.lock: {}", e))
}

// ── Content hashing ───────────────────────────────────────────────────────────

/// Compute SHA-256 of all files under `dir`, sorted by relative path.
/// Returns a hex string, or `None` if the directory doesn't exist.
fn dir_checksum(dir: &Path) -> Option<String> {
    let mut entries: Vec<PathBuf> = Vec::new();
    collect_files(dir, dir, &mut entries).ok()?;
    if entries.is_empty() {
        return None;
    }
    entries.sort();
    let mut hasher = Sha256::new();
    for rel in &entries {
        let abs = dir.join(rel);
        if let Ok(data) = fs::read(&abs) {
            hasher.update(rel.to_string_lossy().as_bytes());
            hasher.update(&[0u8; 1]);
            hasher.update(&data);
            hasher.update(&[0u8; 1]);
        }
    }
    Some(hex_encode(&hasher.finalize()))
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

fn collect_files(base: &Path, dir: &Path, out: &mut Vec<PathBuf>) -> io::Result<()> {
    if !dir.is_dir() {
        return Ok(());
    }
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let rel = path.strip_prefix(base).unwrap_or(&path).to_path_buf();
        if entry.file_type()?.is_dir() {
            collect_files(base, &path, out)?;
        } else {
            out.push(rel);
        }
    }
    Ok(())
}

// ── Registry helpers ──────────────────────────────────────────────────────────

/// Fetch the registry index and resolve a version constraint to a concrete
/// version + download URL + checksum.
fn resolve_registry_dep(
    registry_url: &str,
    package: &str,
    version_req: &str,
    _offline: bool,
) -> Result<(Version, String, String), String> {
    // Parse the version requirement.
    let req = if version_req == "*" {
        VersionReq::STAR
    } else {
        VersionReq::parse(version_req)
            .map_err(|e| format!("invalid version requirement '{}': {}", version_req, e))?
    };

    // Fetch the registry index.
    let index_url = format!("{}/index.json", registry_url.trim_end_matches('/'));
    let resp = ureq_get(&index_url).map_err(|e| format!("failed to fetch registry index: {}", e))?;
    let text = String::from_utf8(resp).map_err(|e| format!("invalid UTF-8 in registry index: {}", e))?;
    let index: RegistryIndex =
        serde_json::from_str(&text).map_err(|e| format!("invalid registry index: {}", e))?;

    let versions = index
        .packages
        .get(package)
        .and_then(|p| p.versions.as_ref())
        .ok_or_else(|| format!("package '{}' not found in registry", package))?;

    // Find the latest matching version.
    let mut best: Option<(Version, &RegistryVersionEntry)> = None;
    for (ver_str, entry) in versions {
        if let Ok(ver) = Version::parse(ver_str) {
            if req.matches(&ver) {
                let is_better = match &best {
                    None => true,
                    Some((b, _)) => ver > *b,
                };
                if is_better {
                    best = Some((ver, entry));
                }
            }
        }
    }

    let (version, entry) = best.ok_or_else(|| {
        format!(
            "no version matching '{}' found for package '{}'",
            version_req, package
        )
    })?;

    Ok((version, entry.url.clone(), entry.checksum.clone()))
}

#[derive(Deserialize, Debug)]
struct RegistryIndex {
    packages: BTreeMap<String, RegistryPackage>,
}

#[derive(Deserialize, Debug)]
struct RegistryPackage {
    #[serde(default)]
    versions: Option<BTreeMap<String, RegistryVersionEntry>>,
}

#[derive(Deserialize, Debug, Clone)]
struct RegistryVersionEntry {
    url: String,
    /// "sha256:<hex>" format.
    #[serde(default)]
    checksum: String,
}

/// Download a tarball from `url`, verify its checksum, and extract into `target`.
fn download_and_extract(url: &str, expected_checksum: &str, target: &Path, name: &str) -> Result<(), String> {
    eprintln!("  {} — downloading {} ...", name, url);
    let data = ureq_get(url).map_err(|e| format!("failed to download {}: {}", url, e))?;

    // Verify checksum if provided.
    if let Some(expected) = expected_checksum.strip_prefix("sha256:") {
        let actual = hex_encode(&Sha256::digest(&data));
        if actual != expected {
            return Err(format!(
                "checksum mismatch for {}: expected sha256:{} but got sha256:{}",
                name, expected, actual
            ));
        }
        eprintln!("  {} — checksum OK", name);
    }

    // Extract tarball (gzipped).
    let decoder = flate2::read::GzDecoder::new(&data[..]);
    let mut archive = tar::Archive::new(decoder);
    if target.exists() {
        fs::remove_dir_all(target).map_err(|e| format!("cannot remove {}: {}", target.display(), e))?;
    }
    fs::create_dir_all(target).map_err(|e| format!("cannot create {}: {}", target.display(), e))?;
    archive
        .unpack(target)
        .map_err(|e| format!("failed to extract {}: {}", name, e))?;

    eprintln!("  {} — extracted to {}", name, target.display());
    Ok(())
}

/// Minimal HTTP GET via curl.exe (Windows) or `curl` (Unix).
fn ureq_get(url: &str) -> Result<Vec<u8>, String> {
    let output = Command::new(if cfg!(windows) { "curl.exe" } else { "curl" })
        .args(["-sS", "-L", url])
        .output()
        .map_err(|e| format!("curl failed: {}", e))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("curl error: {}", stderr.trim()));
    }
    Ok(output.stdout)
}

// ── Commands ──────────────────────────────────────────────────────────────────

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
        registry: None,
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
        )
        .map_err(|e| format!("cannot write main.iris: {}", e))?;
    }

    fs::create_dir_all(cwd.join(".iris")).map_err(|e| format!("cannot create .iris/: {}", e))?;

    eprintln!("initialized IRIS project '{}' in {}", dir_name, cwd.display());
    Ok(())
}

pub fn cmd_add(name: &str, dep: Dep) -> Result<(), String> {
    let (path, mut manifest) = load_manifest()?;
    manifest.deps.insert(name.to_string(), dep.clone());
    save_manifest(&path, &manifest)?;
    eprintln!("added dependency '{}' = {}", name, dep);
    Ok(())
}

pub fn cmd_remove(name: &str) -> Result<(), String> {
    let (path, mut manifest) = load_manifest()?;
    if manifest.deps.remove(name).is_none() {
        return Err(format!("dependency '{}' not found in iris.toml", name));
    }
    let mut lock = load_lock(&path);
    lock.entries.remove(name);
    save_manifest(&path, &manifest)?;
    save_lock(&path, &lock)?;
    eprintln!("removed dependency '{}'", name);
    Ok(())
}

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
                    &e.commit.as_deref().unwrap_or("")[..8.min(e.commit.as_deref().unwrap_or("").len())]
                ),
                Some(e) if e.checksum.is_some() => format!(
                    " [{}]",
                    &e.checksum.as_deref().unwrap_or("")[..12.min(e.checksum.as_deref().unwrap_or("").len())]
                ),
                _ => String::new(),
            };
            eprintln!("  {} = {}{}", name, dep, pin);
        }
    }
    Ok(())
}

pub fn cmd_check() -> Result<(), String> {
    let (manifest_path, manifest) = load_manifest()?;
    let project_dir = manifest_path.parent().ok_or("cannot determine project directory")?;
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
        Err(format!("{} dependency/ies missing — run `iris pkg install`", missing.len()))
    }
}

/// `iris pkg install [--offline]`
pub fn cmd_install(offline: bool) -> Result<(), String> {
    let (manifest_path, manifest) = load_manifest()?;
    let project_dir = manifest_path.parent().ok_or("cannot determine project directory")?;
    let deps_dir = project_dir.join(".iris").join("deps");

    fs::create_dir_all(&deps_dir).map_err(|e| format!("cannot create .iris/deps/: {}", e))?;

    if manifest.deps.is_empty() {
        eprintln!("no dependencies to install");
        return Ok(());
    }

    let mut lock = load_lock(&manifest_path);

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
                let cs = dir_checksum(&target);
                let entry = LockEntry {
                    kind: "path".into(),
                    source: source.to_string_lossy().into_owned(),
                    commit: None,
                    checksum: cs,
                };
                (entry, source.clone())
            }
            Dep::Git { url, branch, tag, rev } => {
                let commit = install_git_dep(url, branch.as_deref(), tag.as_deref(), rev.as_deref(), &target, &name)?;
                let cs = dir_checksum(&target);
                let entry = LockEntry {
                    kind: "git".into(),
                    source: url.clone(),
                    commit: Some(commit),
                    checksum: cs,
                };
                (entry, target.clone())
            }
            Dep::Registry { package, version_req } => {
                let registry_url = effective_registry(manifest.registry.as_deref());
                let (version, url, checksum) = resolve_registry_dep(&registry_url, package, version_req, offline)?;
                download_and_extract(&url, &checksum, &target, &name)?;
                let cs = dir_checksum(&target);
                let entry = LockEntry {
                    kind: "registry".into(),
                    source: format!("{}@{}", package, version),
                    commit: None,
                    checksum: cs,
                };
                (entry, target.clone())
            }
        };

        lock.entries.insert(name.clone(), lock_entry);

        // Transitive dependencies.
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

/// `iris pkg update [name]`.
pub fn cmd_update(only: Option<&str>) -> Result<(), String> {
    let (manifest_path, manifest) = load_manifest()?;
    let project_dir = manifest_path.parent().ok_or("cannot determine project directory")?;
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
            Dep::Git { url, branch, tag, rev } => {
                let target = deps_dir.join(name);
                if !target.exists() {
                    eprintln!("  {} — not installed, skipping (run `iris pkg install`)", name);
                    continue;
                }
                eprintln!("  {} — updating {}", name, url);
                let commit = git_pull_or_fetch(&target, url, branch.as_deref(), tag.as_deref(), rev.as_deref(), name)?;
                let cs = dir_checksum(&target);
                lock.entries.insert(
                    name.clone(),
                    LockEntry {
                        kind: "git".into(),
                        source: url.clone(),
                        commit: Some(commit),
                        checksum: cs,
                    },
                );
                updated += 1;
            }
            Dep::Path(_) => {
                eprintln!("  {} — path dep, nothing to update", name);
            }
            Dep::Registry { package, version_req } => {
                let registry_url = effective_registry(manifest.registry.as_deref());
                let target = deps_dir.join(name);
                if target.exists() {
                    fs::remove_dir_all(&target)
                        .map_err(|e| format!("cannot remove {}: {}", target.display(), e))?;
                }
                let (version, url, checksum) =
                    resolve_registry_dep(&registry_url, package, version_req, false)?;
                download_and_extract(&url, &checksum, &target, name)?;
                let cs = dir_checksum(&target);
                lock.entries.insert(
                    name.clone(),
                    LockEntry {
                        kind: "registry".into(),
                        source: format!("{}@{}", package, version),
                        commit: None,
                        checksum: cs,
                    },
                );
                updated += 1;
            }
        }
    }

    save_lock(&manifest_path, &lock)?;
    eprintln!("updated {} dependencies", updated);
    Ok(())
}

/// `iris pkg vendor` — download all deps into `.iris/vendor/` for offline builds.
pub fn cmd_vendor() -> Result<(), String> {
    let (manifest_path, manifest) = load_manifest()?;
    let project_dir = manifest_path.parent().ok_or("cannot determine project directory")?;
    let vendor_dir = project_dir.join(VENDOR_DIR);
    let deps_dir = project_dir.join(".iris").join("deps");

    fs::create_dir_all(&vendor_dir)
        .map_err(|e| format!("cannot create {}: {}", vendor_dir.display(), e))?;

    if manifest.deps.is_empty() {
        eprintln!("no dependencies to vendor");
        return Ok(());
    }

    // Ensure deps are installed first.
    if !deps_dir.exists() || deps_dir.read_dir().map(|mut i| i.next().is_none()).unwrap_or(true) {
        eprintln!("no installed deps found — running install first ...");
        cmd_install(false)?;
    }

    let mut vendored = 0usize;
    for name in manifest.deps.keys() {
        let src = deps_dir.join(name);
        let dst = vendor_dir.join(name);
        if !src.exists() {
            eprintln!("  {} — not installed, skipping", name);
            continue;
        }
        if dst.exists() {
            remove_dir_all_safe(&dst)?;
        }
        copy_dir_recursive(&src, &dst)
            .map_err(|e| format!("cannot copy {}: {}", name, e))?;
        eprintln!("  {} → {}", name, dst.display());
        vendored += 1;
    }

    eprintln!("vendored {} dependencies into {}", vendored, vendor_dir.display());
    Ok(())
}

// ── Install helpers ───────────────────────────────────────────────────────────

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

fn install_git_dep(
    url: &str,
    branch: Option<&str>,
    tag: Option<&str>,
    rev: Option<&str>,
    target: &Path,
    name: &str,
) -> Result<String, String> {
    if target.join(".git").exists() {
        git_pull_or_fetch(target, url, branch, tag, rev, name)
    } else {
        git_clone(url, branch, tag, rev, target, name)
    }
}

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

    if let Some(r) = rev {
        git_checkout(target, r, name)?;
    }

    git_head_commit(target, name)
}

fn git_pull_or_fetch(
    target: &Path,
    url: &str,
    branch: Option<&str>,
    tag: Option<&str>,
    rev: Option<&str>,
    name: &str,
) -> Result<String, String> {
    let _ = Command::new("git")
        .args(["remote", "set-url", "origin", url])
        .current_dir(target)
        .status();

    if let Some(r) = rev {
        let _ = Command::new("git")
            .args(["fetch", "--depth", "1", "origin", r])
            .current_dir(target)
            .status();
        git_checkout(target, r, name)?;
    } else if let Some(t) = tag {
        let _ = Command::new("git")
            .args(["fetch", "--depth", "1", "origin", &format!("refs/tags/{}", t)])
            .current_dir(target)
            .status();
        git_checkout(target, t, name)?;
    } else {
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

fn git_checkout(target: &Path, git_ref: &str, name: &str) -> Result<(), String> {
    let status = Command::new("git")
        .args(["checkout", git_ref])
        .current_dir(target)
        .status()
        .map_err(|e| format!("dependency '{}': git checkout failed: {}", name, e))?;
    if !status.success() {
        return Err(format!("dependency '{}': git checkout '{}' failed", name, git_ref));
    }
    Ok(())
}

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

pub fn cmd_build(run_after: bool, offline: bool) -> Result<(), String> {
    let (manifest_path, manifest) = load_manifest()?;
    let project_dir = manifest_path.parent().ok_or("cannot determine project directory")?;

    cmd_install(offline)?;

    let entry_path = project_dir.join(&manifest.entry);
    if !entry_path.exists() {
        return Err(format!(
            "entry file '{}' not found (set `entry` in [package])",
            entry_path.display()
        ));
    }

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
    let ir = crate::compile_ast_to_module(&main_ast, module_name, None)
        .map_err(|e| format!("{}", e))?;

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

// ── Default registry ──────────────────────────────────────────────────────────

/// Hard-coded default registry for when `[package].registry` is not set.
/// Users can override by setting `IRIS_REGISTRY` env var.
const DEFAULT_REGISTRY: &str = "https://iris-pkg.example.com";

fn effective_registry(manifest_registry: Option<&str>) -> String {
    if let Ok(env_reg) = std::env::var("IRIS_REGISTRY") {
        return env_reg;
    }
    manifest_registry.unwrap_or(DEFAULT_REGISTRY).to_string()
}

// ── CLI dispatcher ────────────────────────────────────────────────────────────

pub fn run_pkg_command(args: &[String]) -> Result<(), String> {
    // `args` are the trailing arguments after `iris pkg` (e.g. ["init"] or ["add", "foo", ...]).
    let sub = args.first().map(|s| s.as_str()).unwrap_or("help");

    match sub {
        "init" => cmd_init(),

        "add" => {
            let name = args.get(1)
                .ok_or("usage: iris pkg add <name> --path <p> | --git <url> [--tag t | --rev r | --branch b] | --registry <pkg> --version <req>")?;

            let mut path_val: Option<String> = None;
            let mut git_val: Option<String> = None;
            let mut tag_val: Option<String> = None;
            let mut rev_val: Option<String> = None;
            let mut branch_val: Option<String> = None;
            let mut registry_val: Option<String> = None;
            let mut version_val: Option<String> = None;

            let mut i = 2usize;
            while i < args.len() {
                match args[i].as_str() {
                    "--path" => { i += 1; path_val = args.get(i).cloned(); }
                    "--git" => { i += 1; git_val = args.get(i).cloned(); }
                    "--tag" => { i += 1; tag_val = args.get(i).cloned(); }
                    "--rev" => { i += 1; rev_val = args.get(i).cloned(); }
                    "--branch" => { i += 1; branch_val = args.get(i).cloned(); }
                    "--registry" => { i += 1; registry_val = args.get(i).cloned(); }
                    "--version" => { i += 1; version_val = args.get(i).cloned(); }
                    other => return Err(format!("unknown flag: {}", other)),
                }
                i += 1;
            }

            let dep = if let Some(p) = path_val {
                Dep::Path(p)
            } else if let Some(url) = git_val {
                Dep::Git { url, branch: branch_val, tag: tag_val, rev: rev_val }
            } else if let Some(pkg) = registry_val {
                Dep::Registry {
                    package: pkg,
                    version_req: version_val.unwrap_or_else(|| "*".into()),
                }
            } else {
                return Err(
                    "usage: iris pkg add <name> --path <p> | --git <url> [--tag t | --rev r | --branch b] | --registry <pkg> --version <req>"
                        .into(),
                );
            };

            cmd_add(name, dep)
        }

        "remove" | "rm" => {
            let name = args.get(1).ok_or("usage: iris pkg remove <name>")?;
            cmd_remove(name)
        }

        "install" | "i" => {
            let offline = args.contains(&"--offline".to_string());
            cmd_install(offline)
        }

        "update" | "u" => {
            let only = args.get(1).map(|s| s.as_str());
            cmd_update(only)
        }

        "list" | "ls" => cmd_list(),

        "check" => cmd_check(),

        "vendor" => cmd_vendor(),

        "build" | "b" => {
            let offline = args.contains(&"--offline".to_string());
            cmd_build(false, offline)
        }

        "run" | "r" => {
            let offline = args.contains(&"--offline".to_string());
            cmd_build(true, offline)
        }

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
    "IRIS Package Manager\n\
     \n\
     Usage: iris pkg <command> [args...]\n\
     \n\
     Commands:\n\
       init                                        Create a new iris.toml\n\
       add <name> --path <path>                    Add a local path dependency\n\
       add <name> --git <url> [--branch|--tag|--rev] Add a git dependency\n\
       add <name> --registry <pkg> --version <req> Add a registry dependency\n\
       remove <name>                               Remove a dependency\n\
       install [--offline]                         Fetch/sync all deps into .iris/deps/\n\
       update [name]                               Update deps to latest matching ref\n\
       list                                        List dependencies (with lock info)\n\
       check                                       Verify all deps are installed\n\
       vendor                                      Download all deps for offline builds\n\
       build [--offline]                           Install deps and build the project binary\n\
       run [--offline]                             Install deps, build, and run\n\
       help                                        Show this help message\n\
     \n\
     Aliases: rm=remove, i=install, u=update, ls=list, b=build, r=run\n\
     \n\
     The --offline flag uses vendored copies from .iris/vendor/ when set.\n\
     \n\
     Lock file:\n\
       iris.lock is auto-generated next to iris.toml.\n\
       Commit it to source control for reproducible builds.\n\
     \n\
     Registry:\n\
       Set [package].registry in iris.toml or IRIS_REGISTRY env var.\n\
     \n\
     Vendor:\n\
       Run `iris pkg vendor` to copy all deps into .iris/vendor/.\n\
       Then use `iris pkg install/build/run --offline` to use them\n\
       without network access.\n\
     \n\
     Transitive dependencies:\n\
       If an installed dep has its own iris.toml, its dependencies\n\
       are automatically installed into the same .iris/deps/ directory.\n"
}
