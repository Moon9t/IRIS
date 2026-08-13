//! Basic package manager for IRIS.
//!
//! Supports `iris install` (install deps from `iris.toml`) and
//! `iris install <url>` (install a package from a Git URL).
//!
//! Packages live in `iris_packages/<name>/` next to the project.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// A parsed package from an `iris.toml` manifest.
#[derive(Debug, Clone)]
pub struct Package {
    pub name: String,
    pub version: String,
    pub description: String,
    pub path: PathBuf,
}

const PACKAGES_DIR: &str = "iris_packages";

/// Directory where installed packages live.
fn packages_dir() -> Result<PathBuf, String> {
    let cwd = std::env::current_dir().map_err(|e| format!("cannot read cwd: {}", e))?;
    Ok(cwd.join(PACKAGES_DIR))
}

/// Install a package from a Git URL into `iris_packages/<name>/`.
pub fn install_from_url(url: &str) -> Result<Package, String> {
    let name = extract_package_name(url)?;
    let pkgs_dir = packages_dir()?;
    fs::create_dir_all(&pkgs_dir)
        .map_err(|e| format!("cannot create {}: {}", PACKAGES_DIR, e))?;

    let target = pkgs_dir.join(&name);

    if target.join(".git").exists() {
        eprintln!("  {} — updating ...", name);
        let status = Command::new("git")
            .args(["pull", "--ff-only"])
            .current_dir(&target)
            .status()
            .map_err(|e| format!("git pull failed: {}", e))?;
        if !status.success() {
            return Err(format!("git pull failed for '{}'", name));
        }
    } else {
        eprintln!("  {} — cloning {} ...", name, url);
        let status = Command::new("git")
            .args(["clone", "--depth", "1", url, target.to_str().unwrap_or("")])
            .status()
            .map_err(|e| format!("git clone failed: {}", e))?;
        if !status.success() {
            return Err(format!("git clone failed for '{}'", name));
        }
    }

    let pkg = read_manifest(&target)?;
    eprintln!("  installed {} v{}", pkg.name, pkg.version);
    Ok(pkg)
}

/// Install all dependencies listed in `iris.toml` in the current directory.
pub fn install_all() -> Result<Vec<Package>, String> {
    let cwd = std::env::current_dir().map_err(|e| format!("cannot read cwd: {}", e))?;
    let manifest_path = cwd.join("iris.toml");

    if !manifest_path.exists() {
        return Err("no iris.toml found in current directory".into());
    }

    let text = fs::read_to_string(&manifest_path)
        .map_err(|e| format!("cannot read iris.toml: {}", e))?;
    let deps = parse_dependencies(&text)?;

    if deps.is_empty() {
        eprintln!("no dependencies to install");
        return Ok(vec![]);
    }

    let pkgs_dir = packages_dir()?;
    fs::create_dir_all(&pkgs_dir)
        .map_err(|e| format!("cannot create {}: {}", PACKAGES_DIR, e))?;

    let mut installed = Vec::new();
    for (dep_name, dep_url) in &deps {
        let target = pkgs_dir.join(dep_name);

        if target.join(".git").exists() {
            eprintln!("  {} — updating ...", dep_name);
            let status = Command::new("git")
                .args(["pull", "--ff-only"])
                .current_dir(&target)
                .status()
                .map_err(|e| format!("git pull failed: {}", e))?;
            if !status.success() {
                return Err(format!("git pull failed for '{}'", dep_name));
            }
        } else {
            eprintln!("  {} — cloning {} ...", dep_name, dep_url);
            let status = Command::new("git")
                .args(["clone", "--depth", "1", dep_url, target.to_str().unwrap_or("")])
                .status()
                .map_err(|e| format!("git clone failed: {}", e))?;
            if !status.success() {
                return Err(format!("git clone failed for '{}'", dep_name));
            }
        }

        match read_manifest(&target) {
            Ok(pkg) => {
                eprintln!("  installed {} v{}", pkg.name, pkg.version);
                installed.push(pkg);
            }
            Err(_) => {
                eprintln!("  warning: no valid iris.toml in '{}', using directory name", dep_name);
                installed.push(Package {
                    name: dep_name.clone(),
                    version: "0.0.0".into(),
                    description: String::new(),
                    path: target,
                });
            }
        }
    }

    eprintln!("installed {} package(s)", installed.len());
    Ok(installed)
}

/// List all installed packages in `iris_packages/`.
pub fn list_packages() -> Vec<Package> {
    let pkgs_dir = match packages_dir() {
        Ok(d) => d,
        Err(_) => return vec![],
    };
    if !pkgs_dir.exists() {
        return vec![];
    }

    let mut packages = Vec::new();
    if let Ok(entries) = fs::read_dir(&pkgs_dir) {
        for entry in entries.flatten() {
            if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                let pkg_dir = entry.path();
                match read_manifest(&pkg_dir) {
                    Ok(pkg) => packages.push(pkg),
                    Err(_) => {
                        let name = entry.file_name().to_string_lossy().into_owned();
                        packages.push(Package {
                            name,
                            version: "0.0.0".into(),
                            description: String::new(),
                            path: pkg_dir,
                        });
                    }
                }
            }
        }
    }
    packages.sort_by(|a, b| a.name.cmp(&b.name));
    packages
}

/// Read `iris.toml` from a package directory.
pub fn read_manifest(pkg_dir: &Path) -> Result<Package, String> {
    let toml_path = pkg_dir.join("iris.toml");
    if !toml_path.exists() {
        return Err(format!("no iris.toml in {}", pkg_dir.display()));
    }
    let text = fs::read_to_string(&toml_path)
        .map_err(|e| format!("cannot read {}: {}", toml_path.display(), e))?;
    parse_manifest(&text, pkg_dir)
}

// ---------------------------------------------------------------------------
// Simple TOML-like parser (no external dependency)
// ---------------------------------------------------------------------------

fn parse_manifest(text: &str, pkg_dir: &Path) -> Result<Package, String> {
    let mut name = None;
    let mut version = None;
    let mut description = None;
    let mut in_package_section = false;

    for line in text.lines() {
        let trimmed = line.trim();

        // Skip comments and empty lines.
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        // Section headers.
        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            let section = &trimmed[1..trimmed.len() - 1].trim();
            in_package_section = *section == "package";
            continue;
        }

        if in_package_section {
            if let Some((key, value)) = parse_key_value(trimmed) {
                match key.as_str() {
                    "name" => name = Some(value),
                    "version" => version = Some(value),
                    "description" => description = Some(value),
                    _ => {}
                }
            }
        }
    }

    Ok(Package {
        name: name.unwrap_or_else(|| {
            pkg_dir
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown")
                .to_string()
        }),
        version: version.unwrap_or_else(|| "0.0.0".into()),
        description: description.unwrap_or_default(),
        path: pkg_dir.to_path_buf(),
    })
}

/// Parse the `[dependencies]` section of an `iris.toml`, returning name → url pairs.
fn parse_dependencies(text: &str) -> Result<Vec<(String, String)>, String> {
    let mut deps = Vec::new();
    let mut in_deps_section = false;

    for line in text.lines() {
        let trimmed = line.trim();

        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            let section = &trimmed[1..trimmed.len() - 1].trim();
            in_deps_section = *section == "dependencies";
            continue;
        }

        if in_deps_section {
            if let Some((key, value)) = parse_key_value(trimmed) {
                deps.push((key, value));
            }
        }
    }

    Ok(deps)
}

/// Parse `key = "value"` from a line. Returns `(key, value)`.
fn parse_key_value(line: &str) -> Option<(String, String)> {
    let eq_pos = line.find('=')?;
    let key = line[..eq_pos].trim().to_string();
    let val_part = line[eq_pos + 1..].trim();

    // Handle quoted strings: "value" or 'value'
    let value = if (val_part.starts_with('"') && val_part.ends_with('"'))
        || (val_part.starts_with('\'') && val_part.ends_with('\''))
    {
        val_part[1..val_part.len() - 1].to_string()
    } else {
        val_part.to_string()
    };

    Some((key, value))
}

/// Extract a package name from a Git URL.
/// `https://github.com/user/my-package.git` → `my-package`
/// `https://github.com/user/my-package` → `my-package`
fn extract_package_name(url: &str) -> Result<String, String> {
    let path = url.trim_end_matches('/');
    let path = path.trim_end_matches(".git");
    let name = path
        .rsplit('/')
        .next()
        .ok_or_else(|| format!("cannot extract package name from URL: {}", url))?;

    if name.is_empty() {
        return Err(format!("cannot extract package name from URL: {}", url));
    }

    // Sanitize: replace non-alphanumeric chars with underscores.
    let sanitized: String = name
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '_' || c == '-' { c } else { '_' })
        .collect();

    Ok(sanitized)
}

/// Resolve an iris_packages directory for a given base path.
/// Checks `base/iris_packages/<name>/` for `main.iris`, `lib.iris`, or `<name>.iris`.
pub fn resolve_package_path(base: &Path, name: &str) -> Option<PathBuf> {
    let pkg_dir = base.join(PACKAGES_DIR).join(name);
    if !pkg_dir.is_dir() {
        return None;
    }
    let lib = pkg_dir.join("lib.iris");
    if lib.exists() {
        return Some(lib);
    }
    let main = pkg_dir.join("main.iris");
    if main.exists() {
        return Some(main);
    }
    let named = pkg_dir.join(format!("{}.iris", name));
    if named.exists() {
        return Some(named);
    }
    // Fall back to the directory itself (FileCompiler will try lib.iris/main.iris).
    Some(pkg_dir)
}

/// Run the `iris install` command.
pub fn run_install(args: &[String]) -> Result<(), String> {
    if args.is_empty() {
        // `iris install` — install all deps from iris.toml
        install_all()?;
    } else {
        // `iris install <url>` — install from URL
        let url = &args[0];
        let pkg = install_from_url(url)?;
        eprintln!("installed {} v{} into {}/", pkg.name, pkg.version, PACKAGES_DIR);
    }
    Ok(())
}
