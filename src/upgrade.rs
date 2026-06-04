//! IRIS self-upgrade subcommand implementation.
//! Queries GitHub Release API for latest compiler version, downloads corresponding
//! platform executable, and performs safe, atomic self-replacement.

use serde::Deserialize;
use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

#[derive(Deserialize, Debug)]
struct GitHubRelease {
    tag_name: String,
    assets: Vec<GitHubAsset>,
}

#[derive(Deserialize, Debug)]
struct GitHubAsset {
    name: String,
    browser_download_url: String,
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct SemVer {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl SemVer {
    pub fn parse(s: &str) -> Option<Self> {
        let clean = s.trim().trim_start_matches('v');
        let parts: Vec<&str> = clean.split('.').collect();
        if parts.len() < 2 {
            return None;
        }
        let major = parts[0].parse().ok()?;
        let minor = parts[1].parse().ok()?;
        let patch = if parts.len() >= 3 {
            let clean_patch: String = parts[2]
                .chars()
                .take_while(|c| c.is_ascii_digit())
                .collect();
            clean_patch.parse().unwrap_or(0)
        } else {
            0
        };
        Some(SemVer {
            major,
            minor,
            patch,
        })
    }
}

fn fetch_latest_release_json() -> Result<String, String> {
    let output = if cfg!(target_os = "windows") {
        Command::new("powershell")
            .args([
                "-NoProfile",
                "-Command",
                "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-RestMethod -Uri 'https://api.github.com/repos/moon9t/iris/releases/latest' | ConvertTo-Json -Depth 10"
            ])
            .output()
    } else {
        Command::new("curl")
            .args([
                "-s",
                "-H",
                "User-Agent: iris-updater",
                "https://api.github.com/repos/moon9t/iris/releases/latest",
            ])
            .output()
    };

    let output = output.map_err(|e| format!("Failed to execute download command: {}", e))?;
    if !output.status.success() {
        return Err(format!(
            "Download process exited with error status: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    String::from_utf8(output.stdout).map_err(|e| format!("Invalid UTF-8 in output: {}", e))
}

fn download_and_extract(url: &str, target_path: &Path) -> Result<(), String> {
    let tmp_dir = env::temp_dir();
    let tmp_zip = tmp_dir.join("iris_update.zip");

    let download_status = if cfg!(target_os = "windows") {
        Command::new("powershell")
            .args([
                "-NoProfile",
                "-Command",
                &format!(
                    "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '{}' -OutFile '{}'",
                    url,
                    tmp_zip.to_string_lossy()
                )
            ])
            .status()
    } else {
        Command::new("curl")
            .args(["-sL", url, "-o", tmp_zip.to_str().unwrap_or("")])
            .status()
    };

    let status = download_status.map_err(|e| format!("Failed to run download command: {}", e))?;
    if !status.success() {
        return Err("Download failed".to_owned());
    }

    if url.ends_with(".zip") {
        let extract_dir = tmp_dir.join("iris_extracted");
        if extract_dir.exists() {
            let _ = fs::remove_dir_all(&extract_dir);
        }
        fs::create_dir_all(&extract_dir)
            .map_err(|e| format!("Failed to create extract dir: {}", e))?;

        let extract_status = if cfg!(target_os = "windows") {
            Command::new("powershell")
                .args([
                    "-NoProfile",
                    "-Command",
                    &format!(
                        "Expand-Archive -Path '{}' -DestinationPath '{}' -Force",
                        tmp_zip.to_string_lossy(),
                        extract_dir.to_string_lossy()
                    ),
                ])
                .status()
        } else {
            Command::new("unzip")
                .args([
                    "-o",
                    tmp_zip.to_str().unwrap_or(""),
                    "-d",
                    extract_dir.to_str().unwrap_or(""),
                ])
                .status()
        };

        let estatus = extract_status.map_err(|e| format!("Failed to extract archive: {}", e))?;
        if !estatus.success() {
            return Err("Extraction failed".to_owned());
        }

        let mut new_binary: Option<PathBuf> = None;
        let exe_name = if cfg!(target_os = "windows") {
            "iris.exe"
        } else {
            "iris"
        };

        fn find_exe(dir: &Path, name: &str) -> Option<PathBuf> {
            if let Ok(entries) = fs::read_dir(dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_dir() {
                        if let Some(p) = find_exe(&path, name) {
                            return Some(p);
                        }
                    } else if path.file_name().and_then(|f| f.to_str()) == Some(name) {
                        return Some(path);
                    }
                }
            }
            None
        }

        if let Some(exe_path) = find_exe(&extract_dir, exe_name) {
            new_binary = Some(exe_path);
        }

        let new_bin_path = new_binary
            .ok_or_else(|| "Could not find iris binary in the downloaded archive".to_owned())?;
        fs::copy(&new_bin_path, target_path)
            .map_err(|e| format!("Failed to copy new binary: {}", e))?;

        let _ = fs::remove_dir_all(&extract_dir);
    } else {
        fs::copy(&tmp_zip, target_path).map_err(|e| format!("Failed to copy new binary: {}", e))?;
    }

    let _ = fs::remove_file(&tmp_zip);
    Ok(())
}

fn perform_swap(new_bin_path: &Path, current_exe_path: &Path) -> Result<(), String> {
    let old_exe_path = current_exe_path.with_extension("exe.old");
    if old_exe_path.exists() {
        let _ = fs::remove_file(&old_exe_path);
    }

    fs::rename(current_exe_path, &old_exe_path)
        .map_err(|e| format!("Failed to rename running executable: {}", e))?;

    if let Err(e) = fs::rename(new_bin_path, current_exe_path) {
        let _ = fs::rename(&old_exe_path, current_exe_path);
        return Err(format!("Failed to place new executable: {}", e));
    }

    if cfg!(target_os = "windows") {
        let cmd = format!(
            "Start-Sleep -Seconds 1; Remove-Item -LiteralPath '{}' -Force",
            old_exe_path.to_string_lossy()
        );
        let _ = Command::new("powershell")
            .args(["-NoProfile", "-Command", &cmd])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn();
    } else {
        let _ = Command::new("sh")
            .args([
                "-c",
                &format!("sleep 1; rm -f '{}'", old_exe_path.to_string_lossy()),
            ])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn();
    }

    Ok(())
}

pub fn run_upgrade_command(check: bool, yes: bool, force: bool) -> Result<(), String> {
    let current_ver_str = env!("CARGO_PKG_VERSION");
    let current_semver = SemVer::parse(current_ver_str)
        .ok_or_else(|| format!("Invalid current version syntax: {}", current_ver_str))?;

    println!("Current IRIS version: v{}", current_ver_str);
    println!("Checking for latest release on GitHub...");

    let json = fetch_latest_release_json()?;
    let release: GitHubRelease = serde_json::from_str(&json)
        .map_err(|e| format!("Failed to parse GitHub release JSON: {}", e))?;

    let latest_semver = SemVer::parse(&release.tag_name)
        .ok_or_else(|| format!("Invalid remote version tag: {}", release.tag_name))?;

    if !force && latest_semver <= current_semver {
        println!("IRIS is already up to date: v{}", current_ver_str);
        return Ok(());
    }

    println!(
        "New version found: v{} (current: v{})",
        release.tag_name, current_ver_str
    );
    if check {
        return Ok(());
    }

    if !yes {
        print!("Do you want to upgrade? [y/N]: ");
        io::stdout().flush().map_err(|e| e.to_string())?;
        let mut response = String::new();
        io::stdin()
            .read_line(&mut response)
            .map_err(|e| e.to_string())?;
        let response = response.trim().to_lowercase();
        if !response.starts_with('y') {
            println!("Upgrade cancelled.");
            return Ok(());
        }
    }

    let platform_tag = if cfg!(target_os = "windows") {
        "windows"
    } else if cfg!(target_os = "macos") {
        "apple-darwin"
    } else {
        "unknown-linux-gnu"
    };

    let mut asset_url = None;
    for asset in &release.assets {
        let name_lower = asset.name.to_lowercase();
        if name_lower.contains(platform_tag)
            && (name_lower.contains(".zip")
                || name_lower.contains(".tar.gz")
                || name_lower.contains(".exe")
                || !name_lower.contains("setup"))
        {
            asset_url = Some(&asset.browser_download_url);
            break;
        }
    }

    if asset_url.is_none() {
        for asset in &release.assets {
            let name_lower = asset.name.to_lowercase();
            if name_lower.contains(".zip") || name_lower.contains(".tar.gz") {
                asset_url = Some(&asset.browser_download_url);
                break;
            }
        }
    }

    let download_url = asset_url
        .ok_or_else(|| "Could not find a suitable release asset for this platform".to_owned())?;
    println!("Downloading new binary from: {}", download_url);

    let current_exe =
        env::current_exe().map_err(|e| format!("Failed to get current executable path: {}", e))?;
    let tmp_bin = current_exe.with_extension("tmp_new");

    download_and_extract(download_url, &tmp_bin)?;

    println!("Applying update...");
    perform_swap(&tmp_bin, &current_exe)?;
    println!(
        "Upgrade complete! IRIS has been updated to v{}",
        release.tag_name
    );

    Ok(())
}
