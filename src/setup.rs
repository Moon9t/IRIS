//! IRIS smart dependency installer.
//! Embeds the PowerShell setup script to automatically fetch and configure LLVM, MinGW,
//! Git, and MSVC build tools on Windows without requiring manual user configuration.

use std::fs;
use std::path::PathBuf;
use std::process::{Command, Stdio};

#[cfg(target_os = "windows")]
const SETUP_SCRIPT: &str = include_str!("../installer/windows/setup_dependencies.ps1");

/// Run the smart dependency setup downloader command.
pub fn run_setup_command() -> Result<(), String> {
    #[cfg(not(target_os = "windows"))]
    {
        println!("Note: Automatic toolchain setup is primarily designed for Windows hosts.");
        println!("- On macOS: Ensure Xcode Command Line Tools are installed (run `xcode-select --install`).");
        println!("- On Linux: Install LLVM/clang using your package manager (e.g. `sudo apt install clang` or `sudo dnf install clang`).");
        Ok(())
    }

    #[cfg(target_os = "windows")]
    {
        println!("Starting automatic IRIS compiler toolchain setup...");
        println!("This will detect and install missing compilation dependencies (LLVM/clang, MinGW sysroot, MSVC, Git).");
        println!("Installing local dependencies into ~/.iris/ directory...");
        println!();

        // 1. Create a temporary .ps1 file
        let temp_dir = std::env::temp_dir();
        let temp_script = temp_dir.join("iris_setup_dependencies.ps1");
        fs::write(&temp_script, SETUP_SCRIPT)
            .map_err(|e| format!("Failed to write temporary setup script: {}", e))?;

        // 2. Resolve the target installation directory (~/.iris)
        let home = std::env::var("USERPROFILE")
            .map(PathBuf::from)
            .map_err(|_| "Failed to locate USERPROFILE environment variable.".to_owned())?;
        let install_dir = home.join(".iris");

        // 3. Spawn powershell to execute the script
        let mut child = Command::new("powershell")
            .args(&[
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                &temp_script.to_string_lossy(),
                "-InstallDir",
                &install_dir.to_string_lossy(),
            ])
            .stdin(Stdio::inherit())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| format!("Failed to spawn PowerShell setup process: {}", e))?;

        let status = child
            .wait()
            .map_err(|e| format!("Failed to wait on setup process: {}", e))?;

        // 4. Cleanup the temporary script file
        let _ = fs::remove_file(&temp_script);

        if !status.success() {
            return Err(format!(
                "Setup process exited with non-zero status: {:?}",
                status.code()
            ));
        }

        println!();
        println!("==========================================================");
        println!("IRIS toolchain setup finished successfully!");
        println!("If you are running in an existing shell, please restart it");
        println!("or source your environment so PATH changes take effect.");
        println!("==========================================================");
        Ok(())
    }
}
