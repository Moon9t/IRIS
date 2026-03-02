; iris.iss — Inno Setup 6 script for IRIS Language Windows installer
; Produces a single self-extracting EXE with all dependencies bundled:
;   - iris.exe  (the compiler / REPL / LSP / DAP)
;   - clang.exe + ld.lld.exe  (LLVM 17)
;   - MinGW ucrt64 sysroot (headers + static libraries)
;   - VSCode extension (.vsix)
;
; Build with:  "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer\iris.iss
;   or via:    powershell -ExecutionPolicy Bypass -File installer\build_installer.ps1

#define AppName      "IRIS Language"
#define AppVersion   "0.2.0"
#define AppPublisher "IRIS Language Project"
#define AppURL       "https://github.com/iris-lang/iris"
#define AppExeName   "iris.exe"

; ---------------------------------------------------------------------------
; The build script stages everything into installer\_stage before ISCC runs.
; Source paths below are relative to the staging directory.
; ---------------------------------------------------------------------------
#define StageDir     "_stage"

[Setup]
AppId={{A7B3C2D1-E4F5-4A6B-8C9D-0E1F2A3B4C5D}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}/issues
AppUpdatesURL={#AppURL}/releases
DefaultDirName={userpf}\IRIS
DefaultGroupName={#AppName}
AllowNoIcons=yes
OutputDir=dist
OutputBaseFilename=IRIS-{#AppVersion}-windows-x64-setup
SetupIconFile=icon.ico
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ChangesEnvironment=yes
UninstallDisplayIcon={app}\{#AppExeName}
UninstallDisplayName={#AppName}
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Types]
Name: "full";    Description: "Full installation (compiler + toolchain + VSCode extension)"
Name: "compact"; Description: "Compiler only (requires clang + MinGW already installed)"
Name: "custom";  Description: "Custom installation"; Flags: iscustom

[Components]
Name: "main";       Description: "IRIS Compiler (iris.exe)";                  Types: full compact custom; Flags: fixed
Name: "llvm";       Description: "LLVM/clang + lld (for native compilation)"; Types: full
Name: "sysroot";    Description: "MinGW sysroot (headers + libs)";            Types: full
Name: "vscode";     Description: "VSCode extension (.vsix)";                  Types: full

[Tasks]
Name: "addtopath";     Description: "Add IRIS to the user PATH";                         GroupDescription: "Environment:"
Name: "addllvmpath";   Description: "Add LLVM to the user PATH";                         GroupDescription: "Environment:"; Components: llvm
Name: "installvscode"; Description: "Install VSCode extension (if VSCode is detected)";   GroupDescription: "Extras:";      Components: vscode

; ---------------------------------------------------------------------------
; Files
; ---------------------------------------------------------------------------
[Files]
; Core compiler
Source: "{#StageDir}\iris.exe";               DestDir: "{app}";                              Flags: ignoreversion;             Components: main

; Documentation / readme
Source: "{#StageDir}\README.md";              DestDir: "{app}";                              Flags: ignoreversion isreadme;    Components: main
Source: "{#StageDir}\icon.png";               DestDir: "{app}";                              Flags: ignoreversion skipifsourcedoesntexist; Components: main

; LLVM toolchain (clang + lld)
Source: "{#StageDir}\toolchain\llvm\bin\*";   DestDir: "{app}\toolchain\llvm\bin";           Flags: ignoreversion recursesubdirs; Components: llvm

; MinGW sysroot — libraries
Source: "{#StageDir}\toolchain\ucrt64\lib\*"; DestDir: "{app}\toolchain\ucrt64\lib";         Flags: ignoreversion recursesubdirs; Components: sysroot

; MinGW sysroot — headers
Source: "{#StageDir}\toolchain\ucrt64\include\*"; DestDir: "{app}\toolchain\ucrt64\include"; Flags: ignoreversion recursesubdirs; Components: sysroot

; VSCode extension
Source: "{#StageDir}\iris-lang-*.vsix";       DestDir: "{app}";                              Flags: ignoreversion skipifsourcedoesntexist; Components: vscode

; ---------------------------------------------------------------------------
; Shortcuts
; ---------------------------------------------------------------------------
[Icons]
Name: "{group}\IRIS REPL";          Filename: "{app}\{#AppExeName}"; Parameters: "repl"; WorkingDir: "{userdocs}"
Name: "{group}\IRIS Documentation"; Filename: "{app}\README.md"
Name: "{group}\Uninstall IRIS";     Filename: "{uninstallexe}"

; ---------------------------------------------------------------------------
; Registry
; ---------------------------------------------------------------------------
[Registry]
Root: HKCU; Subkey: "Software\IRIS"; ValueType: string; ValueName: "InstallDir"; ValueData: "{app}";           Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\IRIS"; ValueType: string; ValueName: "Version";    ValueData: "{#AppVersion}"

; ---------------------------------------------------------------------------
; Pascal Script — PATH management + VSCode extension install
; ---------------------------------------------------------------------------
[Code]

{ ---- PATH helpers ---- }

procedure EnvAddPath(InstallPath: string);
var
  Paths: string;
begin
  if not RegQueryStringValue(HKCU, 'Environment', 'Path', Paths) then
    Paths := '';
  if Pos(';' + Uppercase(InstallPath) + ';', ';' + Uppercase(Paths) + ';') > 0 then
    exit;
  if (Paths <> '') and (Paths[Length(Paths)] <> ';') then
    Paths := Paths + ';';
  Paths := Paths + InstallPath;
  RegWriteExpandStringValue(HKCU, 'Environment', 'Path', Paths);
end;

procedure EnvRemovePath(InstallPath: string);
var
  Paths: string;
  P: Integer;
begin
  if not RegQueryStringValue(HKCU, 'Environment', 'Path', Paths) then
    exit;
  P := Pos(';' + Uppercase(InstallPath), ';' + Uppercase(Paths));
  if P = 0 then exit;
  Delete(Paths, P, Length(InstallPath) + 1);
  RegWriteExpandStringValue(HKCU, 'Environment', 'Path', Paths);
end;

{ ---- Post-install actions ---- }

procedure CurStepChanged(CurStep: TSetupStep);
var
  ResultCode: Integer;
  VsCodeExe, VsixFile: string;
  FindRec: TFindRec;
begin
  if CurStep = ssPostInstall then
  begin
    { Add IRIS to PATH }
    if WizardIsTaskSelected('addtopath') then
      EnvAddPath(ExpandConstant('{app}'));

    { Add LLVM to PATH — use the install-local copy so the user does not
      need a separate LLVM installation in PATH. }
    if WizardIsTaskSelected('addllvmpath') then
      EnvAddPath(ExpandConstant('{app}\toolchain\llvm\bin'));

    { Install VSCode extension }
    if WizardIsTaskSelected('installvscode') then
    begin
      VsCodeExe := ExpandConstant('{localappdata}\Programs\Microsoft VS Code\bin\code.cmd');
      if not FileExists(VsCodeExe) then
        VsCodeExe := ExpandConstant('{pf}\Microsoft VS Code\bin\code.cmd');

      if FileExists(VsCodeExe) then
      begin
        if FindFirst(ExpandConstant('{app}\iris-lang-*.vsix'), FindRec) then
        begin
          VsixFile := ExpandConstant('{app}\') + FindRec.Name;
          Exec(VsCodeExe, '--install-extension "' + VsixFile + '" --force',
               '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
          FindClose(FindRec);
        end;
      end;
    end;
  end;
end;

{ ---- Uninstall cleanup ---- }

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usPostUninstall then
  begin
    EnvRemovePath(ExpandConstant('{app}'));
    EnvRemovePath(ExpandConstant('{app}\toolchain\llvm\bin'));
  end;
end;
