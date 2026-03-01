; iris.iss — Inno Setup script for IRIS Language Windows installer
; Requires Inno Setup 6+ from https://jrsoftware.org/isinfo.php
; Compile with: iscc iris.iss

#define AppName "IRIS Language"
#define AppVersion "0.1.0"
#define AppPublisher "IRIS Language Project"
#define AppURL "https://github.com/iris-lang/iris"
#define AppExeName "iris.exe"

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
; LicenseFile=LICENSE.txt  ; uncomment once LICENSE.txt is added to installer/
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

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "addtopath"; Description: "Add IRIS to the system PATH (recommended)"; GroupDescription: "Additional options:"; Flags: checked
Name: "installvscode"; Description: "Install VSCode extension (if VSCode is detected)"; GroupDescription: "Additional options:"; Flags: checked unchecked

[Files]
Source: "iris.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "iris-lang-*.vsix"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "icon.png"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "README.md"; DestDir: "{app}"; Flags: ignoreversion isreadme

[Icons]
Name: "{group}\IRIS REPL"; Filename: "{app}\{#AppExeName}"; Parameters: "repl"; WorkingDir: "{userdocs}"
Name: "{group}\IRIS Documentation"; Filename: "{app}\README.md"
Name: "{group}\Uninstall IRIS"; Filename: "{uninstallexe}"

[Registry]
Root: HKCU; Subkey: "Software\IRIS"; ValueType: string; ValueName: "InstallDir"; ValueData: "{app}"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\IRIS"; ValueType: string; ValueName: "Version"; ValueData: "{#AppVersion}"

[Code]
procedure EnvAddPath(InstallPath: string);
var
  Paths: string;
begin
  if not RegQueryStringValue(HKCU, 'Environment', 'Path', Paths) then
    Paths := '';
  if Pos(';' + Uppercase(InstallPath) + ';', ';' + Uppercase(Paths) + ';') > 0 then
    exit;
  Paths := Paths + ';' + InstallPath;
  RegWriteExpandStringValue(HKCU, 'Environment', 'Path', Paths);
end;

procedure EnvRemovePath(InstallPath: string);
var
  Paths: string;
  P: Integer;
begin
  if not RegQueryStringValue(HKCU, 'Environment', 'Path', Paths) then
    exit;
  P := Pos(';' + Uppercase(InstallPath) + ';', ';' + Uppercase(Paths) + ';');
  if P = 0 then exit;
  Delete(Paths, P - 1, Length(InstallPath) + 1);
  RegWriteExpandStringValue(HKCU, 'Environment', 'Path', Paths);
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  VsCodePath, VsixPath, ResultCode: Integer;
  VsCodeExe, VsixFile: string;
begin
  if CurStep = ssPostInstall then
  begin
    // Add to PATH if task selected
    if WizardIsTaskSelected('addtopath') then
      EnvAddPath(ExpandConstant('{app}'));

    // Install VSCode extension if task selected
    if WizardIsTaskSelected('installvscode') then
    begin
      VsCodeExe := ExpandConstant('{localappdata}\Programs\Microsoft VS Code\bin\code.cmd');
      if FileExists(VsCodeExe) then
      begin
        VsixFile := FindFirst(ExpandConstant('{app}\iris-lang-*.vsix'), faAnyFile);
        if VsixFile <> '' then
          Exec(VsCodeExe, '--install-extension "' + ExpandConstant('{app}\') + VsixFile + '"', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
      end;
    end;
  end;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usPostUninstall then
    EnvRemovePath(ExpandConstant('{app}'));
end;
