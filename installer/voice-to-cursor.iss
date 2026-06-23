; Inno Setup script for Voice-to-Cursor AI Dictation System
; Generated 2026-06-23

#define MyAppName "Voice-to-Cursor"
#define MyAppVersion "0.6.0"
#define MyAppPublisher "Voice-to-Cursor Project"
#define MyAppURL "https://github.com/yourusername/voice-to-cursor"
#define MyAppExeName "voice-to-cursor.exe"
#define MyAppConsoleExeName "voice-to-cursor-console.exe"
#define SourceDir "..\dist\voice-to-cursor"

[Setup]
AppId={{F4A7B2D1-8E3C-4F8A-9B1E-6D2C3A4B5E7F}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DisableProgramGroupPage=yes
LicenseFile=..\README.md
; Remove the following line to run in administrative install mode (install for all users.)
PrivilegesRequired=lowest
OutputDir=..\dist\installer
OutputBaseFilename=Voice-to-Cursor-{#MyAppVersion}-Setup
SetupIconFile=
Compression=lzma2
SolidCompression=yes
WizardStyle=modern

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "quicklaunchicon"; Description: "{cm:CreateQuickLaunchIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked; OnlyBelowVersion: 6.1; Check: not IsAdminInstallMode

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\.env.template"; DestDir: "{app}"; DestName: ".env.template"; Flags: ignoreversion
; NOTE: Don't use "Flags: ignoreversion" on any shared system files

[Dirs]
Name: "{app}\data"; Permissions: users-modify
Name: "{app}\data\vocab"; Permissions: users-modify
Name: "{app}\models"; Permissions: users-modify
Name: "{app}\data\logs"; Permissions: users-modify

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"
Name: "{autoprograms}\{#MyAppName} (Dry Run)"; Filename: "{app}\{#MyAppConsoleExeName}"; Parameters: "--dry-run"; WorkingDir: "{app}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon; WorkingDir: "{app}"

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\data\logs"
Type: filesandordirs; Name: "{app}\_internal"
