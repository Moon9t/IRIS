import * as vscode from 'vscode';
import * as child_process from 'child_process';
import * as fs from 'fs';
import * as path from 'path';

let statusBar: vscode.StatusBarItem;
let cachedVersionInfo: IrisVersionInfo | null = null;

export interface IrisVersionInfo {
    version: string | null;
    gitCommit: string | null;
    gitBranch: string | null;
    buildDate: string | null;
    target: string | null;
    rustc: string | null;
    fullOutput: string | null;
}

export function initStatusBar(context: vscode.ExtensionContext): vscode.StatusBarItem {
    statusBar = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 10);
    statusBar.command = 'iris.serverMenu';
    statusBar.tooltip = 'IRIS Language Server — Click for options';
    context.subscriptions.push(statusBar);
    updateStatusBar('starting');
    statusBar.show();
    return statusBar;
}

export function getIrisVersionInfo(executablePathFinder: () => string): IrisVersionInfo {
    if (cachedVersionInfo) { return cachedVersionInfo; }
    try {
        const exe = executablePathFinder();
        const out = child_process.execSync(`"${exe}" --version`, { timeout: 5000, encoding: 'utf8' });

        const versionMatch = out.match(/iris\s+(\d+\.\d+\.\d+)/);
        const commitMatch = out.match(/Git commit:\s*([0-9a-f]{7,40})/);
        const branchMatch = out.match(/Git branch:\s*(\S+)/);
        const dateMatch = out.match(/Build date:\s*(\S+)/);
        const targetMatch = out.match(/Target:\s*(\S+)/);
        const rustcMatch = out.match(/Built with:\s*(.+)/);

        cachedVersionInfo = {
            version: versionMatch ? versionMatch[1] : null,
            gitCommit: commitMatch ? commitMatch[1].substring(0, 9) : null,
            gitBranch: branchMatch ? branchMatch[1] : null,
            buildDate: dateMatch ? dateMatch[1] : null,
            target: targetMatch ? targetMatch[1] : null,
            rustc: rustcMatch ? rustcMatch[1].trim() : null,
            fullOutput: out,
        };
        return cachedVersionInfo;
    } catch {
        return { version: null, gitCommit: null, gitBranch: null, buildDate: null, target: null, rustc: null, fullOutput: null };
    }
}

export function clearVersionCache(): void {
    cachedVersionInfo = null;
}

export function updateStatusBar(state: 'starting' | 'running' | 'stopped' | 'error', executablePathFinder?: () => string): void {
    if (!statusBar) { return; }
    const icons: Record<string, string> = {
        starting: '$(loading~spin)',
        running:  '$(check)',
        stopped:  '$(circle-outline)',
        error:    '$(error)',
    };
    const colors: Record<string, string | undefined> = {
        starting: undefined,
        running:  undefined,
        stopped:  new vscode.ThemeColor('statusBarItem.warningBackground') as any,
        error:    new vscode.ThemeColor('statusBarItem.errorBackground') as any,
    };
    const info = executablePathFinder ? getIrisVersionInfo(executablePathFinder) : { version: null, gitCommit: null, gitBranch: null, buildDate: null, target: null, rustc: null };
    const label = info.version ? `IRIS v${info.version}` : 'IRIS';
    statusBar.text = `${icons[state]} ${label}`;
    statusBar.backgroundColor = colors[state] as any;

    const parts: string[] = [`IRIS Language Server: ${state}`];
    if (info.version) { parts.push(`Version: ${info.version}`); }
    if (info.gitCommit) { parts.push(`Commit: ${info.gitCommit}`); }
    if (info.gitBranch) { parts.push(`Branch: ${info.gitBranch}`); }
    if (info.buildDate) { parts.push(`Built: ${info.buildDate}`); }
    if (info.target) { parts.push(`Target: ${info.target}`); }
    if (info.rustc) { parts.push(`Rustc: ${info.rustc}`); }
    parts.push('Click for server options');
    statusBar.tooltip = parts.join('\n');
}

export async function showServerMenu(
    context: vscode.ExtensionContext,
    isLspRunning: boolean,
    actions: {
        restartLsp: () => Promise<void>;
        stopLsp: () => Promise<void>;
        startLsp: () => Promise<void>;
        showServerOutput: () => void;
        openRepl: () => void;
        showFullVersion: () => void;
    }
): Promise<void> {
    const items: vscode.QuickPickItem[] = [
        { label: '$(debug-restart) Restart Language Server', description: 'Restart the IRIS LSP server' },
        { label: isLspRunning ? '$(debug-stop) Stop Language Server' : '$(play) Start Language Server',
          description: isLspRunning ? 'Stop the IRIS LSP server' : 'Start the IRIS LSP server' },
        { label: '$(output) Show Server Output', description: 'Open the language server output channel' },
        { label: '$(terminal) Open REPL', description: 'Open an interactive IRIS session' },
        { label: '$(info) Show Version Info', description: 'Display full IRIS compiler version information' },
        { label: '$(gear) Open Settings', description: 'Configure IRIS extension settings' },
    ];

    const pick = await vscode.window.showQuickPick(items, { placeHolder: 'IRIS Language Server' });
    if (!pick) { return; }

    if (pick.label.includes('Restart')) {
        await actions.restartLsp();
    } else if (pick.label.includes('Stop')) {
        await actions.stopLsp();
    } else if (pick.label.includes('Start')) {
        await actions.startLsp();
    } else if (pick.label.includes('Output')) {
        actions.showServerOutput();
    } else if (pick.label.includes('REPL')) {
        actions.openRepl();
    } else if (pick.label.includes('Version')) {
        actions.showFullVersion();
    } else if (pick.label.includes('Settings')) {
        vscode.commands.executeCommand('workbench.action.openSettings', 'iris');
    }
}
