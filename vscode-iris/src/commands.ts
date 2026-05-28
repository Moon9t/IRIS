import * as vscode from 'vscode';
import * as child_process from 'child_process';
import * as path from 'path';
import * as fs from 'fs';
import { getIrisVersionInfo, clearVersionCache } from './statusBar';

export const runDiagCollection = vscode.languages.createDiagnosticCollection('iris-run');
let lastEmitContent = '';
let lastEmitLanguage = 'plaintext';

export class IrisEmitProvider implements vscode.TextDocumentContentProvider {
    provideTextDocumentContent(_uri: vscode.Uri): string {
        return lastEmitContent;
    }
}

export function runIrisFile(
    _context: vscode.ExtensionContext,
    subcommand: 'run' | 'build',
    getIrisExe: () => string,
    outputChannel: vscode.OutputChannel
): void {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showWarningMessage('No active .iris file.');
        return;
    }
    if (!editor.document.fileName.endsWith('.iris')) {
        vscode.window.showWarningMessage('Active file is not an .iris file.');
        return;
    }
    editor.document.save();
    runFileAtPath(editor.document.fileName, subcommand, getIrisExe, outputChannel);
}

export function runNamedFunction(
    uriStr: string,
    _fnName: string,
    getIrisExe: () => string,
    outputChannel: vscode.OutputChannel
): void {
    const filePath = vscode.Uri.parse(uriStr).fsPath;
    const doc = vscode.workspace.textDocuments.find(d => d.uri.fsPath === filePath);
    if (doc) { doc.save(); }
    runFileAtPath(filePath, 'run', getIrisExe, outputChannel);
}

export function runFileAtPath(
    filePath: string,
    subcommand: 'run' | 'build',
    getIrisExe: () => string,
    outputChannel: vscode.OutputChannel
): void {
    const exe = getIrisExe();
    const showTiming = vscode.workspace.getConfiguration('iris').get<boolean>('showTimingOnRun', true);
    outputChannel.clear();
    outputChannel.show(true);
    outputChannel.appendLine(`$ iris ${subcommand} "${path.basename(filePath)}"`);
    outputChannel.appendLine('');

    const args = subcommand === 'build'
        ? ['build', filePath, '-o', filePath.replace(/\.iris$/, '')]
        : ['run', filePath];

    const startTime = Date.now();

    const proc = child_process.spawn(`"${exe}"`, args, {
        shell: true,
        cwd: path.dirname(filePath),
    });

    proc.stdout.on('data', (data: Buffer) => {
        outputChannel.append(data.toString());
    });
    proc.stderr.on('data', (data: Buffer) => {
        const text = data.toString();
        outputChannel.append(text);
        parseAndShowErrors(text, filePath);
    });
    proc.on('close', (code: number | null) => {
        const elapsed = Date.now() - startTime;
        outputChannel.appendLine('');
        if (code === 0) {
            const timingStr = showTiming ? ` in ${elapsed}ms` : '';
            outputChannel.appendLine(`✓ Done (exit 0)${timingStr}`);
            runDiagCollection.delete(vscode.Uri.file(filePath));
        } else {
            outputChannel.appendLine(`✗ Failed (exit ${code})`);
            vscode.window.showErrorMessage(
                `IRIS ${subcommand} failed — check the output panel for details.`,
                'Show Output',
            ).then(choice => {
                if (choice === 'Show Output') { outputChannel.show(); }
            });
        }
    });
    proc.on('error', (err: Error) => {
        outputChannel.appendLine(`Error: ${err.message}`);
        vscode.window.showErrorMessage(
            `IRIS: Cannot run '${exe}'. Is it installed? Set iris.executablePath in settings.`,
            'Open Settings',
        ).then(choice => {
            if (choice === 'Open Settings') {
                vscode.commands.executeCommand('workbench.action.openSettings', 'iris.executablePath');
            }
        });
    });
}

export async function debugIrisFile(
    target: vscode.Uri | string | undefined,
    getIrisExe: () => string
): Promise<void> {
    let filePath: string | undefined;

    if (target instanceof vscode.Uri) {
        filePath = target.fsPath;
    } else if (typeof target === 'string') {
        filePath = vscode.Uri.parse(target).fsPath;
    } else {
        const editor = vscode.window.activeTextEditor;
        filePath = editor?.document.fileName;
    }

    if (!filePath || !filePath.endsWith('.iris')) {
        vscode.window.showWarningMessage('Open an .iris file first.');
        return;
    }

    const doc = vscode.workspace.textDocuments.find(d => d.uri.fsPath === filePath);
    if (doc) { await doc.save(); }

    const stopOnEntry = vscode.workspace
        .getConfiguration('iris')
        .get<boolean>('debug.stopOnEntry', false);
    const uri = vscode.Uri.file(filePath);

    await vscode.debug.startDebugging(vscode.workspace.getWorkspaceFolder(uri), {
        type: 'iris',
        request: 'launch',
        name: `Debug ${path.basename(filePath)}`,
        program: filePath,
        stopOnEntry,
    });
}

export function openRepl(
    _context: vscode.ExtensionContext,
    getIrisExe: () => string
): void {
    const exe = getIrisExe();
    const existing = vscode.window.terminals.find(t => t.name === 'IRIS REPL');
    if (existing) {
        existing.show();
        return;
    }
    const terminal = vscode.window.createTerminal({
        name: 'IRIS REPL',
        shellPath: exe,
        shellArgs: ['repl'],
    });
    terminal.show();
}

export async function showEmit(
    kind: 'ir' | 'llvm',
    getIrisExe: () => string,
    outputChannel: vscode.OutputChannel
): Promise<void> {
    const editor = vscode.window.activeTextEditor;
    if (!editor || !editor.document.fileName.endsWith('.iris')) {
        vscode.window.showWarningMessage('Open an .iris file first.');
        return;
    }
    const exe = getIrisExe();
    const filePath = editor.document.fileName;
    const flag = kind === 'ir' ? '--emit ir' : '--emit llvm';
    try {
        const out = child_process.execSync(`"${exe}" ${flag} "${filePath}"`, { encoding: 'utf8', timeout: 10000 });
        lastEmitContent = out;
        lastEmitLanguage = kind === 'llvm' ? 'llvm' : 'plaintext';
        const uri = vscode.Uri.parse(`iris-emit://output/${path.basename(filePath)}.${kind}`);
        const doc = await vscode.workspace.openTextDocument(uri);
        await vscode.window.showTextDocument(doc, vscode.ViewColumn.Beside, true);
    } catch (err: any) {
        outputChannel.appendLine(`iris ${flag} failed: ${err.message || err}`);
        outputChannel.show();
    }
}

export function showFullVersion(
    getIrisExe: () => string,
    outputChannel: vscode.OutputChannel
): void {
    clearVersionCache();
    const info = getIrisVersionInfo(getIrisExe);
    if (info.fullOutput) {
        outputChannel.clear();
        outputChannel.appendLine('=== IRIS Compiler Version Info ===');
        outputChannel.appendLine('');
        outputChannel.appendLine(info.fullOutput);
        outputChannel.show(true);
    } else {
        vscode.window.showWarningMessage(
            'Could not retrieve IRIS version info. Is the iris executable in your PATH?'
        );
    }
}

function parseAndShowErrors(stderr: string, filePath: string): void {
    const uri = vscode.Uri.file(filePath);
    const diags: vscode.Diagnostic[] = [];

    const richErrorPattern = /error(?:\[(\w+)\])?\s*:\s*(.+)/g;
    const locationPattern = /-->\s*[^:]+:(\d+):(\d+)/;

    let match: RegExpExecArray | null;
    richErrorPattern.lastIndex = 0;
    const lines = stderr.split('\n');

    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        richErrorPattern.lastIndex = 0;
        match = richErrorPattern.exec(line);
        if (match) {
            const errorCode = match[1] || undefined;
            const message = match[2].trim();
            let lineNum = 0;
            let colNum = 0;
            if (i + 1 < lines.length) {
                const locMatch = lines[i + 1].match(locationPattern);
                if (locMatch) {
                    lineNum = Math.max(0, parseInt(locMatch[1]) - 1);
                    colNum = Math.max(0, parseInt(locMatch[2]) - 1);
                }
            }
            const range = new vscode.Range(lineNum, colNum, lineNum, colNum + 20);
            const diag = new vscode.Diagnostic(range, message, vscode.DiagnosticSeverity.Error);
            if (errorCode) {
                diag.code = errorCode;
            }
            diag.source = 'iris';
            diags.push(diag);
        }
    }

    if (diags.length === 0) {
        const simplePattern = /error:\s*(.+)/gi;
        const lineMatch = /line (\d+)/i;
        for (const line of lines) {
            simplePattern.lastIndex = 0;
            const m = simplePattern.exec(line);
            if (m) {
                const lm = line.match(lineMatch);
                const lineNum = lm ? parseInt(lm[1]) - 1 : 0;
                const range = new vscode.Range(lineNum, 0, lineNum, 999);
                const diag = new vscode.Diagnostic(range, m[1].trim(), vscode.DiagnosticSeverity.Error);
                diag.source = 'iris';
                diags.push(diag);
            }
        }
    }

    if (diags.length > 0) {
        runDiagCollection.set(uri, diags);
    } else {
        runDiagCollection.delete(uri);
    }
}
