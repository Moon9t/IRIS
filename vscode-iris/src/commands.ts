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

    const cfg = vscode.workspace.getConfiguration('iris');
    const extra: string[] = [];
    if (cfg.get<boolean>('sandbox', false)) {
        extra.push('--sandbox');
    }
    if (cfg.get<boolean>('noCache', false)) {
        extra.push('--no-cache');
    }

    let args: string[];
    if (subcommand === 'build') {
        // --target only affects llvm/binary outputs, so it is a build-only flag.
        const target = cfg.get<string>('target', '').trim();
        const targetArgs = target ? ['--target', target] : [];
        args = ['build', ...extra, ...targetArgs, filePath, '-o', filePath.replace(/\.iris$/, '')];
    } else {
        args = ['run', ...extra, filePath];
    }

    const startTime = Date.now();

    // No shell: arguments are passed through verbatim, so paths containing
    // spaces or shell metacharacters need no quoting and cannot be re-parsed.
    const proc = child_process.spawn(exe, args, {
        shell: false,
        windowsHide: true,
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

/** Emit kinds worth showing in the editor, i.e. the ones producing text. */
export const TEXT_EMIT_KINDS = [
    'ir',
    'llvm',
    'llvm-complete',
    'cuda',
    'cuda-ptx',
    'simd',
    'graph',
    'onnx',
    'tensorrt',
    'pgo-instrument',
    'pgo-optimize',
] as const;

export type EmitKind = (typeof TEXT_EMIT_KINDS)[number];

/** Syntax to display each emit kind with. */
function emitLanguageId(kind: EmitKind): string {
    switch (kind) {
        case 'llvm':
        case 'llvm-complete':
        case 'cuda':
        case 'simd':
        case 'pgo-instrument':
        case 'pgo-optimize':
            return 'llvm';
        case 'onnx':
        case 'graph':
            return 'json';
        default:
            return 'plaintext';
    }
}

export async function showEmit(
    kind: EmitKind,
    getIrisExe: () => string,
    outputChannel: vscode.OutputChannel
): Promise<void> {
    const editor = vscode.window.activeTextEditor;
    if (!editor || !editor.document.fileName.endsWith('.iris')) {
        vscode.window.showWarningMessage('Open an .iris file first.');
        return;
    }
    await editor.document.save();
    const exe = getIrisExe();
    const filePath = editor.document.fileName;
    try {
        const out = child_process.execFileSync(exe, ['--emit', kind, filePath], {
            encoding: 'utf8',
            timeout: 60000,
            windowsHide: true,
        });
        lastEmitContent = out;
        lastEmitLanguage = emitLanguageId(kind);
        const uri = vscode.Uri.parse(`iris-emit://output/${path.basename(filePath)}.${kind}`);
        const doc = await vscode.workspace.openTextDocument(uri);
        await vscode.languages.setTextDocumentLanguage(doc, lastEmitLanguage);
        await vscode.window.showTextDocument(doc, vscode.ViewColumn.Beside, true);
    } catch (err: any) {
        // The compiler writes diagnostics to stderr; surface them rather than
        // just the spawn error, which on its own says nothing useful.
        const stderr = err?.stderr?.toString?.() ?? '';
        outputChannel.appendLine(`iris --emit ${kind} failed: ${err.message || err}`);
        if (stderr) {
            outputChannel.appendLine(stderr);
            parseAndShowErrors(stderr, filePath);
        }
        outputChannel.show();
    }
}

/** Ask which emit kind to show, then show it. */
export async function showEmitPick(
    getIrisExe: () => string,
    outputChannel: vscode.OutputChannel
): Promise<void> {
    const picked = await vscode.window.showQuickPick([...TEXT_EMIT_KINDS], {
        title: 'IRIS: Show Compiler Output',
        placeHolder: 'Select an --emit kind',
    });
    if (picked) {
        await showEmit(picked as EmitKind, getIrisExe, outputChannel);
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

/** Run an iris subcommand in a dedicated terminal, reusing one per name. */
function runInTerminal(name: string, exe: string, args: string[], cwd?: string): void {
    const existing = vscode.window.terminals.find(t => t.name === name);
    if (existing) {
        existing.dispose();
    }
    const terminal = vscode.window.createTerminal({ name, shellPath: exe, shellArgs: args, cwd });
    terminal.show();
}

/** The active .iris file, saved, or undefined with a warning already shown. */
async function activeIrisFile(): Promise<string | undefined> {
    const editor = vscode.window.activeTextEditor;
    if (!editor || !editor.document.fileName.endsWith('.iris')) {
        vscode.window.showWarningMessage('Open an .iris file first.');
        return undefined;
    }
    await editor.document.save();
    return editor.document.fileName;
}

/**
 * Explain an error code via `iris explain`.
 *
 * Defaults to a code taken from a diagnostic at the cursor, since that is
 * almost always the one being asked about.
 */
export async function explainErrorCode(
    getIrisExe: () => string,
    outputChannel: vscode.OutputChannel
): Promise<void> {
    const editor = vscode.window.activeTextEditor;
    let seed = '';
    if (editor) {
        const pos = editor.selection.active;
        const hit = vscode.languages
            .getDiagnostics(editor.document.uri)
            .find(d => d.range.contains(pos) && typeof d.code === 'string');
        if (hit) {
            seed = String(hit.code);
        }
    }

    const code = await vscode.window.showInputBox({
        title: 'IRIS: Explain Error Code',
        prompt: 'Error code to explain (e.g. E0100)',
        value: seed,
        validateInput: v => (/^E\d{4}$/i.test(v.trim()) ? null : 'Expected a code like E0100'),
    });
    if (!code) {
        return;
    }

    const exe = getIrisExe();
    try {
        const out = child_process.execFileSync(exe, ['explain', code.trim().toUpperCase()], {
            encoding: 'utf8',
            timeout: 10000,
            windowsHide: true,
        });
        outputChannel.clear();
        outputChannel.appendLine(out);
        outputChannel.show(true);
    } catch (err: any) {
        outputChannel.appendLine(`iris explain ${code} failed: ${err.message || err}`);
        outputChannel.show();
    }
}

/** Verify formatting without modifying, via `iris fmt --check`. */
export async function checkFormatting(
    getIrisExe: () => string,
    outputChannel: vscode.OutputChannel
): Promise<void> {
    const filePath = await activeIrisFile();
    if (!filePath) {
        return;
    }
    const exe = getIrisExe();
    try {
        const out = child_process.execFileSync(exe, ['fmt', '--check', filePath], {
            encoding: 'utf8',
            timeout: 30000,
            windowsHide: true,
        });
        outputChannel.appendLine(out.trim() || `✓ ${path.basename(filePath)} is correctly formatted.`);
        vscode.window.showInformationMessage(`IRIS: ${path.basename(filePath)} is correctly formatted.`);
    } catch (err: any) {
        // A non-zero exit is the documented signal for "needs formatting".
        const detail = (err?.stdout?.toString?.() ?? '') + (err?.stderr?.toString?.() ?? '');
        outputChannel.appendLine(detail.trim() || `${path.basename(filePath)} needs formatting.`);
        outputChannel.show(true);
        const choice = await vscode.window.showWarningMessage(
            `IRIS: ${path.basename(filePath)} is not correctly formatted.`,
            'Format Now',
        );
        if (choice === 'Format Now') {
            await vscode.commands.executeCommand('editor.action.formatDocument');
        }
    }
}

/** Run every `test_` function in the active file. */
export async function runFileTests(getIrisExe: () => string): Promise<void> {
    const filePath = await activeIrisFile();
    if (!filePath) {
        return;
    }
    runInTerminal('IRIS Tests', getIrisExe(), ['test', filePath], path.dirname(filePath));
}

/** Run benchmarks in the active file. */
export async function runBenchmarks(getIrisExe: () => string): Promise<void> {
    const filePath = await activeIrisFile();
    if (!filePath) {
        return;
    }
    runInTerminal('IRIS Bench', getIrisExe(), ['bench', filePath], path.dirname(filePath));
}

/** Generate HTML documentation from doc comments. */
export async function generateDocs(getIrisExe: () => string): Promise<void> {
    const filePath = await activeIrisFile();
    if (!filePath) {
        return;
    }
    runInTerminal('IRIS Docs', getIrisExe(), ['docs', filePath], path.dirname(filePath));
}

/** Drive the package manager: `iris pkg <cmd>`. */
export async function pkgCommand(getIrisExe: () => string): Promise<void> {
    const sub = await vscode.window.showQuickPick(
        ['init', 'add', 'remove', 'install', 'list', 'build', 'run'],
        { title: 'IRIS: Package Manager', placeHolder: 'Select a pkg subcommand' },
    );
    if (!sub) {
        return;
    }

    const args = ['pkg', sub];
    if (sub === 'add' || sub === 'remove') {
        const name = await vscode.window.showInputBox({
            title: `IRIS: pkg ${sub}`,
            prompt: sub === 'add' ? 'Package name or Git URL' : 'Package name to remove',
        });
        if (!name) {
            return;
        }
        args.push(name);
    }

    const cwd = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
    runInTerminal('IRIS Package Manager', getIrisExe(), args, cwd);
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
