import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import * as child_process from 'child_process';
import {
    LanguageClient,
    LanguageClientOptions,
    ServerOptions,
    TransportKind,
    State,
} from 'vscode-languageclient/node';

let client: LanguageClient | undefined;
let statusBar: vscode.StatusBarItem;
let outputChannel: vscode.OutputChannel;
let serverOutputChannel: vscode.OutputChannel;

// ---------------------------------------------------------------------------
// Activation
// ---------------------------------------------------------------------------

export async function activate(context: vscode.ExtensionContext): Promise<void> {
    outputChannel = vscode.window.createOutputChannel('IRIS');
    serverOutputChannel = vscode.window.createOutputChannel('IRIS Language Server');
    context.subscriptions.push(outputChannel, serverOutputChannel);

    // Status bar — click opens a pick menu with server actions
    statusBar = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 10);
    statusBar.command = 'iris.serverMenu';
    statusBar.tooltip = 'IRIS Language Server — Click for options';
    updateStatusBar('starting');
    statusBar.show();
    context.subscriptions.push(statusBar);

    // Commands
    context.subscriptions.push(
        vscode.commands.registerCommand('iris.runFile',    () => runIrisFile(context, 'run')),
        vscode.commands.registerCommand('iris.buildFile',  () => runIrisFile(context, 'build')),
        vscode.commands.registerCommand('iris.openRepl',   () => openRepl(context)),
        vscode.commands.registerCommand('iris.restartLsp', () => restartLsp(context)),
        vscode.commands.registerCommand('iris.stopLsp',    () => stopLsp()),
        vscode.commands.registerCommand('iris.showIR',     () => showEmit('ir')),
        vscode.commands.registerCommand('iris.showLLVM',   () => showEmit('llvm')),
        vscode.commands.registerCommand('iris.runFunction', (uri: string, fnName: string) =>
            runNamedFunction(uri, fnName)),
        vscode.commands.registerCommand('iris.serverMenu', () => showServerMenu(context)),
    );

    // Code lens provider — inline ▷ Run / ⬡ Debug buttons on zero-arg functions
    context.subscriptions.push(
        vscode.languages.registerCodeLensProvider(
            { scheme: 'file', language: 'iris' },
            new IrisCodeLensProvider(),
        ),
    );

    // Virtual document provider for IR/LLVM output
    context.subscriptions.push(
        vscode.workspace.registerTextDocumentContentProvider('iris-emit', new IrisEmitProvider()),
    );

    // Debug adapter
    context.subscriptions.push(
        vscode.debug.registerDebugAdapterDescriptorFactory('iris', new IrisDebugAdapterFactory()),
    );

    // Format on save
    context.subscriptions.push(
        vscode.workspace.onWillSaveTextDocument(e => {
            const cfg = vscode.workspace.getConfiguration('iris');
            if (cfg.get<boolean>('formatOnSave', true) && e.document.languageId === 'iris') {
                e.waitUntil(vscode.commands.executeCommand('editor.action.formatDocument'));
            }
        }),
    );

    // Respond to inlayHint setting changes by toggling the editor setting
    context.subscriptions.push(
        vscode.workspace.onDidChangeConfiguration(e => {
            if (e.affectsConfiguration('iris.inlayHints.enabled')) {
                const enabled = vscode.workspace.getConfiguration('iris').get<boolean>('inlayHints.enabled', true);
                if (!enabled && client) {
                    // Clear inlay hints by restarting the client (simplest approach)
                    // The server will not send hints when the setting is off
                }
            }
        }),
    );

    // Start LSP
    await startLspClient(context);
}

// ---------------------------------------------------------------------------
// Status bar helpers
// ---------------------------------------------------------------------------

function updateStatusBar(state: 'starting' | 'running' | 'stopped' | 'error'): void {
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
    const version = getIrisVersion();
    const label = version ? `IRIS v${version}` : 'IRIS';
    statusBar.text = `${icons[state]} ${label}`;
    statusBar.backgroundColor = colors[state] as any;
    statusBar.tooltip = `IRIS Language Server: ${state}\nClick for server options`;
}

function getIrisVersion(): string | null {
    try {
        const exe = findIrisExe();
        const out = child_process.execSync(`"${exe}" --version`, { timeout: 2000, encoding: 'utf8' });
        const m = out.match(/\d+\.\d+\.\d+/);
        return m ? m[0] : null;
    } catch {
        return null;
    }
}

// ---------------------------------------------------------------------------
// Server action menu (like rust-analyzer status bar click)
// ---------------------------------------------------------------------------

async function showServerMenu(context: vscode.ExtensionContext): Promise<void> {
    const isRunning = client !== undefined;
    const items: vscode.QuickPickItem[] = [
        { label: '$(debug-restart) Restart Language Server', description: 'Restart the IRIS LSP server' },
        { label: isRunning ? '$(debug-stop) Stop Language Server' : '$(play) Start Language Server',
          description: isRunning ? 'Stop the IRIS LSP server' : 'Start the IRIS LSP server' },
        { label: '$(output) Show Server Output', description: 'Open the language server output channel' },
        { label: '$(terminal) Open REPL', description: 'Open an interactive IRIS session' },
        { label: '$(gear) Open Settings', description: 'Configure IRIS extension settings' },
    ];

    const pick = await vscode.window.showQuickPick(items, { placeHolder: 'IRIS Language Server' });
    if (!pick) { return; }

    if (pick.label.includes('Restart')) {
        await restartLsp(context);
    } else if (pick.label.includes('Stop')) {
        await stopLsp();
    } else if (pick.label.includes('Start')) {
        await startLspClient(context);
    } else if (pick.label.includes('Output')) {
        serverOutputChannel.show();
    } else if (pick.label.includes('REPL')) {
        openRepl(context);
    } else if (pick.label.includes('Settings')) {
        vscode.commands.executeCommand('workbench.action.openSettings', 'iris');
    }
}

// ---------------------------------------------------------------------------
// Executable detection
// ---------------------------------------------------------------------------

function findIrisExe(): string {
    const cfg = vscode.workspace.getConfiguration('iris').get<string>('executablePath', '');
    if (cfg && fs.existsSync(cfg)) {
        return cfg;
    }
    // Common Windows install locations
    const candidates = [
        path.join(process.env.USERPROFILE || '', '.cargo', 'bin', 'iris.exe'),
        path.join(process.env.USERPROFILE || '', '.iris', 'bin', 'iris.exe'),
        'C:\\Program Files\\IRIS\\iris.exe',
        'C:\\Users\\' + (process.env.USERNAME || '') + '\\.cargo\\bin\\iris.exe',
        // Already on PATH
        'iris',
    ];
    for (const c of candidates) {
        if (c === 'iris') { return c; }
        if (fs.existsSync(c)) { return c; }
    }
    return 'iris';
}

function getIrisExe(): string { return findIrisExe(); }

// ---------------------------------------------------------------------------
// Language Server Client
// ---------------------------------------------------------------------------

async function startLspClient(context: vscode.ExtensionContext): Promise<void> {
    const exe = getIrisExe();
    const serverOptions: ServerOptions = {
        command: exe,
        args: ['lsp'],
        transport: TransportKind.stdio,
    };
    const clientOptions: LanguageClientOptions = {
        documentSelector: [{ scheme: 'file', language: 'iris' }],
        synchronize: {
            fileEvents: vscode.workspace.createFileSystemWatcher('**/*.iris'),
        },
        outputChannel: serverOutputChannel,
        initializationOptions: {
            inlayHintsEnabled: vscode.workspace.getConfiguration('iris').get<boolean>('inlayHints.enabled', true),
            inlayHintsTypeHints: vscode.workspace.getConfiguration('iris').get<boolean>('inlayHints.typeHints', true),
        },
    };

    client = new LanguageClient('iris', 'IRIS Language Server', serverOptions, clientOptions);

    client.onDidChangeState(e => {
        if (e.newState === State.Running) {
            updateStatusBar('running');
        } else if (e.newState === State.Stopped) {
            updateStatusBar('stopped');
        } else {
            updateStatusBar('starting');
        }
    });

    try {
        await client.start();
        context.subscriptions.push(client);
    } catch (err) {
        updateStatusBar('error');
        const choice = await vscode.window.showErrorMessage(
            `IRIS: Could not start language server using '${exe}'.`,
            'Open Settings',
            'Retry',
        );
        if (choice === 'Open Settings') {
            vscode.commands.executeCommand('workbench.action.openSettings', 'iris.executablePath');
        } else if (choice === 'Retry') {
            await startLspClient(context);
        }
    }
}

async function restartLsp(context: vscode.ExtensionContext): Promise<void> {
    if (client) {
        updateStatusBar('starting');
        try {
            await client.stop();
            client.dispose();
        } catch { /* ignore stop errors */ }
        client = undefined;
    }
    await startLspClient(context);
    vscode.window.showInformationMessage('IRIS: Language server restarted.');
}

async function stopLsp(): Promise<void> {
    if (client) {
        try {
            await client.stop();
            client.dispose();
        } catch { /* ignore stop errors */ }
        client = undefined;
        updateStatusBar('stopped');
        vscode.window.showInformationMessage('IRIS: Language server stopped.');
    } else {
        vscode.window.showInformationMessage('IRIS: Language server is not running.');
    }
}

// ---------------------------------------------------------------------------
// Run / Build
// ---------------------------------------------------------------------------

function runIrisFile(_context: vscode.ExtensionContext, subcommand: 'run' | 'build'): void {
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
    runFileAtPath(editor.document.fileName, subcommand);
}

function runNamedFunction(uriStr: string, _fnName: string): void {
    const filePath = vscode.Uri.parse(uriStr).fsPath;
    // Save the document first
    const doc = vscode.workspace.textDocuments.find(d => d.uri.fsPath === filePath);
    if (doc) { doc.save(); }
    runFileAtPath(filePath, 'run');
}

function runFileAtPath(filePath: string, subcommand: 'run' | 'build'): void {
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
            // Clear run diagnostics on success
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

// ---------------------------------------------------------------------------
// Diagnostics from run output
// ---------------------------------------------------------------------------

const runDiagCollection = vscode.languages.createDiagnosticCollection('iris-run');

function parseAndShowErrors(stderr: string, filePath: string): void {
    const uri = vscode.Uri.file(filePath);
    const diags: vscode.Diagnostic[] = [];

    // New rich format: "error[E0100]: message" followed by " --> file:line:col"
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
            // Look ahead for location line
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

    // Fallback: simple "error: msg at line N" pattern
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

// ---------------------------------------------------------------------------
// REPL
// ---------------------------------------------------------------------------

function openRepl(_context: vscode.ExtensionContext): void {
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

// ---------------------------------------------------------------------------
// IR / LLVM virtual document viewer
// ---------------------------------------------------------------------------

let lastEmitContent = '';
let lastEmitLanguage = 'plaintext';

class IrisEmitProvider implements vscode.TextDocumentContentProvider {
    provideTextDocumentContent(_uri: vscode.Uri): string {
        return lastEmitContent;
    }
}

async function showEmit(kind: 'ir' | 'llvm'): Promise<void> {
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

// ---------------------------------------------------------------------------
// Code Lens — inline ▷ Run / ⬡ Debug buttons
// ---------------------------------------------------------------------------

class IrisCodeLensProvider implements vscode.CodeLensProvider {
    // Find all zero-argument function definitions: `def name() ->` or `pub def name() ->`
    private readonly zeroArgFn = /^(?:pub\s+)?def\s+(\w+)\s*\(\s*\)\s*->/gm;

    provideCodeLenses(document: vscode.TextDocument): vscode.CodeLens[] {
        const lenses: vscode.CodeLens[] = [];
        const text = document.getText();
        let match: RegExpExecArray | null;
        this.zeroArgFn.lastIndex = 0;

        while ((match = this.zeroArgFn.exec(text)) !== null) {
            const fnName = match[1];
            if (fnName.startsWith('__')) { continue; } // skip internal fns
            const pos = document.positionAt(match.index);
            const range = new vscode.Range(pos, pos);
            const uri = document.uri.toString();

            lenses.push(
                new vscode.CodeLens(range, {
                    title: '▷ Run',
                    command: 'iris.runFunction',
                    arguments: [uri, fnName],
                    tooltip: `Run ${fnName}()`,
                }),
                new vscode.CodeLens(range, {
                    title: '⬡ Debug',
                    command: 'workbench.action.debug.start',
                    tooltip: `Debug ${fnName}()`,
                }),
            );
        }
        return lenses;
    }
}

// ---------------------------------------------------------------------------
// Debug Adapter
// ---------------------------------------------------------------------------

class IrisDebugAdapterFactory implements vscode.DebugAdapterDescriptorFactory {
    createDebugAdapterDescriptor(): vscode.ProviderResult<vscode.DebugAdapterDescriptor> {
        return new vscode.DebugAdapterExecutable(getIrisExe(), ['dap']);
    }
}

// ---------------------------------------------------------------------------
// Deactivation
// ---------------------------------------------------------------------------

export function deactivate(): Thenable<void> | undefined {
    runDiagCollection.dispose();
    return client?.stop();
}
