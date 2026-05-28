import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import { initStatusBar, updateStatusBar, showServerMenu } from './statusBar';
import { startLspClient, restartLsp, stopLsp, client } from './lspClient';
import {
    runIrisFile,
    runNamedFunction,
    debugIrisFile,
    openRepl,
    showEmit,
    showFullVersion,
    IrisEmitProvider,
    runDiagCollection,
} from './commands';
import { IrisDebugAdapterFactory } from './debugAdapter';
import { IrisCodeLensProvider } from './codelens';
import { IrisTaskProvider } from './tasks';

let outputChannel: vscode.OutputChannel;
let serverOutputChannel: vscode.OutputChannel;

export async function activate(context: vscode.ExtensionContext): Promise<void> {
    outputChannel = vscode.window.createOutputChannel('IRIS');
    serverOutputChannel = vscode.window.createOutputChannel('IRIS Language Server');
    context.subscriptions.push(outputChannel, serverOutputChannel);

    // Initialize Status Bar
    initStatusBar(context);
    updateStatusBar('starting', getIrisExe);

    // Register Commands
    context.subscriptions.push(
        vscode.commands.registerCommand('iris.runFile',    () => runIrisFile(context, 'run', getIrisExe, outputChannel)),
        vscode.commands.registerCommand('iris.buildFile',  () => runIrisFile(context, 'build', getIrisExe, outputChannel)),
        vscode.commands.registerCommand('iris.debugFile',  (uri?: vscode.Uri | string) => debugIrisFile(uri, getIrisExe)),
        vscode.commands.registerCommand('iris.openRepl',   () => openRepl(context, getIrisExe)),
        vscode.commands.registerCommand('iris.restartLsp', () => restartLsp(context, getIrisExe, serverOutputChannel)),
        vscode.commands.registerCommand('iris.stopLsp',    () => stopLsp(getIrisExe)),
        vscode.commands.registerCommand('iris.showIR',     () => showEmit('ir', getIrisExe, outputChannel)),
        vscode.commands.registerCommand('iris.showLLVM',   () => showEmit('llvm', getIrisExe, outputChannel)),
        vscode.commands.registerCommand('iris.showVersion', () => showFullVersion(getIrisExe, outputChannel)),
        vscode.commands.registerCommand('iris.runFunction', (uri: string, fnName: string) =>
            runNamedFunction(uri, fnName, getIrisExe, outputChannel)),
        vscode.commands.registerCommand('iris.serverMenu', () => showServerMenu(context, client !== undefined, {
            restartLsp: () => restartLsp(context, getIrisExe, serverOutputChannel),
            stopLsp: () => stopLsp(getIrisExe),
            startLsp: () => startLspClient(context, getIrisExe, serverOutputChannel),
            showServerOutput: () => serverOutputChannel.show(),
            openRepl: () => openRepl(context, getIrisExe),
            showFullVersion: () => showFullVersion(getIrisExe, outputChannel),
        })),
        vscode.commands.registerCommand('iris.runTest', (uriStr: string, fnName: string) => {
            const filePath = vscode.Uri.parse(uriStr).fsPath;
            const terminal = vscode.window.createTerminal({
                name: `IRIS Test - ${fnName}`,
                shellPath: getIrisExe(),
                shellArgs: ['test', filePath, '--filter', fnName],
            });
            terminal.show();
        }),
        vscode.commands.registerCommand('iris.debugTest', (uriStr: string, fnName: string) => {
            const filePath = vscode.Uri.parse(uriStr).fsPath;
            vscode.debug.startDebugging(vscode.workspace.getWorkspaceFolder(vscode.Uri.parse(uriStr)), {
                type: 'iris',
                request: 'launch',
                name: `Debug Test ${fnName}`,
                program: filePath,
            });
        }),
    );

    // Register CodeLens Provider
    context.subscriptions.push(
        vscode.languages.registerCodeLensProvider(
            { scheme: 'file', language: 'iris' },
            new IrisCodeLensProvider(),
        ),
    );

    // Register Text Document Content Provider for emitting IR/LLVM
    context.subscriptions.push(
        vscode.workspace.registerTextDocumentContentProvider('iris-emit', new IrisEmitProvider()),
    );

    // Register Debug Adapter Descriptor Factory
    context.subscriptions.push(
        vscode.debug.registerDebugAdapterDescriptorFactory('iris', new IrisDebugAdapterFactory(getIrisExe)),
    );

    // Register Task Provider
    context.subscriptions.push(
        vscode.tasks.registerTaskProvider('iris', new IrisTaskProvider(getIrisExe)),
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

    // Respond to inlayHint setting changes
    context.subscriptions.push(
        vscode.workspace.onDidChangeConfiguration(e => {
            if (e.affectsConfiguration('iris.inlayHints.enabled')) {
                // Settings handler if needed in future
            }
        }),
    );

    // Start LSP Client
    await startLspClient(context, getIrisExe, serverOutputChannel);
}

export function getIrisExe(): string {
    const cfg = vscode.workspace.getConfiguration('iris').get<string>('executablePath', '');
    if (cfg && fs.existsSync(cfg)) {
        return cfg;
    }
    const candidates = [
        path.join(process.env.USERPROFILE || '', '.cargo', 'bin', 'iris.exe'),
        path.join(process.env.USERPROFILE || '', '.iris', 'bin', 'iris.exe'),
        'C:\\Program Files\\IRIS\\iris.exe',
        'C:\\Users\\' + (process.env.USERNAME || '') + '\\.cargo\\bin\\iris.exe',
        'iris',
    ];
    for (const c of candidates) {
        if (c === 'iris') { return c; }
        if (fs.existsSync(c)) { return c; }
    }
    return 'iris';
}

export function deactivate(): Thenable<void> | undefined {
    runDiagCollection.dispose();
    return client?.stop();
}
