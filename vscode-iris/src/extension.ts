import * as vscode from 'vscode';
import { initStatusBar, updateStatusBar, showServerMenu } from './statusBar';
import { startLspClient, restartLsp, stopLsp, client } from './lspClient';
import { getIrisExe, resetIrisExeCache, diagnoseCandidates } from './irisExe';
import {
    runIrisFile,
    runNamedFunction,
    debugIrisFile,
    openRepl,
    showEmit,
    showEmitPick,
    showFullVersion,
    explainErrorCode,
    checkFormatting,
    runFileTests,
    runBenchmarks,
    generateDocs,
    pkgCommand,
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
        vscode.commands.registerCommand('iris.showEmit',    () => showEmitPick(getIrisExe, outputChannel)),
        vscode.commands.registerCommand('iris.showGraph',   () => showEmit('graph', getIrisExe, outputChannel)),
        vscode.commands.registerCommand('iris.showCuda',    () => showEmit('cuda', getIrisExe, outputChannel)),
        vscode.commands.registerCommand('iris.showSimd',    () => showEmit('simd', getIrisExe, outputChannel)),
        vscode.commands.registerCommand('iris.explainError', () => explainErrorCode(getIrisExe, outputChannel)),
        vscode.commands.registerCommand('iris.checkFormat', () => checkFormatting(getIrisExe, outputChannel)),
        vscode.commands.registerCommand('iris.runTests',    () => runFileTests(getIrisExe)),
        vscode.commands.registerCommand('iris.runBench',    () => runBenchmarks(getIrisExe)),
        vscode.commands.registerCommand('iris.generateDocs', () => generateDocs(getIrisExe)),
        vscode.commands.registerCommand('iris.pkg',         () => pkgCommand(getIrisExe)),
        vscode.commands.registerCommand('iris.diagnostics', () => {
            outputChannel.clear();
            outputChannel.appendLine('=== IRIS executable resolution ===');
            outputChannel.appendLine('');
            outputChannel.appendLine(`selected: ${getIrisExe()}`);
            outputChannel.appendLine('');
            outputChannel.appendLine('candidates (best first):');
            for (const line of diagnoseCandidates()) {
                outputChannel.appendLine(line);
            }
            outputChannel.show(true);
        }),
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

    // Settings that are only read at server startup need a restart to apply:
    // the executable choice, and the inlay-hint flags sent as
    // initializationOptions.
    context.subscriptions.push(
        vscode.workspace.onDidChangeConfiguration(async e => {
            if (e.affectsConfiguration('iris.executablePath')) {
                resetIrisExeCache();
                await restartLsp(context, getIrisExe, serverOutputChannel);
            } else if (
                e.affectsConfiguration('iris.inlayHints.enabled') ||
                e.affectsConfiguration('iris.inlayHints.typeHints')
            ) {
                await restartLsp(context, getIrisExe, serverOutputChannel);
            }
        }),
    );

    // Start LSP Client
    await startLspClient(context, getIrisExe, serverOutputChannel);
}

export function deactivate(): Thenable<void> | undefined {
    runDiagCollection.dispose();
    return client?.stop();
}
