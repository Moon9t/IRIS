import * as vscode from 'vscode';
import {
    LanguageClient,
    LanguageClientOptions,
    ServerOptions,
    TransportKind,
    State,
} from 'vscode-languageclient/node';
import { updateStatusBar, clearVersionCache } from './statusBar';

export let client: LanguageClient | undefined;

export async function startLspClient(
    context: vscode.ExtensionContext,
    getIrisExe: () => string,
    serverOutputChannel: vscode.OutputChannel
): Promise<void> {
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
            updateStatusBar('running', getIrisExe);
        } else if (e.newState === State.Stopped) {
            updateStatusBar('stopped', getIrisExe);
        } else {
            updateStatusBar('starting', getIrisExe);
        }
    });

    try {
        await client.start();
        context.subscriptions.push(client);
    } catch (err) {
        updateStatusBar('error', getIrisExe);
        const choice = await vscode.window.showErrorMessage(
            `IRIS: Could not start language server using '${exe}'.`,
            'Open Settings',
            'Retry',
        );
        if (choice === 'Open Settings') {
            vscode.commands.executeCommand('workbench.action.openSettings', 'iris.executablePath');
        } else if (choice === 'Retry') {
            await startLspClient(context, getIrisExe, serverOutputChannel);
        }
    }
}

export async function restartLsp(
    context: vscode.ExtensionContext,
    getIrisExe: () => string,
    serverOutputChannel: vscode.OutputChannel
): Promise<void> {
    clearVersionCache();
    if (client) {
        updateStatusBar('starting', getIrisExe);
        try {
            await client.stop();
            client.dispose();
        } catch { /* ignore stop errors */ }
        client = undefined;
    }
    await startLspClient(context, getIrisExe, serverOutputChannel);
    vscode.window.showInformationMessage('IRIS: Language server restarted.');
}

export async function stopLsp(getIrisExe: () => string): Promise<void> {
    if (client) {
        try {
            await client.stop();
            client.dispose();
        } catch { /* ignore stop errors */ }
        client = undefined;
        updateStatusBar('stopped', getIrisExe);
        vscode.window.showInformationMessage('IRIS: Language server stopped.');
    } else {
        vscode.window.showInformationMessage('IRIS: Language server is not running.');
    }
}
