import * as vscode from 'vscode';
import {
    LanguageClient,
    LanguageClientOptions,
    ServerOptions,
    TransportKind,
    State,
} from 'vscode-languageclient/node';
import { updateStatusBar, clearVersionCache } from './statusBar';
import { probeIrisExe, diagnoseCandidates, resetIrisExeCache } from './irisExe';

export let client: LanguageClient | undefined;

/**
 * Report an executable that cannot launch.
 *
 * Without this the language client reports only "server exited with code 1",
 * then gives up after five restarts — which is indistinguishable from a bug in
 * the server itself. A binary that fails to load is by far the more common
 * cause, so name it explicitly and say what to do about it.
 */
async function reportBrokenExe(
    exe: string,
    reason: string,
    serverOutputChannel: vscode.OutputChannel
): Promise<void> {
    serverOutputChannel.appendLine(`IRIS: '${exe}' cannot be launched — ${reason}`);
    serverOutputChannel.appendLine('');
    serverOutputChannel.appendLine('Candidates considered:');
    for (const line of diagnoseCandidates()) {
        serverOutputChannel.appendLine(line);
    }
    serverOutputChannel.appendLine('');
    serverOutputChannel.appendLine(
        'Fix: build the compiler (`cargo build`) so a working target/debug binary '
        + 'exists, or set iris.executablePath to a binary that runs.'
    );

    const choice = await vscode.window.showErrorMessage(
        `IRIS language server did not start: ${reason}`,
        'Show Details',
        'Open Settings',
    );
    if (choice === 'Show Details') {
        serverOutputChannel.show();
    } else if (choice === 'Open Settings') {
        vscode.commands.executeCommand('workbench.action.openSettings', 'iris.executablePath');
    }
}

export async function startLspClient(
    context: vscode.ExtensionContext,
    getIrisExe: () => string,
    serverOutputChannel: vscode.OutputChannel
): Promise<void> {
    const exe = getIrisExe();

    // Preflight: confirm the binary runs before handing it to the language
    // client, so a load failure is reported as such instead of as five opaque
    // server crashes.
    const probe = probeIrisExe(exe);
    if (!probe.ok) {
        updateStatusBar('error', getIrisExe);
        await reportBrokenExe(exe, probe.detail, serverOutputChannel);
        return;
    }
    serverOutputChannel.appendLine(`IRIS: using ${exe} (${probe.detail})`);

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
        serverOutputChannel.appendLine(
            `IRIS: language server start failed using '${exe}': ${(err as Error)?.message ?? err}`
        );
        const choice = await vscode.window.showErrorMessage(
            `IRIS: Could not start language server using '${exe}'.`,
            'Show Details',
            'Open Settings',
            'Retry',
        );
        if (choice === 'Show Details') {
            serverOutputChannel.show();
        } else if (choice === 'Open Settings') {
            vscode.commands.executeCommand('workbench.action.openSettings', 'iris.executablePath');
        } else if (choice === 'Retry') {
            // Re-resolve first: the user may have just built or installed a
            // working binary, and the cached choice would ignore it.
            resetIrisExeCache();
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
    // A restart is the natural moment to reconsider which binary to use — it is
    // usually what the user just changed (built, installed, or reconfigured).
    resetIrisExeCache();
    if (client) {
        updateStatusBar('starting', getIrisExe);
        try {
            await client.stop();
            client.dispose();
        } catch { /* ignore stop errors */ }
        client = undefined;
    }
    await startLspClient(context, getIrisExe, serverOutputChannel);
    if (client) {
        vscode.window.showInformationMessage('IRIS: Language server restarted.');
    }
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
