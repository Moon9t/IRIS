import * as vscode from 'vscode';

export class IrisDebugAdapterFactory implements vscode.DebugAdapterDescriptorFactory {
    private readonly getIrisExe: () => string;

    constructor(getIrisExe: () => string) {
        this.getIrisExe = getIrisExe;
    }

    createDebugAdapterDescriptor(): vscode.ProviderResult<vscode.DebugAdapterDescriptor> {
        return new vscode.DebugAdapterExecutable(this.getIrisExe(), ['dap']);
    }
}
