import * as vscode from 'vscode';

export class IrisCodeLensProvider implements vscode.CodeLensProvider {
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

            if (fnName.startsWith('test_')) {
                lenses.push(
                    new vscode.CodeLens(range, {
                        title: '▷ Run Test',
                        command: 'iris.runTest',
                        arguments: [uri, fnName],
                        tooltip: `Run Test ${fnName}()`,
                    }),
                    new vscode.CodeLens(range, {
                        title: '⬡ Debug Test',
                        command: 'iris.debugTest',
                        arguments: [uri, fnName],
                        tooltip: `Debug Test ${fnName}()`,
                    }),
                );
            } else {
                lenses.push(
                    new vscode.CodeLens(range, {
                        title: '▷ Run',
                        command: 'iris.runFunction',
                        arguments: [uri, fnName],
                        tooltip: `Run ${fnName}()`,
                    }),
                    new vscode.CodeLens(range, {
                        title: '⬡ Debug',
                        command: 'iris.debugFile',
                        arguments: [document.uri],
                        tooltip: `Debug ${fnName}()`,
                    }),
                );
            }
        }
        return lenses;
    }
}
