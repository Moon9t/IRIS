import * as vscode from 'vscode';

export class IrisTaskProvider implements vscode.TaskProvider {
    private readonly getIrisExe: () => string;

    constructor(getIrisExe: () => string) {
        this.getIrisExe = getIrisExe;
    }

    public provideTasks(): vscode.ProviderResult<vscode.Task[]> {
        const tasks: vscode.Task[] = [];

        // 1. Run task
        const runTask = new vscode.Task(
            { type: 'iris', task: 'run' },
            vscode.TaskScope.Workspace,
            'run',
            'iris',
            new vscode.ProcessExecution(this.getIrisExe(), ['run', '${file}']),
            '$iris'
        );
        // No group: VS Code's TaskGroup has only Clean, Build, Rebuild and Test,
        // so a run task stays ungrouped.
        tasks.push(runTask);

        // 2. Build task
        const buildTask = new vscode.Task(
            { type: 'iris', task: 'build' },
            vscode.TaskScope.Workspace,
            'build',
            'iris',
            new vscode.ProcessExecution(this.getIrisExe(), ['build', '${file}']),
            '$iris'
        );
        buildTask.group = vscode.TaskGroup.Build;
        tasks.push(buildTask);

        // 3. Test task
        const testTask = new vscode.Task(
            { type: 'iris', task: 'test' },
            vscode.TaskScope.Workspace,
            'test',
            'iris',
            new vscode.ProcessExecution(this.getIrisExe(), ['test']),
            '$iris'
        );
        testTask.group = vscode.TaskGroup.Test;
        tasks.push(testTask);

        return tasks;
    }

    public resolveTask(task: vscode.Task): vscode.Task | undefined {
        const definition = task.definition;
        if (definition.task) {
            const args: string[] = [];
            if (definition.task === 'run') {
                args.push('run', '${file}');
            } else if (definition.task === 'build') {
                args.push('build', '${file}');
            } else if (definition.task === 'test') {
                args.push('test');
            } else {
                return undefined;
            }
            return new vscode.Task(
                definition,
                task.scope ?? vscode.TaskScope.Workspace,
                task.name ?? definition.task,
                'iris',
                new vscode.ProcessExecution(this.getIrisExe(), args),
                '$iris'
            );
        }
        return undefined;
    }
}
