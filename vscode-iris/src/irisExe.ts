import * as vscode from 'vscode';
import * as child_process from 'child_process';
import * as fs from 'fs';
import * as path from 'path';

const IS_WIN = process.platform === 'win32';
const EXE_NAME = IS_WIN ? 'iris.exe' : 'iris';

export interface ProbeResult {
    ok: boolean;
    /** First line of `--version` on success, or a human-readable reason. */
    detail: string;
    /** Raw exit status when the process launched but failed. */
    status?: number | null;
}

let resolved: string | undefined;
let lastProbe: ProbeResult | undefined;

/**
 * Whether this binary can actually launch.
 *
 * Existence on disk is NOT enough. A stale install can fail during image load —
 * a build that statically imported the ML DLLs dies with STATUS_DLL_NOT_FOUND
 * (0xC0000135) before `main` runs. Through the language client that surfaces
 * only as "server exited with code 1", which says nothing useful, so we probe
 * with `--version` and report the real status.
 */
export function probeIrisExe(exe: string): ProbeResult {
    try {
        const out = child_process.execFileSync(exe, ['--version'], {
            encoding: 'utf8',
            timeout: 8000,
            windowsHide: true,
        });
        const first = out.split('\n')[0].trim();
        return { ok: true, detail: first || 'launched' };
    } catch (err: any) {
        return { ok: false, detail: describeLaunchFailure(err), status: err?.status ?? null };
    }
}

/** Turn a failed spawn into something a user can act on. */
function describeLaunchFailure(err: any): string {
    if (err?.code === 'ENOENT') {
        return 'not found';
    }
    if (err?.code === 'ETIMEDOUT') {
        return 'timed out running --version';
    }
    // Windows surfaces loader failures as NTSTATUS values in the exit code.
    const status: number | null | undefined = err?.status;
    if (typeof status === 'number') {
        const unsigned = status < 0 ? status + 0x100000000 : status;
        switch (unsigned) {
            case 0xc0000135:
                return 'missing DLL (STATUS_DLL_NOT_FOUND 0xC0000135) — the binary cannot load; '
                    + 'it is most likely a stale install that still imports the ML runtime DLLs';
            case 0xc0000139:
                return 'missing DLL entry point (STATUS_ENTRYPOINT_NOT_FOUND 0xC0000139)';
            case 0xc000007b:
                return 'architecture mismatch (STATUS_INVALID_IMAGE_FORMAT 0xC000007B) — 32/64-bit mix';
            case 0xc0000005:
                return 'crashed on start (STATUS_ACCESS_VIOLATION 0xC0000005)';
            default:
                return `exited with status 0x${unsigned.toString(16).toUpperCase()}`;
        }
    }
    return String(err?.message ?? err);
}

/**
 * Candidate binaries, best first.
 *
 * A workspace build is preferred: when working on the compiler itself, the
 * tree's own binary is the one that matches the sources being edited, and the
 * global install is usually older. Ordinary `.iris` projects have no `target/`
 * directory, so this costs them nothing.
 */
function candidatePaths(): string[] {
    const out: string[] = [];
    for (const folder of vscode.workspace.workspaceFolders ?? []) {
        const root = folder.uri.fsPath;
        out.push(path.join(root, 'target', 'release', EXE_NAME));
        out.push(path.join(root, 'target', 'debug', EXE_NAME));
    }
    const home = process.env.USERPROFILE || process.env.HOME || '';
    if (home) {
        out.push(path.join(home, '.iris', 'bin', EXE_NAME));
        out.push(path.join(home, '.cargo', 'bin', EXE_NAME));
    }
    if (IS_WIN) {
        out.push('C:\\Program Files\\IRIS\\iris.exe');
    } else {
        out.push('/usr/local/bin/iris');
    }
    // Last resort: whatever PATH resolves. Has no path to stat, so it is only
    // ever probed, never existence-checked.
    out.push(EXE_NAME);
    return out;
}

/**
 * Resolve the iris executable, preferring one that actually runs.
 *
 * An explicit `iris.executablePath` is honoured verbatim — if a user points at
 * a specific binary we must not silently substitute another, or the errors they
 * see will not match the compiler they think they are running.
 */
export function getIrisExe(): string {
    const configured = vscode.workspace
        .getConfiguration('iris')
        .get<string>('executablePath', '')
        .trim();
    if (configured && configured !== 'iris' && configured !== EXE_NAME) {
        return configured;
    }

    if (resolved) {
        return resolved;
    }

    const candidates = candidatePaths();
    let firstExisting: string | undefined;

    for (const candidate of candidates) {
        const bare = candidate === EXE_NAME;
        if (!bare && !fs.existsSync(candidate)) {
            continue;
        }
        if (!firstExisting) {
            firstExisting = candidate;
        }
        const probe = probeIrisExe(candidate);
        if (probe.ok) {
            resolved = candidate;
            lastProbe = probe;
            return candidate;
        }
        lastProbe = probe;
    }

    // Nothing launched. Return something concrete so the failure message names a
    // real path rather than a bare "iris".
    resolved = firstExisting ?? EXE_NAME;
    return resolved;
}

/** The probe result for the most recently considered candidate, if any. */
export function getLastProbe(): ProbeResult | undefined {
    return lastProbe;
}

/** Re-run resolution — call after settings change or an explicit restart. */
export function resetIrisExeCache(): void {
    resolved = undefined;
    lastProbe = undefined;
}

/** Every candidate with its probe outcome, for diagnostics output. */
export function diagnoseCandidates(): string[] {
    const lines: string[] = [];
    for (const candidate of candidatePaths()) {
        const bare = candidate === EXE_NAME;
        if (!bare && !fs.existsSync(candidate)) {
            lines.push(`  [absent ] ${candidate}`);
            continue;
        }
        const probe = probeIrisExe(candidate);
        lines.push(`  [${probe.ok ? 'OK     ' : 'BROKEN '}] ${candidate} — ${probe.detail}`);
    }
    return lines;
}
