# GitHub Linguist Submission for IRIS

This directory contains a PR-prep bundle for adding IRIS to
[github-linguist/linguist](https://github.com/github-linguist/linguist).

It is intentionally split into copy-ready snippets plus a short reality check
about the remaining blockers, so we do not lose time rediscovering the same
constraints during submission.

## Current Status

What is ready here:

- `languages.yml` contains the language entry snippet for `lib/linguist/languages.yml`.
- `heuristics.yml` contains a candidate `.iris` disambiguation block for
  `lib/linguist/heuristics.yml`.
- `samples/` contains stronger IRIS programs that are more representative than
  tutorial snippets.
- `pr_body.md` is a draft PR body based on Linguist's current template.

What is not fully unblocked yet:

1. The current GitHub session does not have a writable fork of
   `github-linguist/linguist`, and `gh auth status` reports an invalid token for
   `Moon9t` as of April 26, 2026.
2. Linguist only accepts grammars from a permissively licensed source.
   `IRIS` is currently licensed `GPL-2.0-or-later`, so the syntax grammar needs
   either:
   - a dedicated permissively licensed grammar repository, or
   - an explicit dual-license / re-license decision for the grammar assets.
3. The `.iris` extension is shared with unrelated files on GitHub, so the PR
   needs good search evidence and likely a stronger disambiguation story than a
   simple extension mapping.

## Upstream Process

Linguist's current `CONTRIBUTING.md` flow is:

1. Add the language entry to `lib/linguist/languages.yml`.
2. Vendor the TextMate grammar with:

   ```bash
   script/add-grammar https://github.com/<owner>/<permissive-grammar-repo>
   ```

3. Add real-world samples to `samples/IRIS/`.
4. Update `lib/linguist/heuristics.yml` if the extension is shared or noisy.
5. Run:

   ```bash
   script/update-ids
   bundle exec rake test
   ```

6. Open a PR using Linguist's template and include GitHub code search evidence.

## Recommended Submission Plan

1. Create a dedicated permissively licensed grammar repo for `source.iris`.
   Reusing `vscode-iris/syntaxes/iris.tmLanguage.json` is fine, but the grammar
   repository itself needs a license Linguist accepts.
2. Fork `github-linguist/linguist`.
3. Apply the snippets from this directory into the fork.
4. Vendor the grammar with `script/add-grammar`.
5. Copy the sample files from `linguist/samples/` into `samples/IRIS/`.
6. Run `script/update-ids` and the test suite.
7. Use `pr_body.md` as the starting point for the pull request text.

## Search Evidence

Linguist requires public GitHub usage evidence. A good starting query for IRIS
syntax is:

```text
NOT is:fork path:*.iris ("bring std" OR "def main" OR "record " OR "val ")
```

Because `.iris` is reused by unrelated projects, expect to refine this query and
manually validate repository diversity before submission.

## Sample Notes

The current sample set is intentionally skewed toward representative IRIS code:

- algorithms and control flow
- concurrency and pipelines
- machine-learning style numeric code
- network / service style code

Avoid submitting `hello world` style samples to Linguist. Their current
contributing guide explicitly calls that out as insufficient.
