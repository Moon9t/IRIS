## Description

Add IRIS language detection and syntax highlighting to GitHub Linguist.

This submission adds:

- an `IRIS` language entry in `lib/linguist/languages.yml`
- a `source.iris` TextMate grammar from a dedicated permissively licensed repo
- representative IRIS samples under `samples/IRIS/`
- `.iris` heuristics to reduce false positives from unrelated `.iris` files

## Checklist

- [ ] **I am adding a new language.**
  - [ ] The extension of the new language is used in hundreds of repositories on GitHub.com.
    - Search results for each extension:
      - https://github.com/search?type=code&q=NOT+is%3Afork+path%3A*.iris+%28%22bring+std%22+OR+%22def+main%22+OR+%22record+%22+OR+%22val+%22%29
  - [ ] I have included a real-world usage sample for all extensions added in this PR:
    - Sample source(s):
      - IRIS project examples prepared in `linguist/samples/`
    - Sample license(s):
      - To be confirmed in the final PR once the sample licensing path is chosen
  - [ ] I have included a syntax highlighting grammar: [URL to permissively licensed IRIS grammar repo]
  - [x] I have added a color
    - Hex value: `#6A0DAD`
    - Rationale: close to the existing IRIS/editor branding and visually distinct from adjacent language colors
  - [ ] I have updated the heuristics to distinguish my language from others using the same extension.

## Notes

- `language_id` should be generated in the Linguist fork with `script/update-ids`.
- `.iris` is not extension-clean on GitHub, so the PR will need careful search evidence and disambiguation justification.
- The grammar source must be permissively licensed to satisfy Linguist's vendor policy.
