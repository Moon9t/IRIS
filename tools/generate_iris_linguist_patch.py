from __future__ import annotations

import difflib
import pathlib
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "tools" / "iris-linguist.patch"

FILES = {
    "lib/linguist/languages.yml": "https://raw.githubusercontent.com/github/linguist/master/lib/linguist/languages.yml",
    "lib/linguist/heuristics.yml": "https://raw.githubusercontent.com/github/linguist/master/lib/linguist/heuristics.yml",
    "grammars.yml": "https://raw.githubusercontent.com/github/linguist/master/grammars.yml",
}

IRIS_LANGUAGE_BLOCK = """IRIS:\n  type: programming\n  color: \"#6A0DAD\"\n  aliases:\n    - iris-lang\n  extensions:\n    - \".iris\"\n  tm_scope: source.iris\n  ace_mode: text\n  interpreters:\n    - iris\n\n"""

IRIS_HEURISTIC_BLOCK = """  - extensions: ['.iris']\n    rules:\n      - language: IRIS\n        and:\n          - pattern: '\\bdef\\s+[a-zA-Z_][a-zA-Z0-9_]*\\s*\\('\n          - pattern:\n              - '\\b(val|var|bring|record|choice|spawn)\\b'\n              - '->\\s*[A-Za-z_][A-Za-z0-9_<>, ]*'\n              - '\\blist<[^>]+>'\n              - '\\boption<[^>]+>'\n              - '\\bresult<[^>]+>'\n          - negative_pattern:\n              - '^\\s*#include\\b'\n              - '^\\s*<\\?xml\\b'\n"""

IRIS_GRAMMAR_BLOCK = """vendor/grammars/iris-tmLanguage:\n- source.iris\n\n"""


def fetch(url: str) -> str:
    return urllib.request.urlopen(url).read().decode("utf-8")


def write_diff(relpath: str, original: str, modified: str) -> str:
    return "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            modified.splitlines(keepends=True),
            fromfile=f"a/{relpath}",
            tofile=f"b/{relpath}",
            n=3,
        )
    )


def main() -> None:
    diffs: list[str] = []

    languages = fetch(FILES["lib/linguist/languages.yml"])
    languages_new = languages.replace("iCalendar:\n", IRIS_LANGUAGE_BLOCK + "iCalendar:\n", 1)
    diffs.append(write_diff("lib/linguist/languages.yml", languages, languages_new))

    heuristics = fetch(FILES["lib/linguist/heuristics.yml"])
    heuristics_new = heuristics.replace("disambiguations:\n", "disambiguations:\n" + IRIS_HEURISTIC_BLOCK, 1)
    diffs.append(write_diff("lib/linguist/heuristics.yml", heuristics, heuristics_new))

    grammars = fetch(FILES["grammars.yml"])
    grammars_new = grammars.replace(
        "vendor/grammars/idris:\n",
        IRIS_GRAMMAR_BLOCK + "vendor/grammars/idris:\n",
        1,
    )
    diffs.append(write_diff("grammars.yml", grammars, grammars_new))

    OUT.write_text("".join(diffs), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
