@echo off
echo Building PDF (generating .tex then running XeLaTeX twice)...
"C:\Users\Moon\pandoc\pandoc-3.6.4\pandoc.exe" -s docs\BOOK.md -o "docs\The-IRIS-Programming-Language.tex" ^
  --syntax-definition=docs\iris.xml ^
  --highlight-style=docs\iris-theme.theme ^
  --toc --toc-depth=2 --number-sections ^
  --top-level-division=chapter -V book=true ^
  -V "title=The IRIS Programming Language" ^
  -V "author=Moon9t" ^
  -V "date=May 2026" ^
  -V "lang=en"

if %errorlevel% neq 0 (
  echo Failed to generate LaTeX source!
  exit /b %errorlevel%
)

REM Run XeLaTeX three times to resolve cross-references and hyperlinks
"C:\Users\Moon\AppData\Local\Programs\MiKTeX\miktex\bin\x64\xelatex.exe" -interaction=nonstopmode -halt-on-error -output-directory=docs docs\The-IRIS-Programming-Language.tex
if %errorlevel% neq 0 (
  echo XeLaTeX first pass failed!
  exit /b %errorlevel%
)
"C:\Users\Moon\AppData\Local\Programs\MiKTeX\miktex\bin\x64\xelatex.exe" -interaction=nonstopmode -halt-on-error -output-directory=docs docs\The-IRIS-Programming-Language.tex
if %errorlevel% neq 0 (
  echo XeLaTeX second pass failed!
  exit /b %errorlevel%
)
"C:\Users\Moon\AppData\Local\Programs\MiKTeX\miktex\bin\x64\xelatex.exe" -interaction=nonstopmode -halt-on-error -output-directory=docs docs\The-IRIS-Programming-Language.tex
if %errorlevel% neq 0 (
  echo XeLaTeX third pass failed!
  exit /b %errorlevel%
)

echo PDF built successfully!
