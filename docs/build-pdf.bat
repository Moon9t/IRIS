@echo off
echo Building PDF...
"C:\Users\Moon\pandoc\pandoc-3.6.4\pandoc.exe" docs\BOOK.md -o "docs\The-IRIS-Programming-Language.pdf" ^
  --syntax-definition=docs\iris.xml ^
  --highlight-style=docs\iris-theme.theme ^
  --pdf-engine="C:\Users\Moon\AppData\Local\Programs\MiKTeX\miktex\bin\x64\xelatex.exe" ^
  --toc --toc-depth=2 --number-sections ^
  --top-level-division=chapter -V book=true ^
  -V "title=The IRIS Programming Language" ^
  -V "author=Moon9t" ^
  -V "date=May 2026" ^
  -V "lang=en"

if %errorlevel% neq 0 (
  echo Failed to build PDF!
  exit /b %errorlevel%
)
echo PDF built successfully!
