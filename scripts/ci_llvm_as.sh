#!/usr/bin/env bash
set -e
OUTDIR=/work/ci-artifacts
LLF=$OUTDIR/iris-26e59a331c3c7647.ll
ERRF=$OUTDIR/iris.ll.llvm-as.err
mkdir -p "$OUTDIR"
if command -v llvm-as-22 >/dev/null 2>&1; then
  llvm-as-22 "$LLF" -o "$OUTDIR/iris.bc" 2> "$ERRF" || true
elif command -v llvm-as-21 >/dev/null 2>&1; then
  llvm-as-21 "$LLF" -o "$OUTDIR/iris.bc" 2> "$ERRF" || true
elif command -v llvm-as-20 >/dev/null 2>&1; then
  llvm-as-20 "$LLF" -o "$OUTDIR/iris.bc" 2> "$ERRF" || true
elif command -v llvm-as-11 >/dev/null 2>&1; then
  llvm-as-11 "$LLF" -o "$OUTDIR/iris.bc" 2> "$ERRF" || true
elif command -v llvm-as >/dev/null 2>&1; then
  llvm-as "$LLF" -o "$OUTDIR/iris.bc" 2> "$ERRF" || true
else
  echo "llvm-as not found" > "$OUTDIR/llvm-as.missing"
fi
echo "--- llvm-as.err ---"
if [ -f "$ERRF" ]; then
  sed -n '1,200p' "$ERRF"
else
  echo "(no error file)"
fi
echo "--- ls ---"
ls -lh "$OUTDIR" | sed -n '1,200p'
