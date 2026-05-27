#!/usr/bin/env bash
set -e
mkdir -p /work/ci-artifacts
# Try to emit LLVM IR for the iris lib; continue on errors
cargo rustc -p iris --lib --release -- --emit=llvm-ir || true
# Copy any generated .ll files to the shared artifacts dir
find target -type f -name '*.ll' -print -exec cp {} /work/ci-artifacts/ \; || true
# Ensure the loop handles no-matches gracefully
shopt -s nullglob
for ll in /work/ci-artifacts/*.ll; do
  echo "Checking $ll"
  if command -v clang-22 >/dev/null 2>&1; then
    clang_cmd=clang-22
  elif command -v clang-21 >/dev/null 2>&1; then
    clang_cmd=clang-21
  elif command -v clang-20 >/dev/null 2>&1; then
    clang_cmd=clang-20
  else
    clang_cmd=clang
  fi
  "$clang_cmd" -c "$ll" -o /dev/null 2> "$ll.clang.err" || true
done
ls -l /work/ci-artifacts
