#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 <input.ll> <output.ll>" >&2
  exit 2
fi

input_path=$1
output_path=$2

cp "$input_path" "$output_path"

# LLVM 19 still rejects some rustc-emitted parameter attribute syntax in the
# captured artifact. Strip the unsupported attributes only on LLVM signature
# lines so quoted string literals remain untouched.
perl -0pi -e '
  @lines = split(/\n/, $_, -1);
  for (@lines) {
    if (/(?:\bdefine\b|\bdeclare\b|\bcall\b|\btail call\b|\binvoke\b)/) {
      s/captures\(none\) //g;
      s/captures\(address\) //g;
      s/captures\(address, read_provenance\) //g;
      s/captures\(address_is_null\) //g;
      s/range\([^)]*\) //g;
    }
    s/icmp samesign /icmp /g;
  }
  $_ = join("\n", @lines);
' "$output_path"