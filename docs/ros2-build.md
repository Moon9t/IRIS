# Building the ROS 2 bridge

`std.ros2` calls into `iris_ros2.dll`, a thin C bridge over `rcl`
(`src/runtime/ros2_bridge.c`). The DLL is **not** built by `cargo build` — it
needs a ROS 2 installation, so it is built separately and placed where the
program can load it.

Verified 2026-08-16 against **ROS 2 Humble for Windows**, installed at
`C:\dev\ros2_humble\ros2-windows`.

## What works

A native IRIS binary and the interpreter both create a node and publisher and
publish `std_msgs/Float64` messages:

```
rcl_init -> 0
node -> 1877887568
publisher -> 1878009216
publish -> true
```

**Not yet verified:** subscriptions and the six `take_*` payload functions, QoS
profiles, services, actions, tf2, lifecycle nodes. See
`docs/autonomy-stack-assessment.md` — the 2/10 rating covers the whole stack and
is still broadly right. What changed is that the publish path is demonstrated
rather than assumed.

## The two obstacles

Neither is IRIS's fault, and both cost time, so they are written down.

**1 — ROS 2's Windows headers are MSVC-only.** Under `__GNUC__` they place
`__attribute__((dllimport))` on `enum` declarations:

```c
// rmw/visibility_control.h
#ifdef __GNUC__
  #define RMW_IMPORT __attribute__ ((dllimport))
#endif
#define RMW_PUBLIC_TYPE RMW_PUBLIC     // -> dllimport, applied to an enum
```

MSVC accepts this with a warning; GCC and clang's GNU target reject it as a
syntax error. So the bridge must be compiled with
`-target x86_64-pc-windows-msvc`, **not** the MinGW target the rest of the
project uses. Patching the ROS 2 headers would work and should not be done.

**2 — the MSVC BuildTools install here has headers but no CRT.**
`VC/Tools/MSVC/14.44.35207/lib/x64` contains exactly two files
(`pgobootrun.lib`, `pgort.lib`). `vcruntime.lib` is absent, which is where
`memcpy` and `_fltused` live. Both are supplied by a two-function shim below.
Installing the "MSVC v143 C++ x64 build tools" component removes the need for it.

## Build

```bash
R=/c/dev/ros2_humble/ros2-windows
INC=""; for p in rcl rcutils rmw rosidl_runtime_c rosidl_typesupport_interface \
                 std_msgs geometry_msgs builtin_interfaces rcl_yaml_param_parser \
                 rcpputils service_msgs type_description_interfaces \
                 rosidl_dynamic_typesupport; do
  [ -d "$R/include/$p" ] && INC="$INC -I$R/include/$p"
done

# 1. Compile the bridge with the MSVC target.
"/c/Program Files/LLVM/bin/clang.exe" -target x86_64-pc-windows-msvc -c \
  -o ros2_bridge.obj src/runtime/ros2_bridge.c $INC -Wno-ignored-attributes

# 2. CRT shim -- only needed because vcruntime.lib is missing (see above).
cat > crtshim.c <<'EOF'
#include <stddef.h>
int _fltused = 0x9875;
void *memcpy(void *d, const void *s, size_t n) {
    unsigned char *dd = (unsigned char *)d;
    const unsigned char *ss = (const unsigned char *)s;
    for (size_t i = 0; i < n; i++) dd[i] = ss[i];
    return d;
}
EOF
"/c/Program Files/LLVM/bin/clang.exe" -target x86_64-pc-windows-msvc -c -O2 \
  -o crtshim.obj crtshim.c

# 3. Link. Dash-form flags, not /DLL -- msys rewrites a leading slash into a
#    path and lld-link then looks for C:/Program Files/Git/DLL.
UCRT="C:/Program Files (x86)/Windows Kits/10/Lib/10.0.26100.0/ucrt/x64"
UM="C:/Program Files (x86)/Windows Kits/10/Lib/10.0.26100.0/um/x64"
"/c/Program Files/LLVM/bin/lld-link.exe" -dll -noentry -out:iris_ros2.dll \
  ros2_bridge.obj crtshim.obj \
  -libpath:C:/dev/ros2_humble/ros2-windows/lib "-libpath:$UCRT" "-libpath:$UM" \
  rcl.lib rcutils.lib rmw.lib rosidl_runtime_c.lib \
  std_msgs__rosidl_typesupport_c.lib std_msgs__rosidl_generator_c.lib \
  geometry_msgs__rosidl_typesupport_c.lib geometry_msgs__rosidl_generator_c.lib \
  ucrt.lib kernel32.lib
```

`-noentry` avoids needing `_DllMainCRTStartup`, which also lives in the absent
`vcruntime.lib`. The UCRT is itself a DLL and self-initialises, so the bridge's
`malloc` and `fprintf` work without CRT startup having run.

## Run

`iris_ros2.dll` must be findable by the loader (the working directory is enough),
and ROS 2's own DLLs must be on `PATH`:

```bash
export PATH="/c/dev/ros2_humble/ros2-windows/bin:$PATH"
cp iris_ros2.dll .
target/debug/iris.exe --emit eval your_program.iris
```

## What this exercise found

Running against real ROS 2 immediately exposed **known-issues #33**: every native
FFI call carrying arguments crashed with an access violation, because codegen
passed arguments flat where `iris_ffi_call_i64` expects a pointer to an array
plus a count. `rcl_init`, node creation and publisher creation take no arguments
and worked; `iris_rcl_publish(handle, value)` was the first two-argument call and
segfaulted.

Nothing in the tree exercised an argument-carrying FFI call on the native
backend, so it had never been observed — the interpreter marshals correctly and
`--emit eval` falls back to it when a native build fails.

That is the general lesson: **the FFI surface cannot be validated without
something real on the other end of it.** A mock that IRIS also wrote would have
had the same misunderstanding of the calling convention.
