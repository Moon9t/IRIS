import os
import subprocess
import sys

def main():
    ros2_dir = r"C:\dev\ros2_humble\ros2-windows"
    if not os.path.exists(ros2_dir):
        print(f"Error: ROS2 directory {ros2_dir} does not exist.")
        sys.exit(1)

    include_base = os.path.join(ros2_dir, "include")
    lib_dir = os.path.join(ros2_dir, "lib")

    # MSVC and Windows SDK library paths for x64
    msvc_lib = r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\lib\x64"
    ucrt_lib = r"C:\Program Files (x86)\Windows Kits\10\Lib\10.0.26100.0\ucrt\x64"
    um_lib   = r"C:\Program Files (x86)\Windows Kits\10\Lib\10.0.26100.0\um\x64"

    # Gather all subdirectories in the include directory.
    # Use -isystem so ROS2 headers don't shadow system headers like <string.h>
    # (CycloneDDS/string.h, mimick/string.h exist in the ROS2 tree).
    include_paths = []
    if os.path.exists(include_base):
        for entry in os.scandir(include_base):
            if entry.is_dir():
                include_paths.append(entry.path)

    # Build the clang command
    cmd = [
        "clang",
        "-target", "x86_64-pc-windows-msvc",
        "-fuse-ld=lld",
        "-shared",
        "-o", "iris_ros2.dll",
        "src/runtime/ros2_bridge.c",
        f"-Wl,/def:src/runtime/ros2_bridge.def",
        "-isystem", include_base,
    ]

    for path in include_paths:
        cmd.extend(["-isystem", path])

    cmd.extend([
        # MSVC / Windows SDK lib paths
        f"-L{msvc_lib}",
        f"-L{ucrt_lib}",
        f"-L{um_lib}",
        # ROS2 lib path
        f"-L{lib_dir}",
        # Suppress non-fatal warnings from ROS2 headers
        "-Wno-ignored-attributes",
        "-Wno-pragma-pack",
        "-Wno-microsoft-static-assert",
        # CycloneDDS/string.h shadows system <string.h> in ROS2 include tree,
        # causing strcmp/strncpy to appear undeclared. The functions link fine.
        "-Wno-implicit-function-declaration",
        # Core ROS2 client libraries
        "-lrcl",
        "-lrcutils",
        "-lrosidl_runtime_c",
        "-lrosidl_typesupport_c",
        "-lrosidl_typesupport_cpp",
        # std_msgs
        "-lstd_msgs__rosidl_generator_c",
        "-lstd_msgs__rosidl_typesupport_c",
        # geometry_msgs
        "-lgeometry_msgs__rosidl_generator_c",
        "-lgeometry_msgs__rosidl_typesupport_c",
    ])

    print("Running compilation command...")
    print(" ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("Compilation FAILED!")
        print("Stdout:")
        print(res.stdout)
        print("Stderr:")
        print(res.stderr)
        sys.exit(res.returncode)
    else:
        print("Compilation SUCCESSFUL! Generated iris_ros2.dll")

if __name__ == "__main__":
    main()
