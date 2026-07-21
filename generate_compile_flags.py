import os

def main():
    ros2_dir = r"C:\dev\ros2_humble\ros2-windows"
    if not os.path.exists(ros2_dir):
        print(f"Error: ROS2 directory {ros2_dir} does not exist.")
        return

    include_base = os.path.join(ros2_dir, "include")

    flags = [
        "-target",
        "x86_64-pc-windows-msvc",
        "-isystem",
        include_base
    ]

    # Add all subdirectories
    if os.path.exists(include_base):
        for entry in os.scandir(include_base):
            if entry.is_dir():
                flags.extend(["-isystem", entry.path])

    # Write to compile_flags.txt in the workspace root
    with open("compile_flags.txt", "w") as f:
        for flag in flags:
            f.write(flag + "\n")
            
    print("Successfully generated compile_flags.txt for IDE indexer.")

if __name__ == "__main__":
    main()
