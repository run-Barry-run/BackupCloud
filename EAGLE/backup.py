import os
import shutil

def copy_directory_structure(src_dir, dst_dir):
    """
    Walks through the source directory and copies its contents to the destination directory
    while preserving the directory structure.
    
    :param src_dir: The source directory to copy from.
    :param dst_dir: The destination directory to copy to.
    """
    for root, dirs, files in os.walk(src_dir):
        # Create the corresponding directory in the destination
        relative_path = os.path.relpath(root, src_dir)
        dest_path = os.path.join(dst_dir, relative_path)
        
        if dest_path.endswith('backup') or dirs.endswith('backup'):
            continue

        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        
        # Copy files to the destination directory
        for file in files:
            if file.endswith('.sh') or file.endswith('.py'):
                src_file_path = os.path.join(root, file)
                dest_file_path = os.path.join(dest_path, file)
                shutil.copy2(src_file_path, dest_file_path)
                print(f"Copied: {src_file_path} -> {dest_file_path}")

# Example usage
source_directory = "/home1/hxl/disk/EAGLE"
destination_directory = "/home1/hxl/disk/EAGLE/backup"

copy_directory_structure(source_directory, destination_directory)
