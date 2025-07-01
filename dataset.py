# import os
# import sys
# import shutil
# import kagglehub
# from pathlib import Path

# def download_dataset():
#     try:
#         # Create resources directory if it doesn't exist
#         resources_dir = Path("resources")
#         resources_dir.mkdir(exist_ok=True)
        
#         print("Downloading Indian Sign Language dataset...")
#         # Download latest version of the dataset
#         cache_path = kagglehub.dataset_download("prathumarikeri/indian-sign-language-isl")
        
#         print(f"Dataset downloaded to cache: {cache_path}")
#         print("Copying dataset to resources folder...")
        
#         # Copy all files from cache to resources
#         for src in Path(cache_path).glob('*'):
#             dst = resources_dir / src.name
#             if src.is_file():
#                 shutil.copy2(src, dst)
#             elif src.is_dir():
#                 if dst.exists():
#                     shutil.rmtree(dst)
#                 shutil.copytree(src, dst)
        
#         print(f"\nDataset successfully copied to: {resources_dir.absolute()}")
#         print("\nNext steps:")
#         print("1. Run pre_processing.py to prepare the data")
#         print("2. Run training.py to train the model")
        
#         return True
    
#     except Exception as e:
#         print(f"Error downloading/copying dataset: {str(e)}")
#         print("\nTroubleshooting steps:")
#         print("1. Make sure you have kagglehub installed: pip install kagglehub")
#         print("2. Make sure you're authenticated with Kaggle:")
#         print("   - Go to kaggle.com → Account → Create API Token")
#         print("   - Download kaggle.json and place it in ~/.kaggle/")
#         return False

# if __name__ == "__main__":
#     download_dataset() 






import os
import shutil

# Paths
folder1 = 'data'         # Original folder (e.g., data/A/..., data/0/...)
folder2 = 'data2'        # Folder to merge from
merged_folder = 'data_collect'  # Final merged output

# Create merged folder if it doesn't exist
os.makedirs(merged_folder, exist_ok=True)

# Subfolders: '0'-'9' and 'A'-'Z'
subfolders = [str(i) for i in range(10)] + [chr(c) for c in range(ord('A'), ord('Z') + 1)]

for sub in subfolders:
    path1 = os.path.join(folder1, sub)
    path2 = os.path.join(folder2, sub)
    dest = os.path.join(merged_folder, sub)

    os.makedirs(dest, exist_ok=True)

    # Copy files from folder1
    if os.path.exists(path1):
        for filename in os.listdir(path1):
            src_file = os.path.join(path1, filename)
            dest_file = os.path.join(dest, filename)
            shutil.copy2(src_file, dest_file)

    # Get count of files in merged folder so far
    existing_files = os.listdir(dest)
    next_index = len(existing_files)

    # Copy from folder2 with renamed filenames
    if os.path.exists(path2):
        for filename in os.listdir(path2):
            src_file = os.path.join(path2, filename)
            new_filename = f"{next_index}.jpg"
            dest_file = os.path.join(dest, new_filename)
            shutil.copy2(src_file, dest_file)
            next_index += 1

print("✅ Merge complete!")
