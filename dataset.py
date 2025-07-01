import os
import sys
import shutil
import kagglehub
from pathlib import Path

def download_dataset():
    try:
        # Create resources directory if it doesn't exist
        resources_dir = Path("resources")
        resources_dir.mkdir(exist_ok=True)
        
        print("Downloading Indian Sign Language dataset...")
        # Download latest version of the dataset
        cache_path = kagglehub.dataset_download("prathumarikeri/indian-sign-language-isl")
        
        print(f"Dataset downloaded to cache: {cache_path}")
        print("Copying dataset to resources folder...")
        
        # Copy all files from cache to resources
        for src in Path(cache_path).glob('*'):
            dst = resources_dir / src.name
            if src.is_file():
                shutil.copy2(src, dst)
            elif src.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
        
        print(f"\nDataset successfully copied to: {resources_dir.absolute()}")
        print("\nNext steps:")
        print("1. Run pre_processing.py to prepare the data")
        print("2. Run training.py to train the model")
        
        return True
    
    except Exception as e:
        print(f"Error downloading/copying dataset: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Make sure you have kagglehub installed: pip install kagglehub")
        print("2. Make sure you're authenticated with Kaggle:")
        print("   - Go to kaggle.com → Account → Create API Token")
        print("   - Download kaggle.json and place it in ~/.kaggle/")
        return False

if __name__ == "__main__":
    download_dataset() 