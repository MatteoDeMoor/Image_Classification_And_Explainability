import shutil
import random
from pathlib import Path
from sklearn.model_selection import train_test_split

# Configuration
SRC_DIR    = Path("data/PlantVillage")
TARGET_DIR = Path("data")
TRAIN_FRAC = 0.8
VAL_FRAC   = 0.1

random.seed(42)

def make_splits():
    for class_dir in SRC_DIR.iterdir():
        if not class_dir.is_dir(): 
            continue
        
        # Create target subdirectories
        train_dir = TARGET_DIR / "train" / class_dir.name
        val_dir   = TARGET_DIR / "val"   / class_dir.name
        test_dir  = TARGET_DIR / "test"  / class_dir.name
        for d in (train_dir, val_dir, test_dir):
            d.mkdir(parents=True, exist_ok=True)

        # All images in this class
        images = list(class_dir.glob("*.*"))
        # First split: train vs temp
        train_imgs, tmp_imgs = train_test_split(images, train_size=TRAIN_FRAC, random_state=42)
        # Second split: validation vs test from the temp set
        val_size = VAL_FRAC / (1 - TRAIN_FRAC)
        val_imgs, test_imgs = train_test_split(tmp_imgs, train_size=val_size, random_state=42)

        # Copy files
        for img in train_imgs:
            shutil.copy(img, train_dir / img.name)
        for img in val_imgs:
            shutil.copy(img, val_dir   / img.name)
        for img in test_imgs:
            shutil.copy(img, test_dir  / img.name)

        print(f"Class `{class_dir.name}`: train={len(train_imgs)}, val={len(val_imgs)}, test={len(test_imgs)}")

if __name__ == "__main__":
    make_splits()
