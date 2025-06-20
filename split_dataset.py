#!/usr/bin/env python3
import shutil
import random
from pathlib import Path
from sklearn.model_selection import train_test_split

# Configuratie
SRC_DIR    = Path("data/PlantVillage")  # bron met subfolders per klasse
TARGET_DIR = Path("data")               # hier worden train/ val/ test/ aangemaakt
TRAIN_FRAC = 0.8
VAL_FRAC   = 0.1  # van de resterende 20% wordt VAL_FRAC gebruikt voor val, de rest is test

random.seed(42)

def make_splits():
    for class_dir in SRC_DIR.iterdir():
        if not class_dir.is_dir(): 
            continue
        
        # Maak target subdirs aan
        train_dir = TARGET_DIR / "train" / class_dir.name
        val_dir   = TARGET_DIR / "val"   / class_dir.name
        test_dir  = TARGET_DIR / "test"  / class_dir.name
        for d in (train_dir, val_dir, test_dir):
            d.mkdir(parents=True, exist_ok=True)

        # Alle afbeeldingen in deze klasse
        images = list(class_dir.glob("*.*"))
        # Eerste split: train vs tmp
        train_imgs, tmp_imgs = train_test_split(images, train_size=TRAIN_FRAC, random_state=42)
        # Tweede split: val vs test uit de tmp set
        val_size = VAL_FRAC / (1 - TRAIN_FRAC)
        val_imgs, test_imgs = train_test_split(tmp_imgs, train_size=val_size, random_state=42)

        # Kopieer
        for img in train_imgs:
            shutil.copy(img, train_dir / img.name)
        for img in val_imgs:
            shutil.copy(img, val_dir   / img.name)
        for img in test_imgs:
            shutil.copy(img, test_dir  / img.name)

        print(f"Class `{class_dir.name}`: train={len(train_imgs)}, val={len(val_imgs)}, test={len(test_imgs)}")

if __name__ == "__main__":
    make_splits()
