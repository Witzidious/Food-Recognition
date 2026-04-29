import cv2
import os
from PIL import Image
import tensorflow as tf
from keras import layers

dataset_path = r"dataset"

valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
for split in ["train", "val"]:
    split_path = os.path.join(dataset_path, split)
    for class_name in os.listdir(split_path):
        class_path = os.path.join(split_path, class_name)
        for fname in os.listdir(class_path):
            fpath = os.path.join(class_path, fname)
            if os.path.getsize(fpath) == 0:
                print("Zero byte file:", fpath)
                continue
            img = cv2.imread(fpath)
            if img is None:
                print("Cannot read image:", fpath)
            
            if not fname.lower().endswith(valid_extensions):
                print(f"Invalid file format: {fpath}")
                # os.remove(fpath)
                continue
            try:
                img = Image.open(fpath)
                img.verify()
            except Exception as e:
                print(f"Corrupt image: {fpath} → {e}")
                # os.remove(fpath)import os

