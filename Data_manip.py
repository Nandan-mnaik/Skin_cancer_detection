import random
import shutil
import os
import csv
from pathlib import Path

# Define constants
SEED = 42
TRAIN_RATIO = 0.8
VALIDATION_RATIO = 0.1
TEST_RATIO = 0.1

# Define file paths (adjust these to match your laptop's directory structure)
MAIN_FOLDER = Path("C:/Users/Admin/Desktop/Skin-cancer/data")
CSV_LOCATION = MAIN_FOLDER / "HAM10000_metadata.csv"
TRAIN_FOLDER = MAIN_FOLDER / "train"
TEST_FOLDER = MAIN_FOLDER / "test"
VALIDATION_FOLDER = MAIN_FOLDER / "validation"

# Define the two image subfolders
IMAGE_FOLDER_1 = MAIN_FOLDER / "HAM10000_images_part_1"
IMAGE_FOLDER_2 = MAIN_FOLDER / "HAM10000_images_part_2"

def create_directories():
    for folder in [TRAIN_FOLDER, TEST_FOLDER, VALIDATION_FOLDER]:
        for tumor_type in ['mel', 'nv', 'bkl', 'bcc', 'df', 'vasc', 'akiec']:
            (folder / tumor_type).mkdir(parents=True, exist_ok=True)

def split_data():
    random.seed(SEED)
    counts = {'train': 0, 'test': 0, 'validation': 0}

    if TRAIN_FOLDER.exists():
        print('Image manipulation not needed')
        return

    create_directories()

    with open(CSV_LOCATION, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            img_file = row['image_id']
            tumor_type = row['dx']
            
            rand_num = random.random()
            if rand_num < TRAIN_RATIO:
                destination = TRAIN_FOLDER
                counts['train'] += 1
            elif rand_num < TRAIN_RATIO + VALIDATION_RATIO:
                destination = VALIDATION_FOLDER
                counts['validation'] += 1
            else:
                destination = TEST_FOLDER
                counts['test'] += 1
            
            # Check both image folders for the source file
            source = IMAGE_FOLDER_1 / f"{img_file}.jpg"
            if not source.exists():
                source = IMAGE_FOLDER_2 / f"{img_file}.jpg"
            
            dest = destination / tumor_type / f"{img_file}.jpg"
            
            try:
                shutil.copy(source, dest)
            except FileNotFoundError:
                print(f"Warning: File not found: {source}")

    for split, count in counts.items():
        print(f"Number of {split} examples: {count}")

if __name__ == "__main__":
    split_data()
    print("Data splitting complete!")