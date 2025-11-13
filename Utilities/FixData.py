from pathlib import Path
import shutil
import random


def fix_dataset_files():
    """
    Scan actual image locations and create corrected train/test splits
    """
    data_dir = Path('datasets')
    class_folders = ['n02124075', 'n02504458', 'n03792782', 'n07753592']

    # Read current train and test files
    with open(data_dir / 'train.txt', 'r') as f:
        train_filenames = [line.strip() for line in f if line.strip()]

    with open(data_dir / 'test.txt', 'r') as f:
        test_filenames = [line.strip() for line in f if line.strip()]

    print(f"Original train.txt has {len(train_filenames)} entries")
    print(f"Original test.txt has {len(test_filenames)} entries")

    def find_actual_location(filename):
        """Find which class folder actually contains this image"""
        for class_folder in class_folders:
            class_path = data_dir / class_folder
            if (class_path / filename).exists():
                return class_folder
        return None

    corrected_train = []
    missing_train = []

    print("\nProcessing train.txt...")
    for filename in train_filenames:
        actual_location = find_actual_location(filename)
        if actual_location:
            base_name = filename.split('_', 1)[1]
            corrected_name = f"{actual_location}_{base_name}"
            corrected_train.append(corrected_name)
        else:
            missing_train.append(filename)

    corrected_test = []
    missing_test = []

    print("Processing test.txt...")
    for filename in test_filenames:
        actual_location = find_actual_location(filename)
        if actual_location:
            base_name = filename.split('_', 1)[1]
            corrected_name = f"{actual_location}_{base_name}"
            corrected_test.append(corrected_name)
        else:
            missing_test.append(filename)

    with open(data_dir / 'train.txt', 'w') as f:
        for filename in corrected_train:
            f.write(f"{filename}\n")

    with open(data_dir / 'test.txt', 'w') as f:
        for filename in corrected_test:
            f.write(f"{filename}\n")

    print("\nResults:")
    print(f"  Corrected train.txt: {len(corrected_train)} images")
    print(f"  Corrected test.txt: {len(corrected_test)} images")
    print(f"  Missing from train: {len(missing_train)} images")
    print(f"  Missing from test: {len(missing_test)} images")

    print("\nClass distribution in corrected train.txt:")
    for class_folder in class_folders:
        count = sum(1 for f in corrected_train if f.startswith(class_folder))
        print(f"  {class_folder}: {count} images")

    print("\nClass distribution in corrected test.txt:")
    for class_folder in class_folders:
        count = sum(1 for f in corrected_test if f.startswith(class_folder))
        print(f"  {class_folder}: {count} images")

    if missing_train or missing_test:
        print("\nMissing files (first 10):")
        if missing_train:
            print(f"  Train: {missing_train[:10]}")
        if missing_test:
            print(f"  Test: {missing_test[:10]}")

    return corrected_train, corrected_test


def create_balanced_split():
    """
    Create a new balanced train/test split from all available images
    """
    data_dir = Path('datasets')
    class_folders = ['n02124075', 'n02504458', 'n03792782', 'n07753592']

    all_train = []
    all_test = []

    print("\nCreating balanced 80/20 split...")

    for class_folder in class_folders:
        class_path = data_dir / class_folder
        if class_path.exists():
            images = list(class_path.glob('*.JPEG'))
            print(f"  {class_folder}: {len(images)} images found")

            if len(images) > 0:
                random.shuffle(images)
                split_idx = int(0.8 * len(images))

                for img in images[:split_idx]:
                    all_train.append(img.name)
                for img in images[split_idx:]:
                    all_test.append(img.name)

    random.shuffle(all_train)
    random.shuffle(all_test)

    shutil.copy2(data_dir / 'train.txt', data_dir / 'train_backup.txt')
    shutil.copy2(data_dir / 'test.txt', data_dir / 'test_backup.txt')

    with open(data_dir / 'train.txt', 'w') as f:
        for filename in all_train:
            f.write(f"{filename}\n")

    with open(data_dir / 'test.txt', 'w') as f:
        for filename in all_test:
            f.write(f"{filename}\n")

    print("\nBalanced dataset created:")
    print(f"  train_balanced.txt: {len(all_train)} images")
    print(f"  test_balanced.txt: {len(all_test)} images")

    print("\nClass distribution in balanced train.txt:")
    for class_folder in class_folders:
        count = sum(1 for f in all_train if f.startswith(class_folder))
        print(f"  {class_folder}: {count} images")

    return all_train, all_test


if __name__ == "__main__":
    random.seed(42)

    print("=" * 60)
    print("REPLACING ORIGINAL DATASET FILES")
    print("=" * 60)

    balanced_train, balanced_test = create_balanced_split()

    print("\n" + "=" * 60)
    print("SUCCESS!")
    print("=" * 60)
    print("Original train.txt and test.txt have been replaced")
    print("Backup files saved as train_backup.txt and test_backup.txt")
    print("Your model will now train on all 4 classes properly!")
