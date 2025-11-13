from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset

CLASS_MAPPING = {
    'n02124075': 0,
    'n07753592': 1,
    'n02504458': 2,
    'n03792782': 3
}

# All possible class folders
CLASS_FOLDERS = ['n02124075', 'n02504458', 'n03792782', 'n07753592']


class ImageDataset(Dataset):
    """Custom Dataset for loading images"""
    def __init__(self, file_list_path, data_dir, transform=None, img_size=64):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.img_size = img_size
        self.missing_files = set()  # Track missing files

        with open(file_list_path, 'r') as f:
            self.image_files = [line.strip() for line in f if line.strip()]

        print(f"Loaded {len(self.image_files)} images from {file_list_path}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]

        class_id = img_name.split('_')[0]
        label = CLASS_MAPPING[class_id]

        img_path = None
        for folder in CLASS_FOLDERS:
            potential_path = self.data_dir / folder / img_name
            if potential_path.exists():
                img_path = potential_path
                break

        if img_path and img_path.exists():
            try:
                image = Image.open(img_path).convert('RGB')
            except Exception as e:
                if img_name not in self.missing_files:
                    print(f"Error loading {img_path}: {e}")
                    self.missing_files.add(img_name)
                image = Image.new('RGB', (self.img_size, self.img_size))
        else:
            if img_name not in self.missing_files:
                self.missing_files.add(img_name)
            image = Image.new('RGB', (self.img_size, self.img_size))

        if self.transform:
            image = self.transform(image)

        return image, label

    def report_missing_files(self):
        """Report summary of missing files"""
        if self.missing_files:
            print(f"Warning: {len(self.missing_files)} image(s) could not be "
                  f"loaded (using blank images instead)")
            if len(self.missing_files) <= 10:
                print("Missing files:", list(self.missing_files))
            else:
                print("First 10 missing files:", list(self.missing_files)[:10])
