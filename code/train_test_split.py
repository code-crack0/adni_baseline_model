import os
import shutil
from sklearn.model_selection import train_test_split

def split_data(base_dir, output_dir, test_size=0.2, random_state=42):
    """
    Splits MRI scans into train and test directories.

    Parameters:
        base_dir (str): Directory containing `AD`, `CN`, and `MCI` subfolders.
        output_dir (str): Directory where train/test folders will be created.
        test_size (float): Proportion of data to use for testing.
        random_state (int): Random seed for reproducibility.
    """
    categories = ['AD', 'CN', 'MCI']  # Subfolders for each class
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for category in categories:
        category_dir = os.path.abspath(os.path.join(base_dir, category))
        if not os.path.exists(category_dir):
            print(f"Directory {category_dir} does not exist. Skipping...")
            continue

        images = [os.path.join(category_dir, img) for img in os.listdir(category_dir) if img.endswith(('.png'))]

        # Split into train and test sets
        train_images, test_images = train_test_split(images, test_size=test_size, random_state=random_state)

        # Create category folders in train and test directories
        train_category_dir = os.path.abspath(os.path.join(train_dir, category))
        test_category_dir = os.path.abspath(os.path.join(test_dir, category))
        os.makedirs(train_category_dir, exist_ok=True)
        os.makedirs(test_category_dir, exist_ok=True)

        # Move files
        for img in train_images:
            shutil.copy(img, os.path.join(train_category_dir, os.path.basename(img)))
        for img in test_images:
            shutil.copy(img, os.path.join(test_category_dir, os.path.basename(img)))

    print("Data has been split into train and test sets.")

# Example usage
base_dir = os.path.abspath("../processed_images")  # Directory containing `AD`, `CN`, and `MCI` subfolders
output_dir = os.path.abspath("../train_test_split")  # Directory where train/test folders will be created
split_data(base_dir, output_dir, test_size=0.2, random_state=42)