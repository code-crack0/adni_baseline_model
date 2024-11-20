import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
import pandas as pd
from tqdm import tqdm
import os
from os import walk, path, makedirs

# Define the function to create the directory structure for the PNG files
def create_directory(filename, img_dir):
    dir_list = filename.rsplit(os.sep, 1)  # Split by '/' to handle directories

    name = dir_list[-1]  # Get the file name
    name = name.rsplit('.', 1)[0]  # Strip out the .nii extension

    dir_list = dir_list[:-1]  # Retain the original directory structure
    
    current_dir = img_dir
    for next_dir in dir_list:  # Recreate directory structure in PNG root
        current_dir = os.path.join(current_dir, next_dir)
        if not path.isdir(current_dir):
            makedirs(current_dir)

    for_csv = current_dir  # For CSV tracking
    return for_csv, os.path.join(current_dir, name)

# Function to process the NIfTI image and save the slices as PNG files
def make_images(destination, data):
    images = []
    for i in range(data.shape[0]):
        array = np.array(data[i, :, :].tolist())  # Get an axial slice from the volumetric data
        images.append(array)

    norm_images = []
    for array in images:
        max_element = np.amax(array)
        if max_element > 0:
            array = (array / max_element) * 255.0  # Normalize to 255
        norm_images.append(array)

    for img, i in zip(norm_images, range(len(norm_images))):
        im = Image.fromarray(img)
        if im.mode != 'L':
            im = im.convert('L')  # Convert to grayscale
        im.save(f"{destination}_{len(norm_images) - 1 - i}.png")  # Save bottom-to-top

# Function to iterate through the directory and process the files
def process_nii_to_png(nifti_dir, img_dir, output_csv):
    filenames = []

    for root, _, files in walk(nifti_dir):
        for file in files:
            if file.endswith('.nii'):
                filenames.append(path.join(root, file))

    name_list = []
    count_list = []

    print("Total number of .nii files found:", len(filenames))

    for name in tqdm(filenames):
        img = nib.load(name)  # Load the .nii image
        data = np.asanyarray(img.dataobj)  # Convert to numpy array
        for_csv, destination = create_directory(name, img_dir)  # Recreate directory
        make_images(destination, data)  # Convert and save PNGs
        name_list.append(for_csv)
        count_list.append(data.shape)

    # Write the details to a CSV file
    df = pd.DataFrame(data={"dir_name": name_list, "file_count": count_list})
    df.to_csv(output_csv, sep=',', index=False)
    print(f"CSV file saved as {output_csv}")

# Main execution function
if __name__ == '__main__':
    # Set the base directory where your .nii files are located
    nifti_dir = "../adni/"
    # Set the directory where the processed PNG files will be stored
    img_dir = "../processed_png_images/"
    # Output CSV file to track directory names and image slice counts
    output_csv = "mri_image_data.csv"
    
    # Ensure the image directory exists
    os.makedirs(img_dir, exist_ok=True)
    
    # Process all .nii files and convert them to PNG
    process_nii_to_png(nifti_dir, img_dir, output_csv)
