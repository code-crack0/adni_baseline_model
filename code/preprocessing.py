import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import shutil
# Define paths
data_dir = "../adni/"  # Path to the MRI images
metadata_file = "../baseline.csv"  # Metadata CSV file
output_dir = "../processed_images/"  # Directory for processed images
os.makedirs(output_dir, exist_ok=True)

# Load metadata
metadata = pd.read_csv(metadata_file)
print(metadata.head())  # Inspect the first few rows

# # Plot Age Distribution
# plt.figure(figsize=(10, 6))
# plt.hist(metadata['Age'], bins=20, edgecolor='black')
# plt.title('Age Distribution')
# plt.xlabel('Age')
# plt.ylabel('Frequency')
# plt.show()

# # Plot Group Distribution
# plt.figure(figsize=(10, 6))
# metadata['Group'].value_counts().plot(kind='bar')
# plt.title('Group Distribution')
# plt.xlabel('Group')
# plt.ylabel('Count')
# plt.show()


# Create directories for each group
# groups = metadata['Group'].unique()
# for group in groups:
#     os.makedirs(os.path.join(output_dir, group), exist_ok=True)

# # Function to copy images to the respective group folder
# def copy_images(subject_id, group):
#     subject_folder = os.path.join(data_dir, subject_id)
#     if os.path.exists(subject_folder):
#         for root, dirs, files in os.walk(subject_folder):
#             for file in files:
#                 if file.endswith(('.png')):  
#                     src_file = os.path.join(root, file)
#                     dest_folder = os.path.join(output_dir, group)
#                     shutil.copy(src_file, dest_folder)

# # Iterate through the metadata and copy images
# for index, row in metadata.iterrows():
#     subject_id = row['Subject']
#     group = row['Group']
#     copy_images(subject_id, group)

# print("Images have been organized into respective group folders.")
# number of images in the AD,CN and MCI folders

ad_images = len(os.listdir(output_dir + 'AD'))
cn_images = len(os.listdir(output_dir + 'CN'))
mci_images = len(os.listdir(output_dir + 'MCI'))

print(f"Number of images in AD folder: {ad_images}")
print(f"Number of images in CN folder: {cn_images}")
print(f"Number of images in MCI folder: {mci_images}")

