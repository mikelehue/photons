#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: counts.py
Description: This script counts the numbers of photons in a picture taken by a sCMOS camera.
Author: Miguel Lopez Varga
Fecha de creación: 2025-04-25
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
from scipy.ndimage import find_objects

# First, we load the image and select the region of interest (ROI) to analyze, a 250x250 pixel square in the center of the image.
img = np.load("dark1.npy")  # Load the image from a .npy file
# Define the region of interest (ROI) size
roi_size = 250
# Calculate the center of the image
center_x, center_y = img.shape[1] // 2, img.shape[0] // 2
# Define the ROI coordinates
roi_x_start = center_x - roi_size // 2
roi_x_end = center_x + roi_size // 2
roi_y_start = center_y - roi_size // 2
roi_y_end = center_y + roi_size // 2
# Extract the ROI from the labeled array
roi = img[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

# Now I would like to check how many pixels are above a certain threshold
# Let's set a threshold value. This value can be adjusted based on the image characteristics.
threshold = 110  # Example threshold value
# Count the number of pixels above the threshold
num_pixels_above_threshold = np.sum(roi > threshold)
print(f"Number of pixels above the threshold ({threshold}): {num_pixels_above_threshold}")

# I would like to keep only the pixels above the threshold that are in groups of 9 or more pixels.
# This is done by labeling the connected components in the binary image.
# Create a binary image where pixels above the threshold are set to 1, and others are set to 0
binary_image = (roi > threshold).astype(int)
# Label the connected components in the binary image
labeled_array, num_features = label(binary_image)
print(f"Number of features (connected components) found: {num_features}")
# Find the slices of the labeled array that contain the features
slices = find_objects(labeled_array, num_features)
# Initialize a list to store the number of pixels in each feature
num_pixels_per_feature = []
# Loop through the slices and count the number of pixels in each feature
for s in slices:
    # Count the number of pixels in the feature
    num_pixels = np.sum(labeled_array[s] == labeled_array[s].max())
    num_pixels_per_feature.append(num_pixels)
# Filter the features based on the minimum size (9 pixels)
min_size = 9
filtered_features = [num for num in num_pixels_per_feature if num >= min_size]
print(f"Number of features with size >= {min_size}: {len(filtered_features)}")
# Calculate the total number of photons in the filtered features
total_photons = np.sum(filtered_features)
print(f"Total number of photons in features with size >= {min_size}: {total_photons}")
# Plot the original image and the binary image for visualization
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(roi, cmap='gray')
plt.title('Original Image (ROI)')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(binary_image, cmap='gray')
plt.title('Binary Image (Thresholded)')
plt.axis('off')
plt.tight_layout()
plt.show()
# Save the filtered features to a .npy file for further analysis
np.save("filtered_features.npy", filtered_features)
# Save the labeled array to a .npy file for further analysis

np.save("labeled_array.npy", labeled_array)
# Save the binary image to a .npy file for further analysis
np.save("binary_image.npy", binary_image)
# Save the original image to a .npy file for further analysis
np.save("original_image.npy", roi)

# I have taken some pictures with the camera and I would like to analyze them all at once.
# I will create a function that takes the image file name as input and returns the number of photons in the image.
# The pictures are in folders called: "dark", "laserOD6pt4847", "laserOD6pt8847", "laserODpt4847"
# The files inside the folders have 200 photos in one .dat for the dark and 100 for the laser in respectively .dat's.

# Let's open the .dat files and read the images, the function has to have as parameter the name of the folder, the name of the file, the size of the image and the number of images in the file:
# After reshaping the data into the image size, we have to take a ROI x>1200, x<1750, y>700, y<1300
# and then we have to take the average of the images in the file, and then we have to take the number of photons in the image.
# The function has to return the number of photons in the image and the average image.
# The function has to save the average image in a .npy file with the name of the folder and the name of the file.
# The function has to save the number of photons in a .npy file with the name of the folder and the name of the file.


def analyze_images(folder_name, file_name, image_size, num_images, threshold=117, min_size = 9):
    # Load the .dat file and read the images
    data = np.fromfile(f"{folder_name}/{file_name}", dtype=np.uint16)
    # Reshape the data into the image size
    images = data.reshape(num_images, image_size[0], image_size[1])
    # Define the ROI coordinates
    roi_x_start = 1000
    roi_x_end = 1500
    roi_y_start = 800
    roi_y_end = 1300
    # Extract the ROI from the images
    roi_images = images[:, roi_y_start:roi_y_end, roi_x_start:roi_x_end]    
    # We need to take de features of the images without taking the average
    # Create a binary image where pixels above the threshold are set to 1, and others are set to 0
    binary_images = (roi_images > threshold).astype(int)
    # Label the connected components in the binary images
    labeled_arrays = [label(binary_image)[0] for binary_image in binary_images]
    # Find the slices of the labeled arrays that contain the features
    slices = [find_objects(labeled_array) for labeled_array in labeled_arrays]
    # Initialize a list to store the number of pixels in each feature
    num_pixels_per_feature = []
    # Loop through the slices and count the number of pixels in each feature
    for labeled_array, slice_list in zip(labeled_arrays, slices):
        for s in slice_list:
            # Count the number of pixels in the feature
            num_pixels = np.sum(labeled_array[s] == labeled_array[s].max())
            num_pixels_per_feature.append(num_pixels)
    # Filter the features based on the minimum size (9 pixels)
    filtered_features = [num for num in num_pixels_per_feature if num >= min_size]
    print(f"Folder: {folder_name}, File: {file_name}")
    print(f"Mean number of features with size >= {min_size}: {len(filtered_features)/num_images}")
    # Calculate the total number of photons in the filtered features
    total_photons = np.sum(filtered_features)
    # Calculate the average image
    avg_image = np.mean(roi_images, axis=0)
    # Save the average image to a .npy file
    np.save(f"{folder_name}/{file_name}_avg_image.npy", avg_image)
    # Save the total number of photons to a .npy file
    np.save(f"{folder_name}/{file_name}_total_photons.npy", total_photons)
    # Return the total number of photons and the average image
    return total_photons, avg_image

# Let's analyze the images in the folders
folders = ["dark", "laserOD6pt4847", "laserOD6pt8847", "laserOD7pt4847"]
files = ["200DarkPhotos.dat", "100photos.dat", "100photos.dat", "100photos.dat"]
image_size = (2160, 2560)  # Image size (height, width)
num_images = [200, 100, 100, 100]  # Number of images in the .dat files
# Initialize a list to store the total number of photons for each folder
total_photons_list = []
# Loop through the folders and analyze the images
for folder, file, num in zip(folders, files, num_images):
    total_photons, avg_image = analyze_images(folder, file, image_size, num)
    total_photons_list.append(total_photons)
    # Plot the average image for each folder
    plt.figure(figsize=(6, 6))
    plt.imshow(avg_image, cmap='gray')
    plt.title(f'Average Image - {folder}/{file}')
    plt.axis('off')
    plt.show()
    # Print the total number of photons for each folder
    print(f"Total number of photons in {folder}/{file}: {total_photons}")
