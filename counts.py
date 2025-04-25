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
img = np.load("foto1.npy")  # Load the image from a .npy file
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
threshold = 120  # Example threshold value
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