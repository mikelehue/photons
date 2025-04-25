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

# First, we load the image
img = np.load("foto1.npy")  # Load the image from a .npy file

# Now we can plot the image to see what we are working with 
plt.imshow(img, cmap='gray')
plt.colorbar() # Add a colorbar to the plot