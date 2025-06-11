# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 16:40:45 2025


@author: Miguel Lopez Varga

This script is for calculate the power of your laser in order to obtain N photons per second.
"""
import numpy as np

# Parameters
h = 6.626e-34 # Plank costant [J.s]
c = 3e8 # speed of light [m/s]
lam = 808e-9 # Wavelenght [m]
N = 3600 # Number of photons per second
#%% Calculate the power needed to obtain N photons per second
P = N * h * c / lam

print('Wanted power:',P)

# %% Calculate the optical density (OD) for a given input power
# and the calculated power
inp = 8e-9 # Input power [W]
out = P # Output power [W]
T = out/inp # Transmission
print('Transmission T = ', T)
OD = -np.log10(T) # Optical Density
print('OD = ', OD)

# Calculate de amout of photons per second for a given input power
def photons_per_second(power, wavelength):
    """Calculate the number of photons per second for a given power and wavelength."""
    return power * (1 / (h * c / wavelength))
# Calculate the number of photons per second for the input power
N_inp = photons_per_second(inp, lam)
print('Input power:', inp)
print('Input photons per second:', N_inp)

# Calculate the power of the laser for a given number of photons per second and a OD
def power_for_photons(N, wavelength, OD):
    """Calculate the power needed to obtain N photons per second with a given optical density."""
    T = 10 ** (-OD)  # Transmission from OD
    return N * (h * c / wavelength) / T

# Calculate the power needed to obtain N photons per second with a given OD
N = 3600  # Number of photons per second
OD = 6.7393  # Optical Density
P_needed = power_for_photons(N, lam, OD)
print('Power needed to obtain', N, 'photons per second with OD =', OD, ':', P_needed)

# Calculate the amount of photons per pixel per second for a given power, OD and pixel area (in pixels squared)
def photons_per_pixel(power, wavelength, OD, pixel_area):
    """Calculate the number of photons per pixel per second for a given power, wavelength, OD and pixel area."""
    T = 10 ** (-OD)  # Transmission from OD
    return (power * T) * (1 / (h * c / wavelength)) / pixel_area

# Calculate the number of photons per pixel per second for the input power, OD and pixel area
pixel_area = 600**2  # Pixel area in m^2 (for example, 1.12e-6 m^2)
power_inp = 0.4183e-6  # Input power in W
N_inp_pixel = photons_per_pixel(power_inp, lam, OD, pixel_area)
print('Input power:', power_inp)
print('Input photons per pixel per second:', N_inp_pixel)

# Calculate the number of total photons per second for the input power, for two different ODs
# The function has to return three different values for the three different ODs
OD1 = 3.3705 # First Optical Density (NE20B-B  + FF01-810/10-25)
OD2 = 6.7511  # Second Optical Density (NE20B-B + NE20A-B + FF01-810/10-25)
OD3 = 8.6954  # Third Optical Density (NE20B-B + NE20A-B + NE10B-B + FF01-810/10-25)
def total_photons_per_second(power, wavelength, OD1, OD2, OD3):
    """Calculate the total number of photons per second for a given power, wavelength and three ODs."""
    T1 = 10 ** (-OD1)  # Transmission from OD1
    T2 = 10 ** (-OD2)  # Transmission from OD2
    T3 = 10 ** (-OD3)  # Transmission from OD3
    N1 = power * T1 * (1 / (h * c / wavelength))  # Photons per second for OD1
    N2 = power * T2 * (1 / (h * c / wavelength))  # Photons per second for OD2
    N3 = power * T3 * (1 / (h * c / wavelength))  # Photons per second for OD3
    return N1, N2, N3

# Open an empty file to append the results
# We need to save: (Angle of polarizers, Input power, Number of photons per second with OD1, Number of photons per second with OD2, Number of photons per second with OD3)
# The function has to append the results to the file and ask me what is the angle of the polarizers and the input power
filename='results.txt'
def append_results_to_file(angle, input_power, N1, N2, N3, filename):
    """Append the results to a file."""
    with open(filename, 'a') as file:
        file.write(f"{angle}, {input_power}, {N1}, {N2}, {N3}\n")

# Ask for the angle of the polarizers and the input power
angle = float(input("Enter the angle of the polarizers (in degrees): "))
input_power = float(input("Enter the input power (in W): "))
# Calculate the number of photons per second for the input power and the three ODs    
N1, N2, N3 = total_photons_per_second(input_power, lam, OD1, OD2, OD3)
# Append the results to the file
append_results_to_file(angle, input_power, N1, N2, N3)
# Print a message to confirm that the results have been saved
print(f"Results saved to {filename} with angle {angle} and input power {input_power} W.")

#%% Save the results to a CSV file
import pandas as pd
def save_results_to_csv(filename='results.csv'):
    """Save the results to a CSV file."""
    data = {
        'Angle (degrees)': [angle],
        'Input Power (W)': [input_power],
        'Photons per Second OD1': [N1],
        'Photons per Second OD2': [N2]
    }
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")
