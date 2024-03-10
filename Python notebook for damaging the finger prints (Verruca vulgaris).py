import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# As did for cut wounds, we define a function to simulate Verruca vulgaris regions on a single fingerprint image. But with a 
# different strategy for simulating random positions and with different shapes of Verruca vulgaris regions on fingerprint. 
# I chose 6 minimum pixel values in image and then centered different circles with random radii around that pixel value coordinates :) so they overlap and give us what we want

def simulate_verruca_vulgaris(image):
    
    # Calculating the border size to remove (0.5mm)
    border_size = int(0.5 * image.shape[0] / 25.4)  # Assuming 25.4 pixels per mm

    # Removing the borders (This was necessary because for example for some images like 101_5 it was problematic to do segmentation without cropping)
    cropped_image = image[border_size:-border_size, border_size:-border_size]

    # Finding the minimum pixel intensity and its location in the cropped image (the core of my strategy)
    min_intensity = np.min(cropped_image)
    min_intensity_indices = np.where(cropped_image == min_intensity)
    min_intensity_x, min_intensity_y = min_intensity_indices[1][0], min_intensity_indices[0][0]

    # Defining the parameters for multiple circular ablations
    num_circles = 10  # Number of circles
    min_radius = 10  # Minimum radius of circles
    max_radius = 20  # Maximum radius of circles
    center_range = 20  # Range for generating random center offsets

    # Generating random center offsets within a short range around the minimum intensity point
    center_offsets_x = np.random.randint(-center_range, center_range, size=num_circles)
    center_offsets_y = np.random.randint(-center_range, center_range, size=num_circles)

    # Generating random x and y coordinates for circle centers (According to the minimum intensities calculated before)
    center_x = min_intensity_x + center_offsets_x + np.random.randint(-10, 10, size=num_circles)
    center_y = min_intensity_y + center_offsets_y + np.random.randint(-10, 10, size=num_circles)

    # Generating random radii for the circles
    radii = np.random.randint(min_radius, max_radius, size=num_circles)

    # Creating a black mask with white circular regions for each ablation
    mask = np.zeros_like(cropped_image)
    for center_x, center_y, radius in zip(center_x, center_y, radii):
        cv2.circle(mask, (center_x, center_y), radius, (255, 255, 255), -1)

    # Smoothing the border of the circular ablations using Gaussian blur in order to make the damage more realistic 
    mask_smoothed = cv2.GaussianBlur(mask, (15, 15), 0)

    # Adding the smoothed circular regions to the original ROI
    ablated_image = cv2.add(cropped_image, mask_smoothed)

    return ablated_image

# Defining the main function to process all images in the dataset (exactly like the cut wound)

def process_dataset(input_dir, output_dir):
    
    os.makedirs(output_dir, exist_ok=True)    # Creating the output directory if it doesn't exist


    # Loop through each image in the input directory
    
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.tif'):  # Process only TIF images
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, file_name)

            # loading the image
        
            image = cv2.imread(input_path)
            
            if image is not None:
                
                result_image = simulate_verruca_vulgaris(image)     # Simulate Verruca vulgaris regions on the image by the function


                cv2.imwrite(output_path, result_image)             # Saving the result image in the dedicated folder

            else:
                print(f"Failed to read image: {file_name}")

# Defining the input and output directories on my computer
input_directory = r'C:\Users\PREDATOR\Desktop\Newest data science project\New folder\DB1_B' # Has to be changed on your computer 
output_directory = r'C:\Users\PREDATOR\Desktop\Newest data science project\New folder\Output_Verruca vulgaris' # has to be changed on your computer

# Process the dataset
process_dataset(input_directory, output_directory)
