import cv2
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator

# Mentioning the name of the picture
name = input("Enter the name of the person in the image: ")

# Prompt the user to enter the image path
image_path = input("Enter the path of the image: ")

# Load the image
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is None:
    print(f" Error: Unable to load image from {image_path}")
    exit()

# Convert image to RGB (OpenCV loads images in BGR format)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Create an ImageDataGenerator with various augmentations
datagen = ImageDataGenerator(
    rotation_range=20,    # Rotate image randomly within 20 degrees
    width_shift_range=0.2,  # Shift width
    height_shift_range=0.2, # Shift height
    brightness_range=[0.8, 1.2], # Change brightness
    zoom_range=0.2,  # Zoom in/out
    horizontal_flip=True, # Flip horizontally
    fill_mode='nearest'
)

# Convert image to array and reshape for the generator
image_array = np.expand_dims(image, axis=0)

# Create a directory to save augmented images
output_dir = f"dataset\\{name}"
os.makedirs(output_dir, exist_ok=True)

# Generate and save 50 augmented images
i = 0
for batch in datagen.flow(image_array, batch_size=1, save_to_dir=output_dir, save_prefix="suspect", save_format="jpg"):
    i += 1
    if i >= 50:  # Stop after generating 50 images
        break

print(f" Augmentation completed. Augmented images saved in: {output_dir}")
