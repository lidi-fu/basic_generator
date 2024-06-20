import os
from PIL import Image

# Constants and parameters
INPUT_DIR = 'C:/AI/portrait_artworks/data/met_portraits'  # Input directory with original images
OUTPUT_TRAIN_DIR = 'C:/AI/portrait_artworks/data/processed_train'  # Output directory for processed training images
OUTPUT_VALIDATION_DIR = 'C:/AI/portrait_artworks/data/processed_validation'  # Output directory for processed validation images
TARGET_SIZE = (128, 128)  # Target size for resizing images

def preprocess_images(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg"):  # Adjust based on your image format
            try:
                img = Image.open(os.path.join(input_dir, filename))
                img = img.resize(TARGET_SIZE)  # Resize image
                # Optionally, you can perform other transformations here (e.g., convert format, normalize pixels)
                img.save(os.path.join(output_dir, filename))  # Save processed image
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

# Preprocess train images
preprocess_images(os.path.join(INPUT_DIR, 'train'), OUTPUT_TRAIN_DIR)

# Preprocess validation images
preprocess_images(os.path.join(INPUT_DIR, 'validation'), OUTPUT_VALIDATION_DIR)

print("Image preprocessing completed.")
