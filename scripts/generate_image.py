import os
import sys

# Add the parent directory to Python's path
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# scripts/generate_images.py

import torch
from models.generator import ConvGenerator
import torchvision.transforms as transforms
from PIL import Image

# Initialize the generator model
generator = ConvGenerator()

# Define the latent dimension consisten with your trained model
latent_dim = 100

# Load trained model weights
generator.load_state_dict(torch.load('models/generator_final.pth'))
generator.eval()  # Set the model to evaluation mode

# Define transforms (adjust as per your training transforms)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Function to generate images
def generate_images(num_images):
    # Generate random input (replace with actual input generation logic)
    input_data = torch.randn(num_images, latent_dim)  # Example input data

    # Generate images
    with torch.no_grad():
        generated_images = generator(input_data)

    return generated_images

# Example usage to generate and save images
if __name__ == "__main__":
    num_images_to_generate = 10
    generated_images = generate_images(num_images_to_generate)

    # Save or display generated images
    for i, image in enumerate(generated_images):
        image = transforms.ToPILImage()(image.cpu().detach())
        image.save(f'generated_image_{i}.png')  # Save generated image
        # Alternatively, you can display the image using a library like Matplotlib
