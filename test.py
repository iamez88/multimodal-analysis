import os
from PIL import Image
import numpy as np

def load_image(filename):
    """
    Load an image from the specified training data directory.
    
    Parameters:
        filename (str): Name of the image file to load
        
    Returns:
        PIL.Image: Loaded image object
    """
    # Define the base directory path
    base_dir = r"C:\Users\ericw\OneDrive\Documents\Python Scripts\training_data"
    
    # Create the full path to the image
    image_path = os.path.join(base_dir, filename)
    
    # Check if the file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Load and return the image
    image = Image.open(image_path)
    return image

# Example usage
if __name__ == "__main__":
    try:
        # Replace 'example.jpg' with an actual image filename in your directory
        img = load_image('double_bottom_1.png')
        print(f"Image loaded successfully. Size: {img.size}, Mode: {img.mode}")
        
        # You can convert to numpy array if needed for further processing
        img_array = np.array(img)
        print(f"Image shape as array: {img_array.shape}")
        
        # Display the image (uncomment if running in an environment that supports display)
        # img.show()
        
    except Exception as e:
        print(f"Error: {e}")
