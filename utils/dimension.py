from PIL import Image

# Load the image from file
# NOTE: Replace 'path/to/your/image.jpg' with the actual path
try:
    image_pil = Image.open('tester.jpeg')
    
    # Use the .size attribute for (Width, Height)
    width, height = image_pil.size
    
    # Use the .mode attribute to infer channels (e.g., 'RGB' = 3, 'L' = 1)
    if image_pil.mode in ('RGB', 'BGR', 'RGBA'):
        channels = 3
    elif image_pil.mode == 'L':
        channels = 1
    else:
        channels = 'Unknown' # Handle other modes like CMYK
        
    print(f"--- Pillow Dimensions ---")
    print(f"Width, Height: {image_pil.size}")
    print(f"Height (H): {height}")
    print(f"Width (W): {width}")
    print(f"Channels (C): {channels}")

except FileNotFoundError as e:
    print(f"Image not found: {e}")
except Exception as e:
    print(f"An error occurred: {e}")