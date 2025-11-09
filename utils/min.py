import json

# Path to your JSON file
json_path = "train.json"

# Load JSON data
with open(json_path, "r") as f:
    data = json.load(f)

# Extract the list of images
images = data.get("images", [])

# Check if images list is not empty
if not images:
    print("No images found in the JSON file.")
else:
    # Find image with minimum width and height (using area as tie-breaker)
    min_image = min(images, key=lambda x: (x["width"], x["height"]))

    # Print result
    print("Image with minimum dimensions:")
    print(f"File name: {min_image['file_name']}")
    print(f"Width: {min_image['width']}")
    print(f"Height: {min_image['height']}")
