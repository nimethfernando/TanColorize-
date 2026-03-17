import os

# مسیر to your dataset folder
folder_path = "Images Test"

# Supported image extensions
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

# Get all image files
files = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]

# Sort files (important for consistent numbering)
files.sort()

# Step 1: Rename to temporary names to avoid overwrite conflicts
temp_names = []
for i, file in enumerate(files):
    old_path = os.path.join(folder_path, file)
    temp_name = f"temp_{i}.tmp"
    temp_path = os.path.join(folder_path, temp_name)
    os.rename(old_path, temp_path)
    temp_names.append(temp_name)

# Step 2: Rename to final sequential names
for i, temp_name in enumerate(temp_names, start=1):
    temp_path = os.path.join(folder_path, temp_name)
    new_name = f"{i}.jpg"  # Change extension if needed
    new_path = os.path.join(folder_path, new_name)
    os.rename(temp_path, new_path)

print(f"Renamed {len(files)} images successfully.")