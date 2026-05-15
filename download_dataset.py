import kagglehub
import os
import shutil

# Download latest version
print("Downloading dataset...")
path = kagglehub.dataset_download("feyzazkefe/trashnet")
print("Path to dataset files:", path)

# The dataset downloaded by kagglehub is usually in a cache directory.
# Let's move or copy it to the local 'dataset' folder so the main script can find it.
target_dir = "./dataset"

# In the case of trashnet, it usually contains a subfolder with the same name or just the class folders.
# Let's check what's inside.
content = os.listdir(path)
print("Content of downloaded path:", content)

# If 'dataset-original' or similar exists, we might need to go deeper.
# But usually, it has the classes.
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# Copy everything from path to target_dir
# Note: kagglehub path might be read-only or in a system location, so copy is safer.
print(f"Copying files to {target_dir}...")

# Check if there's a 'dataset-original' folder (common in Trashnet kaggle uploads)
source_path = path
if 'dataset-original' in content:
    source_path = os.path.join(path, 'dataset-original')
elif 'trashnet' in content:
     source_path = os.path.join(path, 'trashnet')

for item in os.listdir(source_path):
    s = os.path.join(source_path, item)
    d = os.path.join(target_dir, item)
    if os.path.isdir(s):
        if os.path.exists(d):
            shutil.rmtree(d)
        shutil.copytree(s, d)
    else:
        shutil.copy2(s, d)

print("Dataset is ready in ./dataset")
