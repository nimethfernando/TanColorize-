import os

def generate_meta(dataset_path, output_file):
    print(f"Scanning {dataset_path}...")
    with open(output_file, 'w') as f:
        for root, _, files in os.walk(dataset_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    # Write the filename (relative path logic handled by dataset usually)
                    # Standard BasicSR usually just wants the filename if dataroot is set
                    f.write(f"{file}\n")
    print(f"Created {output_file} with {len(open(output_file).readlines())} images.")

# Update these paths to match your real folders
train_folder = r"C:\Users\nimet\Documents\IIT\L6\FYP\Dataset\SkinTan"
val_folder   = r"C:\Users\nimet\Documents\IIT\L6\FYP\Dataset\Datase_Validation"

generate_meta(train_folder, "meta_train.txt")
generate_meta(val_folder, "meta_val.txt")