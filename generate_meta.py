import os

def generate_meta_absolute(dataset_path, output_file):
    print(f"Scanning {dataset_path}...")
    with open(output_file, 'w') as f:
        for root, _, files in os.walk(dataset_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    # Write the absolute file path
                    full_path = os.path.join(root, file)
                    f.write(f"{full_path}\n")
    print(f"Created {output_file}")

# Update these paths to match your real dataset folders
train_folder = r"C:\Users\nimet\Documents\IIT\L6\FYP\Dataset\SkinTan"
val_folder = r"C:\Users\nimet\Documents\IIT\L6\FYP\Dataset\Datase_Validation"

# Generates the files in the basicsr/ directory
generate_meta_absolute(train_folder, "basicsr/dataset_train.txt")
generate_meta_absolute(val_folder, "basicsr/dataset_validation.txt")