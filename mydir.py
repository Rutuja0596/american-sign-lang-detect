import os
import csv

# Path to the dataset directory
dataset_path = 'C:/Users/Rutuja/Desktop/signs2'
# Path to save the CSV file
csv_path = 'C:/Users/Rutuja/Desktop/sign-lang-detect/labels.csv'

# Create a list to hold filenames and class labels
data = []

# Traverse the dataset directory
for class_name in os.listdir(dataset_path):
    class_dir = os.path.join(dataset_path, class_name)
    if os.path.isdir(class_dir):
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            if os.path.isfile(img_path):
                data.append([img_name, class_name])
            else:
                print(f"Skipped non-file: {img_path}")
    else:
        print(f"Skipped non-directory: {class_dir}")

# Check if data list is populated
if not data:
    print("No data found. Please check the dataset directory structure.")
else:
    # Write the data to a CSV file
    with open(csv_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write header
        csvwriter.writerow(['filename', 'class'])
        # Write data rows
        csvwriter.writerows(data)
    
    print(f"CSV file '{csv_path}' created successfully.")
