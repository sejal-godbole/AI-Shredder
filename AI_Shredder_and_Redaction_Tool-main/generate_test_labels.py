import os
import csv

# Paths
IMAGE_DIR = 'data/images/'
CSV_PATH = 'data/test_labels.csv'

# Get list of image files (assuming .tif extension)
image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith('.tif')]

# Prepare CSV rows
rows = [['doc_id', 'label']]
for img_file in image_files:
    doc_id = os.path.splitext(img_file)[0]  # Get filename without extension
    label = 0  # Dummy label, you can change it later as needed
    rows.append([doc_id, label])

# Write CSV
with open(CSV_PATH, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(rows)

print(f"[INFO] Created {CSV_PATH} with {len(image_files)} entries.")
