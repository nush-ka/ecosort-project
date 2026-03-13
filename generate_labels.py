import os

images_dir = "dataset/train/images"
labels_dir = "dataset/train/labels"

os.makedirs(labels_dir, exist_ok=True)

for file in os.listdir(images_dir):
    if file.endswith(".jpg"):
        name = file.lower()
        if "plastic" in name:
            class_id = 0
        elif "paper" in name:
            class_id = 1
        elif "metal" in name:
            class_id = 2
        else:
            continue

        label_path = os.path.join(labels_dir, file.replace(".jpg", ".txt"))

        # Full image bounding box
        with open(label_path, "w") as f:
            f.write(f"{class_id} 0.5 0.5 1 1\n")

print("Labels generated.")

