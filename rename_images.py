import os
import shutil

# ✅ Set your actual path
source_folder = r'C:\Users\saksh\OneDrive\Desktop\ADT\Project\orl-database-for-training-and-testing'
renamed_folder = os.path.join(source_folder, 'renamed_faces')
os.makedirs(renamed_folder, exist_ok=True)

# ✅ Set target: 40 people × 10 images = 400
num_people = 40
images_per_person = 10

all_files = sorted([f for f in os.listdir(source_folder) if f.endswith('.jpg')])
used_files = all_files[:num_people * images_per_person]

if len(used_files) < num_people * images_per_person:
    print("⚠ Not enough images. Found:", len(used_files))
else:
    person_id = 1
    img_count = 1
    for filename in used_files:
        new_name = f"{person_id}_{img_count}.jpg"
        src_path = os.path.join(source_folder, filename)
        dst_path = os.path.join(renamed_folder, new_name)
        shutil.copy(src_path, dst_path)
        img_count += 1
        if img_count > images_per_person:
            person_id += 1
            img_count = 1
    print(f"✅ Renaming done: 40 people × 10 images → renamed_faces")
