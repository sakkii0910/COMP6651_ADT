import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from PIL import Image, ImageEnhance

# ✅ Step 1: Load images + light augmentation (contrast)
def load_faces_with_contrast(path):
    images, labels = [], []
    label_map = {}
    current_label = 0

    for filename in sorted(os.listdir(path)):
        if filename.endswith('.jpg'):
            try:
                person_id = filename.split('_')[0]
                if person_id not in label_map:
                    label_map[person_id] = current_label
                    current_label += 1
                label = label_map[person_id]

                img_path = os.path.join(path, filename)
                img = Image.open(img_path).convert('L').resize((64, 64))

                # Original
                base = np.array(img).astype(np.float32) / 255.0
                images.append(base.flatten())
                labels.append(label)

                # Contrast-boosted image (Augmented)
                contrast_img = ImageEnhance.Contrast(img).enhance(1.3)
                contrast_array = np.array(contrast_img).astype(np.float32) / 255.0
                images.append(contrast_array.flatten())
                labels.append(label)
            except:
                continue
    return np.array(images), np.array(labels)

# ✅ Step 2: Load dataset from renamed ORL faces
dataset_path = r'C:\Users\saksh\OneDrive\Desktop\ADT\Project\orl-database-for-training-and-testing\renamed_faces'
X, y = load_faces_with_contrast(dataset_path)
print(f"✅ Loaded: {len(X)} images, {len(set(y))} classes")

# ✅ Step 3: Split dataset
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, stratify=y_temp, random_state=42)

# ✅ Step 4: Test different NMF sizes
val_accuracies = []
test_accuracies = []
components_list = list(range(10, 101, 10))

for n in components_list:
    nmf = NMF(n_components=n, init='nndsvda', max_iter=700, random_state=42)
    X_train_nmf = nmf.fit_transform(X_train)
    X_val_nmf = nmf.transform(X_val)
    X_test_nmf = nmf.transform(X_test)

    clf = SVC(kernel='poly', degree=3, C=10, gamma='scale')
    clf.fit(X_train_nmf, y_train)

    val_acc = accuracy_score(y_val, clf.predict(X_val_nmf))
    test_acc = accuracy_score(y_test, clf.predict(X_test_nmf))

    print(f"[n={n}]  Val: {val_acc*100:.2f}%,  Test: {test_acc*100:.2f}%")

    val_accuracies.append(val_acc * 100)
    test_accuracies.append(test_acc * 100)

# ✅ Step 5: Accuracy graph
plt.figure(figsize=(7, 5))
plt.plot(components_list, test_accuracies, marker='o', label='Test Accuracy')
plt.plot(components_list, val_accuracies, marker='x', linestyle='--', label='Validation Accuracy')
plt.xlabel('Number of Components (n)')
plt.ylabel('Accuracy (%)')
plt.title('NMF: Test & Val Accuracy vs Components')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.xticks(components_list)
plt.tight_layout()
plt.savefig("nmf_accuracy_line_plot.png")
plt.show()

# ✅ Step 6: Visualize best NMF parts
best_n = components_list[np.argmax(test_accuracies)]
print(f"\n🥇 Best NMF n_components = {best_n} with Test Accuracy = {max(test_accuracies):.2f}%")

nmf_final = NMF(n_components=best_n, init='nndsvda', max_iter=1000, random_state=42)
nmf_final.fit(X_train)
components = nmf_final.components_[:6]

fig, axes = plt.subplots(2, 3, figsize=(8, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(components[i].reshape((64, 64)), cmap='gray')
    ax.set_title(f'NMF Part {i+1}')
    ax.axis('off')
plt.suptitle(f"Top {best_n} NMF Components (Visual Parts)")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("nmf_face_parts_best_n.png")
plt.show()
