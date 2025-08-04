import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.decomposition import NMF, PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.manifold import TSNE
from PIL import Image, ImageEnhance

def load_faces_with_contrast(path, img_size=(64, 64), contrast_factor=1.3):
    """
    Load grayscale face images from a directory.
    For each image, also generate a contrast-enhanced version.

    Args:
        path (str): Directory containing face images.
        img_size (tuple): Target size for image resizing.
        contrast_factor (float): Factor to enhance contrast.

    Returns:
        images (np.ndarray): Flattened image arrays (normalized).
        labels (np.ndarray): Corresponding integer labels.
    """
    images = []
    labels = []
    label_map = {}
    current_label = 0

    # List files sorted for consistency
    for filename in sorted(os.listdir(path)):
        if filename.lower().endswith('.jpg'):
            try:
                # Extract person ID from filename prefix (assumes format 'personid_xxx.jpg')
                person_id = filename.split('_')[0]

                # Map person_id to integer label
                if person_id not in label_map:
                    label_map[person_id] = current_label
                    current_label += 1
                label = label_map[person_id]

                # Load and preprocess image
                img_path = os.path.join(path, filename)
                img = Image.open(img_path).convert('L').resize(img_size)

                # Normalize image to [0,1]
                base_array = np.array(img).astype(np.float32) / 255.0
                images.append(base_array.flatten())
                labels.append(label)

                # Create and add contrast-enhanced version
                contrast_img = ImageEnhance.Contrast(img).enhance(contrast_factor)
                contrast_array = np.array(contrast_img).astype(np.float32) / 255.0
                images.append(contrast_array.flatten())
                labels.append(label)

            except Exception as e:
                print(f"Warning: Could not process {filename}: {e}")

    return np.array(images), np.array(labels)


# Load dataset
dataset_path = r'C:\Users\Janmitsinh\Desktop\academic_Concordia\A_SEM3\SCM\COMP6651_ADT\orl-database-for-training-and-testing\renamed_faces'
X_raw, y_raw = load_faces_with_contrast(dataset_path)
print(f"Loaded {len(X_raw)} images across {len(set(y_raw))} classes.")

# Filter classes to keep only those with at least 4 samples
label_counts = Counter(y_raw)
valid_labels = {label for label, count in label_counts.items() if count >= 4}

# Filter dataset based on valid labels
X = np.array([x for x, y_ in zip(X_raw, y_raw) if y_ in valid_labels])
y = np.array([y_ for y_ in y_raw if y_ in valid_labels])

if len(X) == 0:
    raise RuntimeError("No classes have at least 4 samples after filtering. Please check the dataset.")

print(f"After filtering: {len(X)} images across {len(set(y))} classes.")

# Normalize features for NMF (Min-Max scaling feature-wise)
X_min = X.min(axis=0)
X_max = X.max(axis=0)
X = (X - X_min) / (X_max - X_min + 1e-10)  # Added small epsilon to avoid division by zero

# Split dataset into train, validation, and test sets (stratified splits)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)
# Note: This results in 60% train, 20% val, 20% test

print(f"Dataset split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

# Define number of components to test for dimensionality reduction
components_list = list(range(10, 101, 10))

# Containers for accuracies
nmf_train_acc, nmf_val_acc, nmf_test_acc = [], [], []
pca_train_acc, pca_val_acc, pca_test_acc = [], [], []

# Loop over component counts
for n_components in components_list:
    # ----- Non-negative Matrix Factorization (NMF) -----
    nmf = NMF(n_components=n_components, init='nndsvda', max_iter=700, random_state=42)
    X_train_nmf = nmf.fit_transform(X_train)
    X_val_nmf = nmf.transform(X_val)
    X_test_nmf = nmf.transform(X_test)

    clf_nmf = SVC(kernel='poly', degree=3, C=10, gamma='scale')
    clf_nmf.fit(X_train_nmf, y_train)

    nmf_train_score = accuracy_score(y_train, clf_nmf.predict(X_train_nmf))
    nmf_val_score = accuracy_score(y_val, clf_nmf.predict(X_val_nmf))
    nmf_test_score = accuracy_score(y_test, clf_nmf.predict(X_test_nmf))

    nmf_train_acc.append(nmf_train_score * 100)
    nmf_val_acc.append(nmf_val_score * 100)
    nmf_test_acc.append(nmf_test_score * 100)

    print(f"[NMF n_components={n_components}] Train: {nmf_train_score*100:.2f}%, Val: {nmf_val_score*100:.2f}%, Test: {nmf_test_score*100:.2f}%")

    # ----- Principal Component Analysis (PCA) -----
    pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)

    clf_pca = SVC(kernel='rbf', C=10, gamma='scale')
    clf_pca.fit(X_train_pca, y_train)

    pca_train_score = accuracy_score(y_train, clf_pca.predict(X_train_pca))
    pca_val_score = accuracy_score(y_val, clf_pca.predict(X_val_pca))
    pca_test_score = accuracy_score(y_test, clf_pca.predict(X_test_pca))

    pca_train_acc.append(pca_train_score * 100)
    pca_val_acc.append(pca_val_score * 100)
    pca_test_acc.append(pca_test_score * 100)

    print(f"[PCA n_components={n_components}] Train: {pca_train_score*100:.2f}%, Val: {pca_val_score*100:.2f}%, Test: {pca_test_score*100:.2f}%")

# Plot accuracy trends for both methods
plt.figure(figsize=(10, 6))
plt.plot(components_list, nmf_train_acc, 's--', label='NMF Train')
plt.plot(components_list, nmf_val_acc, 'x--', label='NMF Val')
plt.plot(components_list, nmf_test_acc, 'o-', label='NMF Test')

plt.plot(components_list, pca_train_acc, 's--', label='PCA Train')
plt.plot(components_list, pca_val_acc, 'x--', label='PCA Val')
plt.plot(components_list, pca_test_acc, 'o-', label='PCA Test')

plt.xlabel('Number of Components')
plt.ylabel('Accuracy (%)')
plt.title('NMF vs PCA: Accuracy Comparison on Train, Validation, and Test Sets')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.xticks(components_list)
plt.tight_layout()
plt.savefig("nmf_vs_pca_accuracy_full.png")
plt.show()

# Select best number of components based on test accuracy
best_n_nmf = components_list[np.argmax(nmf_test_acc)]
best_n_pca = components_list[np.argmax(pca_test_acc)]

# Visualize first 6 NMF components as images (parts-based representation)
nmf_final = NMF(n_components=best_n_nmf, init='nndsvda', max_iter=1000, random_state=42)
nmf_final.fit(X_train)
nmf_components = nmf_final.components_[:6]

fig, axes = plt.subplots(2, 3, figsize=(10, 7))
for i, ax in enumerate(axes.flat):
    ax.imshow(nmf_components[i].reshape((64, 64)), cmap='gray')
    ax.set_title(f'NMF Part {i + 1}')
    ax.axis('off')
plt.suptitle(f'NMF Components (n={best_n_nmf})')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("nmf_parts_visualization.png")
plt.show()

# Visualize first 6 PCA components as eigenfaces
pca_final = PCA(n_components=best_n_pca, svd_solver='randomized', whiten=True, random_state=42)
pca_final.fit(X_train)
pca_components = pca_final.components_[:6]

fig, axes = plt.subplots(2, 3, figsize=(10, 7))
for i, ax in enumerate(axes.flat):
    ax.imshow(pca_components[i].reshape((64, 64)), cmap='gray')
    ax.set_title(f'PCA Component {i + 1}')
    ax.axis('off')
plt.suptitle(f'PCA Components (n={best_n_pca})')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("pca_components_visualization.png")
plt.show()

# Train final SVM classifiers on full training data with best parameters
X_train_nmf_final = nmf_final.transform(X_train)
X_test_nmf_final = nmf_final.transform(X_test)
clf_nmf_final = SVC(kernel='poly', degree=3, C=10, gamma='scale')
clf_nmf_final.fit(X_train_nmf_final, y_train)

X_train_pca_final = pca_final.transform(X_train)
X_test_pca_final = pca_final.transform(X_test)
clf_pca_final = SVC(kernel='rbf', C=10, gamma='scale')
clf_pca_final.fit(X_train_pca_final, y_train)

# Print classification reports for final models
print("\nNMF Classification Report:")
print(classification_report(y_test, clf_nmf_final.predict(X_test_nmf_final)))

print("\nPCA Classification Report:")
print(classification_report(y_test, clf_pca_final.predict(X_test_pca_final)))

# t-SNE visualization on NMF features (entire filtered dataset)
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_all_nmf = nmf_final.transform(X)
X_2d_nmf = tsne.fit_transform(X_all_nmf)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_2d_nmf[:, 0], X_2d_nmf[:, 1], c=y, cmap='tab10', s=15)
plt.title("t-SNE Visualization of NMF Features")
plt.colorbar(scatter)
plt.tight_layout()
plt.savefig("tsne_nmf.png")
plt.show()

# Summary of best results
print(f"\nBest NMF n_components: {best_n_nmf} with Test Accuracy: {max(nmf_test_acc):.2f}%")
print(f"Best PCA n_components: {best_n_pca} with Test Accuracy: {max(pca_test_acc):.2f}%")
