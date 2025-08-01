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

# ✅ Load grayscale face images with added contrast-enhanced versions
def load_faces_with_contrast(path):
    images = []
    labels = []
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

                base = np.array(img).astype(np.float32) / 255.0
                images.append(base.flatten())
                labels.append(label)

                # Add contrast-enhanced version
                contrast_img = ImageEnhance.Contrast(img).enhance(1.3)
                contrast_array = np.array(contrast_img).astype(np.float32) / 255.0
                images.append(contrast_array.flatten())
                labels.append(label)

            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue

    return np.array(images), np.array(labels)

# ✅ Load dataset
dataset_path = r'C:\Users\Janmitsinh\Desktop\academic_Concordia\A_SEM3\SCM\COMP6651_ADT\orl-database-for-training-and-testing\renamed_faces'
X_raw, y_raw = load_faces_with_contrast(dataset_path)
print(f"✅ Loaded: {len(X_raw)} images, {len(set(y_raw))} classes")

# ✅ Filter out classes with fewer than 4 samples
label_counts = Counter(y_raw)
valid_labels = {label for label, count in label_counts.items() if count >= 4}
X = [x for x, y_ in zip(X_raw, y_raw) if y_ in valid_labels]
y = [y_ for y_ in y_raw if y_ in valid_labels]
X = np.array(X)
y = np.array(y)

if len(X) == 0:
    raise RuntimeError("❌ No classes have at least 4 samples. Please check the dataset.")
print(f"📊 After filtering: {len(X)} images, {len(set(y))} classes")

# ✅ Normalize features using Min-Max Scaling for NMF compatibility
X_min = X.min(axis=0)
X_max = X.max(axis=0)
X = (X - X_min) / (X_max - X_min + 1e-6)


# ✅ Train/Val/Test Split (Stratified)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)

# ✅ Prepare for experiment
components_list = list(range(10, 101, 10))
nmf_val_accuracies = []
nmf_test_accuracies = []
nmf_train_accuracies = []

pca_val_accuracies = []
pca_test_accuracies = []
pca_train_accuracies = []

# ✅ Loop through component sizes for NMF and PCA
for n in components_list:
    # ----- NMF -----
    nmf = NMF(n_components=n, init='nndsvda', max_iter=700, random_state=42)
    X_train_nmf = nmf.fit_transform(X_train)
    X_val_nmf = nmf.transform(X_val)
    X_test_nmf = nmf.transform(X_test)

    clf_nmf = SVC(kernel='poly', degree=3, C=10, gamma='scale')
    clf_nmf.fit(X_train_nmf, y_train)

    train_acc_nmf = accuracy_score(y_train, clf_nmf.predict(X_train_nmf))
    val_acc_nmf = accuracy_score(y_val, clf_nmf.predict(X_val_nmf))
    test_acc_nmf = accuracy_score(y_test, clf_nmf.predict(X_test_nmf))

    nmf_train_accuracies.append(train_acc_nmf * 100)
    nmf_val_accuracies.append(val_acc_nmf * 100)
    nmf_test_accuracies.append(test_acc_nmf * 100)

    print(f"[NMF n={n}] Train: {train_acc_nmf*100:.2f}%, Val: {val_acc_nmf*100:.2f}%, Test: {test_acc_nmf*100:.2f}%")

    # ----- PCA -----
    pca = PCA(n_components=n, svd_solver='randomized', whiten=True, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)

    clf_pca = SVC(kernel='rbf', C=10, gamma='scale')
    clf_pca.fit(X_train_pca, y_train)

    train_acc_pca = accuracy_score(y_train, clf_pca.predict(X_train_pca))
    val_acc_pca = accuracy_score(y_val, clf_pca.predict(X_val_pca))
    test_acc_pca = accuracy_score(y_test, clf_pca.predict(X_test_pca))

    pca_train_accuracies.append(train_acc_pca * 100)
    pca_val_accuracies.append(val_acc_pca * 100)
    pca_test_accuracies.append(test_acc_pca * 100)

    print(f"[PCA n={n}] Train: {train_acc_pca*100:.2f}%, Val: {val_acc_pca*100:.2f}%, Test: {test_acc_pca*100:.2f}%")

# ✅ Plot Accuracy Comparison
plt.figure(figsize=(10, 6))
plt.plot(components_list, nmf_test_accuracies, 'o-', label='NMF Test')
plt.plot(components_list, nmf_val_accuracies, 'x--', label='NMF Val')
plt.plot(components_list, nmf_train_accuracies, 's--', label='NMF Train')

plt.plot(components_list, pca_test_accuracies, 'o-', label='PCA Test')
plt.plot(components_list, pca_val_accuracies, 'x--', label='PCA Val')
plt.plot(components_list, pca_train_accuracies, 's--', label='PCA Train')

plt.xlabel('Number of Components')
plt.ylabel('Accuracy (%)')
plt.title('NMF vs PCA: Train, Val, Test Accuracy')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.xticks(components_list)
plt.tight_layout()
plt.savefig("nmf_vs_pca_accuracy_full.png")
plt.show()

# ✅ Visualize Top NMF Components
best_n_nmf = components_list[np.argmax(nmf_test_accuracies)]
nmf_final = NMF(n_components=best_n_nmf, init='nndsvda', max_iter=1000, random_state=42)
nmf_final.fit(X_train)
components_nmf = nmf_final.components_[:6]

fig, axes = plt.subplots(2, 3, figsize=(8, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(components_nmf[i].reshape((64, 64)), cmap='gray')
    ax.set_title(f'NMF Part {i+1}')
    ax.axis('off')
plt.suptitle(f"NMF Parts (n={best_n_nmf})")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("nmf_parts_visualization.png")
plt.show()

# ✅ Visualize Top PCA Components
best_n_pca = components_list[np.argmax(pca_test_accuracies)]
pca_final = PCA(n_components=best_n_pca, svd_solver='randomized', whiten=True, random_state=42)
pca_final.fit(X_train)
components_pca = pca_final.components_[:6]

fig, axes = plt.subplots(2, 3, figsize=(8, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(components_pca[i].reshape((64, 64)), cmap='gray')
    ax.set_title(f'PCA Comp {i+1}')
    ax.axis('off')
plt.suptitle(f"PCA Components (n={best_n_pca})")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("pca_components_visualization.png")
plt.show()

# ✅ Classification Reports
X_test_nmf = nmf_final.transform(X_test)
clf_nmf_final = SVC(kernel='poly', degree=3, C=10, gamma='scale').fit(nmf_final.transform(X_train), y_train)
print("\n📋 NMF Classification Report:")
print(classification_report(y_test, clf_nmf_final.predict(X_test_nmf)))

X_test_pca = pca_final.transform(X_test)
clf_pca_final = SVC(kernel='rbf', C=10, gamma='scale').fit(pca_final.transform(X_train), y_train)
print("\n📋 PCA Classification Report:")
print(classification_report(y_test, clf_pca_final.predict(X_test_pca)))

# ✅ t-SNE Visualization (NMF)
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_all_nmf = nmf_final.transform(X)
X_2d_nmf = tsne.fit_transform(X_all_nmf)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_2d_nmf[:, 0], X_2d_nmf[:, 1], c=y, cmap='tab10', s=15)
plt.title("t-SNE Visualization on NMF Features")
plt.colorbar(scatter)
plt.tight_layout()
plt.savefig("tsne_nmf.png")
plt.show()

# ✅ Summary
print(f"\n🥇 Best NMF n_components = {best_n_nmf} with Test Accuracy = {max(nmf_test_accuracies):.2f}%")
print(f"🥈 Best PCA n_components = {best_n_pca} with Test Accuracy = {max(pca_test_accuracies):.2f}%")
