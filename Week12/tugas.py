import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time

from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# =========================================================
# PATH DATASET
# =========================================================

DATASET_PATH = r"C:\Users\user\Documents\Kampus\Semester 4\Pengolahan Citra Digital\Week12\dataset"

classes = [
    "botol",
    "buku",
    "mainan",
    "mug",
    "remote"
]

# =========================================================
# LOAD DATASET
# =========================================================

def load_dataset():

    images = []
    labels = []

    for label, cls in enumerate(classes):

        folder = os.path.join(DATASET_PATH, cls)

        if not os.path.exists(folder):
            print(f"Folder {folder} tidak ditemukan")
            continue

        for filename in os.listdir(folder):

            path = os.path.join(folder, filename)

            img = cv2.imread(path)

            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            images.append(gray)

            labels.append(label)

    return images, labels

# =========================================================
# FEATURE DETECTOR
# =========================================================

def get_detector(method="SIFT"):

    if method == "SIFT":
        return cv2.SIFT_create()

    elif method == "ORB":
        return cv2.ORB_create(nfeatures=1000)

# =========================================================
# FEATURE EXTRACTION
# =========================================================

def extract_features(image, method="SIFT"):

    detector = get_detector(method)

    start = time.time()

    keypoints, descriptors = detector.detectAndCompute(image, None)

    end = time.time()

    elapsed = end - start

    return keypoints, descriptors, elapsed

# =========================================================
# SHOW KEYPOINTS
# =========================================================

def show_keypoints(image, keypoints, title):

    output = cv2.drawKeypoints(
        image,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    plt.figure(figsize=(6,6))

    plt.imshow(output, cmap='gray')

    plt.title(title)

    plt.axis("off")

    plt.show()

# =========================================================
# BRUTE FORCE MATCHING
# =========================================================

def brute_force_matching(desc1, desc2, method="SIFT"):

    if method == "SIFT":
        matcher = cv2.BFMatcher(cv2.NORM_L2)

    else:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    matches = matcher.knnMatch(desc1, desc2, k=2)

    good_matches = []

    for m, n in matches:

        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    return good_matches

# =========================================================
# DRAW MATCHES
# =========================================================

def draw_matches(img1, kp1, img2, kp2, matches):

    result = cv2.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        matches,
        None,
        flags=2
    )

    plt.figure(figsize=(12,6))

    plt.imshow(result, cmap='gray')

    plt.title("Feature Matching")

    plt.axis("off")

    plt.show()

# =========================================================
# RANSAC HOMOGRAPHY
# =========================================================

def homography_ransac(kp1, kp2, matches):

    if len(matches) < 4:
        return None, None

    src_pts = np.float32([
        kp1[m.queryIdx].pt for m in matches
    ]).reshape(-1,1,2)

    dst_pts = np.float32([
        kp2[m.trainIdx].pt for m in matches
    ]).reshape(-1,1,2)

    H, mask = cv2.findHomography(
        src_pts,
        dst_pts,
        cv2.RANSAC,
        5.0
    )

    return H, mask

# =========================================================
# EXTRACT ALL DESCRIPTORS
# =========================================================

def extract_all_descriptors(images, method="SIFT"):

    descriptor_list = []

    for img in images:

        kp, desc, _ = extract_features(img, method)

        if desc is not None:
            descriptor_list.append(desc)

    return descriptor_list

# =========================================================
# BUILD VOCABULARY
# =========================================================

def build_vocabulary(descriptor_list, k=20):

    descriptors = np.vstack(descriptor_list)

    kmeans = KMeans(
        n_clusters=k,
        random_state=42
    )

    kmeans.fit(descriptors)

    return kmeans

# =========================================================
# BUILD HISTOGRAM
# =========================================================

def build_histogram(descriptors, kmeans):

    histogram = np.zeros(len(kmeans.cluster_centers_))

    clusters = kmeans.predict(descriptors)

    for c in clusters:
        histogram[c] += 1

    return histogram

# =========================================================
# BUILD BOVW DATASET
# =========================================================

def build_bovw_dataset(images, labels, method="SIFT", k=20):

    descriptor_list = extract_all_descriptors(images, method)

    kmeans = build_vocabulary(descriptor_list, k)

    X = []

    for img in images:

        kp, desc, _ = extract_features(img, method)

        hist = build_histogram(desc, kmeans)

        X.append(hist)

    X = np.array(X)

    y = np.array(labels)

    return X, y

# =========================================================
# PCA
# =========================================================

def apply_pca(X, components):

    pca = PCA(n_components=components)

    X_pca = pca.fit_transform(X)

    return X_pca

# =========================================================
# CLASSIFICATION
# =========================================================

def classification_knn(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)

    X_test = scaler.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=3)

    knn.fit(X_train, y_train)

    pred = knn.predict(X_test)

    acc = accuracy_score(y_test, pred)

    print("\nAccuracy :", acc)

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, pred))

    return acc

# =========================================================
# MAIN PROGRAM
# =========================================================

images, labels = load_dataset()

print("Jumlah gambar :", len(images))

if len(images) == 0:
    print("Dataset kosong")
    exit()

# =========================================================
# FEATURE EXTRACTION
# =========================================================

methods = ["SIFT", "ORB"]

for method in methods:

    print("\n================================")
    print("METHOD :", method)
    print("================================")

    kp, desc, elapsed = extract_features(
        images[0],
        method
    )

    print("Jumlah Keypoints :", len(kp))

    print("Waktu Ekstraksi :", elapsed)

    if desc is not None:
        print("Dimensi Descriptor :", desc.shape)

    show_keypoints(
        images[0],
        kp,
        f"{method} Keypoints"
    )

# =========================================================
# FEATURE MATCHING
# =========================================================

img1 = images[0]
img2 = images[1]

kp1, desc1, _ = extract_features(img1, "SIFT")
kp2, desc2, _ = extract_features(img2, "SIFT")

matches = brute_force_matching(
    desc1,
    desc2,
    "SIFT"
)

print("\nJumlah Good Matches :", len(matches))

draw_matches(
    img1,
    kp1,
    img2,
    kp2,
    matches
)

# =========================================================
# RANSAC
# =========================================================

H, mask = homography_ransac(
    kp1,
    kp2,
    matches
)

print("\nHomography Matrix")
print(H)

# =========================================================
# BOVW
# =========================================================

vocab_sizes = [10, 20, 30]

vocab_accuracy = []

for vocab in vocab_sizes:

    print("\n================================")
    print("VOCAB SIZE :", vocab)
    print("================================")

    X, y = build_bovw_dataset(
        images,
        labels,
        method="SIFT",
        k=vocab
    )

    acc = classification_knn(X, y)

    vocab_accuracy.append(acc)

# =========================================================
# PCA
# =========================================================

X, y = build_bovw_dataset(
    images,
    labels,
    method="SIFT",
    k=20
)

components_list = [2, 4, 8, 16]

pca_accuracy = []

for comp in components_list:

    if comp >= min(X.shape):
        continue

    print("\n================================")
    print("PCA COMPONENT :", comp)
    print("================================")

    X_pca = apply_pca(X, comp)

    acc = classification_knn(X_pca, y)

    pca_accuracy.append(acc)

# =========================================================
# GRAPH VOCABULARY
# =========================================================

plt.figure(figsize=(8,5))

plt.plot(vocab_sizes, vocab_accuracy, marker='o')

plt.xlabel("Vocabulary Size")

plt.ylabel("Accuracy")

plt.title("Accuracy vs Vocabulary Size")

plt.grid(True)

plt.show()

# =========================================================
# GRAPH PCA
# =========================================================

valid_components = components_list[:len(pca_accuracy)]

plt.figure(figsize=(8,5))

plt.plot(valid_components, pca_accuracy, marker='o')

plt.xlabel("PCA Components")

plt.ylabel("Accuracy")

plt.title("Accuracy vs PCA Components")

plt.grid(True)

plt.show()

print("\nProgram selesai")