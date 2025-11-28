import os
import warnings
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew, kurtosis, entropy
from skimage.feature import graycomatrix, graycoprops
from sklearn.metrics import classification_report, accuracy_score, jaccard_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def count_histogram_features(img, bins_num):
    hist = cv2.calcHist([img], [0], None, [bins_num], [0, 256])
    hist = hist.flatten()

    bin_width = 256 / bins_num
    bin_centers = np.array([i * bin_width + bin_width / 2 for i in range(bins_num)])
    hist_normalized = hist / np.sum(hist)

    weighted_data = np.repeat(bin_centers, hist.astype(int))

    features = {
        'mean': np.average(bin_centers, weights=hist),
        'variance': np.var(weighted_data) if len(weighted_data) > 0 else 0,
        'std_dev': np.std(weighted_data) if len(weighted_data) > 0 else 0,
        'skewness': skew(weighted_data) if len(weighted_data) > 0 else 0,
        'kurtosis': kurtosis(weighted_data) if len(weighted_data) > 0 else 0,
        'energy': np.sum(hist_normalized ** 2),
        'entropy': entropy(hist_normalized),
        'median': np.median(img),
        'percentile_25': np.percentile(img, 25),
        'percentile_75': np.percentile(img, 75),
        'min': np.min(img),
        'max': np.max(img),
        'contrast': np.max(img) - np.min(img)
    }

    return features, hist_normalized


image_folder = "kth_tips_grey_200x200/KTH_TIPS"
for i in range(3):
    texture_name = os.listdir(image_folder)[i]
    texture_path = os.path.join(image_folder, texture_name)

    image_files = [f for f in os.listdir(texture_path)]
    image_path = os.path.join(texture_path, image_files[10])
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    bins_count = 256
    features, hist = count_histogram_features(image, bins_num=bins_count)

    fig, (ax_img, ax_hist, ax_stats) = plt.subplots(1, 3, figsize=(15, 4))

    ax_img.imshow(image, cmap='gray')
    ax_img.set_title(texture_name)
    ax_img.axis('off')

    ax_hist.bar(range(bins_count), hist, alpha=0.7)
    ax_hist.set_title('Histogram')
    ax_hist.set_xlabel('Bins')
    ax_hist.set_ylabel('Frequency')

    stats_text = '\n'.join([f'{k}: {v:.2f}' for k, v in features.items()])
    ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes, verticalalignment='top')
    ax_stats.set_title('Features')
    ax_stats.axis('off')

    plt.tight_layout()
    plt.show()


def count_laws_texture_features(img):
    kernels = [
        ([1, 4, 6, 4, 1], 'L5'),
        ([-1, -2, 0, 2, 1], 'E5'),
        ([-1, 0, 2, 0, -1], 'S5'),
        ([-1, 2, 0, -2, 1], 'W5'),
        ([1, -4, 6, -4, 1], 'R5')
    ]

    features_dict = {}

    img = img.astype(np.float32)

    for k1, name1 in kernels:
        for k2, name2 in kernels:
            kernel = np.outer(k1, k2)
            filtered = cv2.filter2D(img, -1, kernel)
            energy = np.mean(filtered ** 2)
            features_dict[f"{name1}{name2}"] = energy

    tot = sum(features_dict.values())
    if tot > 0:
        for k in features_dict:
            features_dict[k] /= tot

    return features_dict


def count_glcm_features(img):
    img = (img // 8).astype(np.uint8)

    glcm = graycomatrix(img,
                        distances=[1],
                        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=32,
                        symmetric=True,
                        normed=True)

    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']

    features = {}

    for prop in properties:
        feature_val = np.mean(graycoprops(glcm, prop))
        features[prop] = feature_val

    return features


for i in range(3):
    texture_name = os.listdir(image_folder)[i]
    texture_path = os.path.join(image_folder, texture_name)

    image_files = [f for f in os.listdir(texture_path)]
    image_path = os.path.join(texture_path, image_files[10])
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    features = count_glcm_features(image)

    fig, (ax_img, ax_stats) = plt.subplots(1, 2, figsize=(12, 4))

    ax_img.imshow(image, cmap='gray')
    ax_img.set_title(texture_name)
    ax_img.axis('off')

    stats_text = '\n'.join([f'{k}: {v:.2f}' for k, v in features.items()])

    ax_stats.text(0.1, 0.9, stats_text, transform=ax_stats.transAxes, verticalalignment='top')
    ax_stats.set_title('GLCM Features')
    ax_stats.axis('off')

    plt.tight_layout()
    plt.show()


def load_data(img_folder, feature_type, num_images=50):
    features, labels = [], []

    for texture_name in os.listdir(img_folder):
        texture_path = os.path.join(img_folder, texture_name)
        if not os.path.isdir(texture_path):
            continue

        for image_file in os.listdir(texture_path)[:num_images]:
            img_path = os.path.join(texture_path, image_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is not None:
                img = cv2.resize(img, (128, 128))

                if feature_type == 'Histogram':
                    features_dict, _ = count_histogram_features(img, 256)
                elif feature_type == 'Laws':
                    features_dict = count_laws_texture_features(img)
                else:  # GLCM
                    features_dict = count_glcm_features(img)

                features.append(list(features_dict.values()))
                labels.append(texture_name)

    return np.array(features), labels


def train_models(x, y):
    y_encoded = LabelEncoder().fit_transform(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.25, random_state=42,
                                                        stratify=y_encoded)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    classifiers = {
        'KNN': KNeighborsClassifier(n_neighbors=3, n_jobs=-1),
        'SVM': SVC(kernel='rbf', random_state=42),
        'DecisionTree': DecisionTreeClassifier(random_state=42, max_depth=10)
    }

    models = {name: clf.fit(x_train_scaled, y_train) for name, clf in classifiers.items()}

    return models, x_test_scaled, y_test, scaler


trained_models_dict = {}

feature_types = ['Histogram', 'Laws', 'GLCM']
for feature_type in feature_types:
    print(f"Тренировка {feature_type}-моделей...")

    X, labels = load_data(image_folder, feature_type, 50)
    models, X_test, y_test, scaler = train_models(X, labels)

    trained_models_dict[feature_type] = {
        'models': models, 'scaler': scaler
    }

    print(f"  Data: {X.shape}")
    for name, model in models.items():
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            f1 = report['macro avg']['f1-score']

        print(f"  {name}: Acc={accuracy:.3f}, F1={f1:.3f}")
    print()


def execute_texture_segmentation(img, trained_models_dict, feature_type, model_name,
                                 window_size=32, step=None, n_classes=4):
    original_image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

    if step is None:
        step = window_size

    models = trained_models_dict[feature_type]['models']
    scaler = trained_models_dict[feature_type]['scaler']
    model = models[model_name]

    h, w = original_image.shape
    temp_seg_map = np.zeros((h, w), dtype=np.uint8)
    all_predictions = []

    feature_extractors = {
        'Histogram': lambda img: list(count_histogram_features(img, 32)[0].values()),
        'Laws': lambda img: list(count_laws_texture_features(img).values()),
        'GLCM': lambda img: list(count_glcm_features(img).values())
    }

    for y in range(0, h - window_size + 1, step):
        for x in range(0, w - window_size + 1, step):
            window = original_image[y:y + window_size, x:x + window_size]

            features = feature_extractors[feature_type](window)
            features_scaled = scaler.transform([features])
            prediction = model.predict(features_scaled)[0]

            temp_seg_map[y:y + window_size, x:x + window_size] = prediction
            all_predictions.append(prediction)

    unique, counts = np.unique(all_predictions, return_counts=True)
    top_classes = unique[np.argsort(counts)[-n_classes:]]

    segmentation_map = np.zeros_like(temp_seg_map)

    for new_class, old_class in enumerate(top_classes):
        segmentation_map[temp_seg_map == old_class] = new_class

    mask_not_top = ~np.isin(temp_seg_map, top_classes)
    if np.any(mask_not_top):
        segmentation_map[mask_not_top] = 1

    return segmentation_map, original_image


def show_segment(image, trained_models_dict, **kwargs):
    seg_map, original = execute_texture_segmentation(image, trained_models_dict, **kwargs)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

    ax1.imshow(original, cmap='gray')
    ax1.set_title('Исходное')
    ax1.axis('off')

    ax2.imshow(seg_map, cmap='tab10')
    ax2.set_title(f'Сегментация')
    ax2.axis('off')

    ax3.imshow(seg_map, cmap='tab10', alpha=0.8)
    ax3.imshow(original, cmap='gray', alpha=0.3)
    ax3.set_title('Оверлей')
    ax3.axis('off')

    plt.tight_layout()
    plt.show()


print('1) Laws_SVM')
show_segment("test_image1.png", trained_models_dict, window_size=12, feature_type='Laws', model_name='SVM')
print('1) GLCM_DecisionTree')
show_segment("test_image1.png", trained_models_dict, window_size=2, feature_type='GLCM', model_name='DecisionTree')


def execute_manual_segmentation(img):
    image_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(mask)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(final_mask, [largest_contour], 0, 1, -1)

    return final_mask


def remap_classes(mask, n_classes):
    unique, counts = np.unique(mask, return_counts=True)
    top_classes = unique[np.argsort(-counts)[:n_classes]]

    result = np.zeros_like(mask)
    for new_id, old_id in enumerate(top_classes):
        result[mask == old_id] = new_id

    return result


image_path = "test_image1.png"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

manual_mask = execute_manual_segmentation(image_rgb)
manual_mask = remap_classes(manual_mask, 4)

fig, (ax_orig, ax_segment) = plt.subplots(1, 2, figsize=(10, 5))

# Image
ax_orig.imshow(image_rgb)
ax_orig.set_title('Исходное')
ax_orig.axis('off')

# Segmentation results
ax_segment.imshow(manual_mask, cmap='tab10')
ax_segment.set_title('Ручная сегментация')
ax_segment.axis('off')

plt.show()


def find_best_mapping(true_mask, pred_mask):
    true_flat = true_mask.flatten()
    pred_flat = pred_mask.flatten()

    true_classes = np.unique(true_flat)
    pred_classes = np.unique(pred_flat)

    mapping = {}
    for pred_class in pred_classes:
        best_true_class = None
        best_overlap = -1

        for true_class in true_classes:
            overlap = np.sum((true_flat == true_class) & (pred_flat == pred_class))
            if overlap > best_overlap:
                best_overlap = overlap
                best_true_class = true_class

        mapping[pred_class] = best_true_class

    return mapping


def compare_segmentation(manual_mask, predicted_mask, original_img):
    if manual_mask.shape != predicted_mask.shape:
        predicted_mask = cv2.resize(predicted_mask, (manual_mask.shape[1], manual_mask.shape[0]))

    mapping = find_best_mapping(manual_mask, predicted_mask)
    pred_mapped = np.zeros_like(predicted_mask)

    for pred_class, true_class in mapping.items():
        pred_mapped[predicted_mask == pred_class] = true_class

    accuracy = accuracy_score(manual_mask.flatten(), pred_mapped.flatten())
    iou_scores = jaccard_score(manual_mask.flatten(), pred_mapped.flatten(),
                               average=None, labels=np.unique(manual_mask))

    print(f"Точность: {accuracy:.1%}")

    print("Предсказание по классам:")
    for i, class_id in enumerate(np.unique(manual_mask)):
        print(f"  Класс {class_id}: {iou_scores[i]:.3f}")

    # Visualization
    fig, (ax_orig, ax_manual, ax_pred) = plt.subplots(1, 3, figsize=(15, 5))

    ax_orig.imshow(original_img, cmap='gray')
    ax_orig.set_title('Исходное')
    ax_orig.axis('off')

    ax_manual.imshow(manual_mask, cmap='tab10')
    ax_manual.set_title('Ручная разметка')
    ax_manual.axis('off')

    ax_pred.imshow(pred_mapped, cmap='tab10')
    ax_pred.set_title('Предсказание')
    ax_pred.axis('off')

    plt.tight_layout()
    plt.show()


feature_type = 'GLCM'
model_name = 'DecisionTree'
print(f"{feature_type} - {model_name}\n")

predicted_mask, _ = execute_texture_segmentation(
    image_path,
    trained_models_dict,
    window_size=2,
    feature_type=feature_type,
    model_name=model_name
)

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
compare_segmentation(manual_mask, predicted_mask, image)
