import os
from pathlib import Path
from typing import List, Tuple

import cv2
import joblib
import numpy as np

from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from xgboost import XGBClassifier


DATASET_DIR = "dataset"

MODEL_PATH = "road_surface_xgboost.pkl"
ENCODER_PATH = "surface_label_encoder.pkl"

IMAGE_SIZE = (256, 256)

CLASSES = [
    "asphalt_good",
    "asphalt_damaged",
    "concrete",
    "gravel",
    "dirt",
    "sand",
]


def center_crop(img, crop_ratio=0.75):
    h, w = img.shape[:2]

    new_h = int(h * crop_ratio)
    new_w = int(w * crop_ratio)

    y1 = (h - new_h) // 2
    x1 = (w - new_w) // 2

    return img[y1:y1 + new_h, x1:x1 + new_w]


def load_image(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Не удалось прочитать изображение: {image_path}")

    img = center_crop(img)
    img = cv2.resize(img, IMAGE_SIZE)

    return img


def extract_color_features(img_bgr: np.ndarray) -> List[float]:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    features = []

    for img in [img_rgb, img_hsv]:
        for channel in range(3):
            ch = img[:, :, channel]
            features.extend([
                float(np.mean(ch)),
                float(np.std(ch)),
                float(np.min(ch)),
                float(np.max(ch)),
            ])

    return features


def extract_edge_features(img_bgr: np.ndarray) -> List[float]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size

    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    return [
        float(edge_density),
        float(laplacian_var),
    ]


def extract_lbp_features(img_bgr: np.ndarray) -> List[float]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    radius = 2
    points = 8 * radius

    lbp = local_binary_pattern(
        gray,
        points,
        radius,
        method="uniform"
    )

    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, points + 3),
        range=(0, points + 2)
    )

    hist = hist.astype("float")
    hist /= hist.sum() + 1e-7

    return hist.tolist()


def extract_glcm_features(img_bgr: np.ndarray) -> List[float]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    gray_32 = (gray / 8).astype(np.uint8)

    glcm = graycomatrix(
        gray_32,
        distances=[1, 2, 4],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=32,
        symmetric=True,
        normed=True
    )

    props = [
        "contrast",
        "homogeneity",
        "energy",
        "correlation"
    ]

    features = []

    for prop in props:
        values = graycoprops(glcm, prop)
        features.extend(values.flatten().tolist())

    return [float(x) for x in features]


def extract_damage_features(img_bgr: np.ndarray) -> List[float]:
    """
    Признаки повреждённого асфальта:
    тёмные пятна, неоднородность, локальный контраст.
    """

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)

    dark_threshold = mean_intensity - 1.2 * std_intensity
    dark_mask = gray < dark_threshold

    dark_spot_ratio = np.sum(dark_mask) / dark_mask.size

    bright_threshold = mean_intensity + 1.2 * std_intensity
    bright_mask = gray > bright_threshold

    bright_spot_ratio = np.sum(bright_mask) / bright_mask.size

    local_contrast = std_intensity / (mean_intensity + 1e-7)

    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size

    return [
        float(mean_intensity),
        float(std_intensity),
        float(dark_spot_ratio),
        float(bright_spot_ratio),
        float(local_contrast),
        float(edge_density),
    ]


def extract_features(image_path: str) -> np.ndarray:
    img = load_image(image_path)

    features = []

    features.extend(extract_color_features(img))
    features.extend(extract_edge_features(img))
    features.extend(extract_lbp_features(img))
    features.extend(extract_glcm_features(img))
    features.extend(extract_damage_features(img))

    return np.array(features, dtype=np.float32)


def load_dataset(dataset_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    X = []
    y = []

    dataset_path = Path(dataset_dir)

    for class_name in CLASSES:
        class_dir = dataset_path / class_name

        if not class_dir.exists():
            print(f"Папка не найдена: {class_dir}")
            continue

        image_files = (
            list(class_dir.glob("*.jpg")) +
            list(class_dir.glob("*.jpeg")) +
            list(class_dir.glob("*.png")) +
            list(class_dir.glob("*.webp"))
        )

        print(f"{class_name}: найдено {len(image_files)} изображений")

        for image_path in image_files:
            try:
                features = extract_features(str(image_path))
                X.append(features)
                y.append(class_name)
            except Exception as e:
                print(f"Ошибка обработки {image_path}: {e}")

    if len(X) == 0:
        raise ValueError("Датасет пустой. Проверь папку dataset/")

    return np.array(X), np.array(y)


def train_model():
    print("Загрузка датасета...")

    X, y = load_dataset(DATASET_DIR)

    print(f"Всего изображений: {len(X)}")
    print(f"Размерность признаков: {X.shape[1]}")

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    print("Классы:", list(encoder.classes_))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    model = XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.03,

        subsample=0.85,
        colsample_bytree=0.85,

        min_child_weight=2,
        gamma=0.1,

        reg_lambda=1.0,
        reg_alpha=0.1,

        objective="multi:softprob",
        num_class=len(encoder.classes_),
        eval_metric="mlogloss",

        random_state=42,
        n_jobs=-1
    )

    print("Обучение XGBoost...")
    model.fit(X_train, y_train)

    print("Оценка модели...")
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))

    print(classification_report(
        y_test,
        y_pred,
        target_names=encoder.classes_
    ))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoder, ENCODER_PATH)

    print(f"Модель сохранена: {MODEL_PATH}")
    print(f"Кодировщик сохранён: {ENCODER_PATH}")


if __name__ == "__main__":
    train_model()