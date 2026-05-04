import os
import json
from pathlib import Path

import joblib
import numpy as np

from train_surface_xgboost import extract_features


class RoadSurfaceAnalyzer:
    """
    Анализ дорожного покрытия по спутниковым снимкам
    с использованием XGBoost-модели.
    """

    def __init__(
        self,
        model_path="road_surface_xgboost.pkl",
        encoder_path="surface_label_encoder.pkl"
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Не найдена модель: {model_path}")

        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Не найден encoder: {encoder_path}")

        self.model = joblib.load(model_path)
        self.encoder = joblib.load(encoder_path)

        self.speed_factors = {
            "asphalt_good": 1.00,
            "asphalt_damaged": 0.75,
            "concrete": 0.95,
            "gravel": 0.85,
            "dirt": 0.70,
            "sand": 0.60,
        }

    def classify_image(self, image_path: str):
        features = extract_features(image_path).reshape(1, -1)

        pred_id = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]

        surface = self.encoder.inverse_transform([pred_id])[0]
        confidence = float(np.max(probabilities)) * 100

        return {
            "surface": surface,
            "confidence": round(confidence, 1),
            "speed_factor": self.speed_factors.get(surface, 1.0),
            "all_factors": {
                class_name: round(float(prob) * 100, 1)
                for class_name, prob in zip(self.encoder.classes_, probabilities)
            }
        }

    def analyze_batch(self, image_paths):
        print(f"\nАнализ {len(image_paths)} снимков...")

        results = []

        for path in image_paths:
            if not os.path.exists(path):
                print(f"Файл не найден: {path}")
                continue

            result = self.classify_image(path)
            result["image"] = path
            results.append(result)

            print(
                f"{Path(path).name}: "
                f"{result['surface']} "
                f"({result['confidence']}%) "
                f"speed_factor={result['speed_factor']}"
            )

        output_path = "surface_analysis_results.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\nРезультаты сохранены: {output_path}")

        return results


if __name__ == "__main__":
    analyzer = RoadSurfaceAnalyzer()

    test_images = [
        "data/satellite_images/point_000_zoom19.png"
    ]

    analyzer.analyze_batch(test_images)


#Как запускать
#python train_surface_xgboost.py
#появятся: road_surface_xgboost.pkl
#surface_label_encoder.pkl
#surface_analyzer.py    