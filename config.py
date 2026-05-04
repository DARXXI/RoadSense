# config.py - Конфигурация всего пайплайна
import os

# API ключи
LINZ_API_KEY = "a409431e01c2452c8ff46173aecec35c"
LINZ_LAYER_ID = 121752  # Auckland 0.075m Urban Aerial Photos

# NZTA API (требуется регистрация на opendata.nzta.govt.nz)
NZTA_API_KEY = os.environ.get("NZTA_API_KEY", "")

# Пути к данным
DATA_DIR = "data"
SATELLITE_DIR = os.path.join(DATA_DIR, "satellite_images")
SURFACE_DIR = os.path.join(DATA_DIR, "surface_analysis")
OUTPUT_DIR = "output"

# Параметры модели
V0 = 90  # максимальная скорость (км/ч)

# Коэффициенты покрытия (калиброваны по спутниковым снимкам)
K_SURF = {
    'asphalt': 1.00,
    'gravel': 0.92,
    'dirt': 0.85,
    'sand': 0.70,
}

# Коэффициенты погоды
K_WEATHER = {
    'fine': 1.00,
    'rain': 0.94,
    'heavy_rain': 0.88,
    'fog': 0.90,
}

# Коэффициенты времени суток
K_TIME = {
    'day': 1.00,
    'peak_hour': 0.90,
    'night': 0.95,
}

# Создание папок
for d in [DATA_DIR, SATELLITE_DIR, SURFACE_DIR, OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)