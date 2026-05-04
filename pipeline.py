# pipeline_fixed.py - ИСПРАВЛЕННАЯ ВЕРСИЯ (без двоеточий)
import ee
import requests
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import re

# ============ 1. ИНИЦИАЛИЗАЦИЯ ============
PROJECT_ID = "200722038057"

try:
    ee.Initialize(project=PROJECT_ID)
    print(f"✅ Earth Engine инициализирован с проектом: {PROJECT_ID}")
except Exception as e:
    print(f"❌ Ошибка: {e}")
    print("Выполните ee.Authenticate() и повторите")
    exit(1)


def sanitize_filename(filename: str) -> str:
    """Очистка имени файла от недопустимых символов Windows"""
    # Заменяем двоеточия и другие недопустимые символы
    invalid_chars = r'[<>:"/\\|?*]'
    return re.sub(invalid_chars, '_', filename)


# ============ 2. ЗАГРУЗЧИК СНИМКОВ ДЛЯ НОВОЙ ЗЕЛАНДИИ ============

class NZSatelliteDownloader:
    def __init__(self):
        # Координаты Окленда
        self.auckland = ee.Geometry.Polygon([
            [174.6, -36.7],   # северо-запад
            [174.9, -36.7],   # северо-восток
            [174.9, -37.1],   # юго-восток
            [174.6, -37.1]    # юго-запад
        ])
    
    def download_images(self, start_date: str, end_date: str, 
                        max_cloud_cover: int = 30,
                        max_images: int = 10,
                        output_folder: str = "nz_satellite_images"):
        """
        Загрузка спутниковых снимков Landsat 8 для Окленда
        """
        print(f"\n🛰️  Загрузка снимков: {start_date} - {end_date}")
        
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        # Landsat 8 Collection 2 Level 2
        collection = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                      .filterBounds(self.auckland)
                      .filterDate(start_date, end_date)
                      .filterMetadata('CLOUD_COVER', 'less_than', max_cloud_cover)
                      .select(['SR_B4', 'SR_B3', 'SR_B2'])
                      )
        
        num_images = collection.size().getInfo()
        print(f"   Найдено снимков: {num_images}")
        print(f"   Облачность < {max_cloud_cover}%")
        
        # Ограничиваем количество
        download_count = min(num_images, max_images)
        image_list = collection.toList(download_count)
        
        downloaded_files = []
        
        for i in range(download_count):
            try:
                image = ee.Image(image_list.get(i))
                date = image.date().format().getInfo()
                
                # Нормализация для лучшего отображения (0-255)
                image = image.unitScale(0, 20000).multiply(255).uint8()
                
                # URL для скачивания
                url = image.getDownloadURL({
                    'scale': 30,
                    'region': self.auckland,
                    'format': 'PNG'
                })
                
                # Очищаем дату от недопустимых символов
                clean_date = sanitize_filename(date.replace('T', '_').replace(':', '-'))
                
                # Используем только дату (без времени) для простоты
                date_only = clean_date.split('_')[0] if '_' in clean_date else clean_date
                
                output_path = f"{output_folder}/auckland_{date_only}_{i}.png"
                
                # Скачиваем
                response = requests.get(url)
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                
                downloaded_files.append(output_path)
                print(f"   ✅ Снимок {i+1}/{download_count}: {date_only}")
                
            except Exception as e:
                print(f"   ❌ Ошибка {i}: {e}")
                continue
        
        print(f"\n✅ Загружено: {len(downloaded_files)} снимков")
        return downloaded_files


# ============ 3. АНАЛИЗ ИЗОБРАЖЕНИЙ ============

class RoadAnalyzer:
    """Анализ дорожного покрытия на спутниковых снимках"""
    
    def __init__(self):
        # Цветовые диапазоны для разных типов покрытия
        self.surface_ranges = {
            'асфальт': {
                'lower': np.array([70, 70, 70]),
                'upper': np.array([130, 130, 130])
            },
            'грунт': {
                'lower': np.array([60, 50, 40]),
                'upper': np.array([110, 90, 70])
            },
            'песок': {
                'lower': np.array([180, 170, 130]),
                'upper': np.array([235, 215, 185])
            }
        }
    
    def analyze(self, image_path: str) -> dict:
        """Анализ одного изображения"""
        try:
            img = np.array(Image.open(image_path))
            
            if len(img.shape) == 2:
                img = np.stack([img, img, img], axis=2)
            elif img.shape[2] == 4:
                img = img[:, :, :3]
            
            total_pixels = img.shape[0] * img.shape[1]
            results = {}
            
            for surface_name, ranges in self.surface_ranges.items():
                mask = cv2.inRange(img, ranges['lower'], ranges['upper'])
                percentage = (np.sum(mask > 0) / total_pixels) * 100
                results[surface_name] = percentage
            
            return results
        except Exception as e:
            print(f"   Ошибка анализа {image_path}: {e}")
            return {'асфальт': 0, 'грунт': 0, 'песок': 0}


# ============ 4. ПОГОДНЫЕ ДАННЫЕ ============

def get_weather_auckland(date: str) -> dict:
    """Получение погодных данных для Окленда"""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": -36.8485,
        "longitude": 174.7633,
        "start_date": date,
        "end_date": date,
        "daily": ["precipitation_sum", "temperature_2m_max", "wind_speed_10m_max"]
    }
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        if 'daily' in data and data['daily']:
            precip = data['daily'].get('precipitation_sum', [0])[0]
            temp = data['daily'].get('temperature_2m_max', [0])[0]
            wind = data['daily'].get('wind_speed_10m_max', [0])[0]
            
            return {
                'precipitation': precip if precip is not None else 0,
                'temperature': temp if temp is not None else 0,
                'wind_speed': wind if wind is not None else 0
            }
    except Exception as e:
        print(f"   Ошибка погоды: {e}")
    
    return {'precipitation': 0, 'temperature': 0, 'wind_speed': 0}


# ============ 5. ОСНОВНОЙ ПАЙПЛАЙН ============

def main():
    print("=" * 60)
    print("🚀 АНАЛИЗ ДОРОГ НОВОЙ ЗЕЛАНДИИ")
    print("=" * 60)
    
    # 1. Загрузка спутниковых снимков
    downloader = NZSatelliteDownloader()
    image_files = downloader.download_images(
        start_date='2024-01-01',
        end_date='2024-03-31',
        max_cloud_cover=30,
        max_images=5,  # Начните с 5 для теста
        output_folder='nz_satellite_images'
    )
    
    if not image_files:
        print("❌ Нет загруженных снимков")
        return
    
    # 2. Анализ
    analyzer = RoadAnalyzer()
    results = []
    
    for image_path in image_files:
        # Извлекаем дату из имени файла
        filename = Path(image_path).stem
        parts = filename.split('_')
        
        # Формируем дату
        if len(parts) >= 2:
            date_str = parts[1]
            # Приводим к формату YYYY-MM-DD
            if len(date_str) == 8:
                date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        else:
            date_str = "2024-01-01"
        
        print(f"\n📅 {date_str}")
        
        # Анализ дорог
        road_stats = analyzer.analyze(image_path)
        print(f"   Асфальт: {road_stats['асфальт']:.2f}%")
        print(f"   Грунт: {road_stats['грунт']:.2f}%")
        print(f"   Песок: {road_stats['песок']:.2f}%")
        
        # Погода
        weather = get_weather_auckland(date_str)
        print(f"   Осадки: {weather['precipitation']:.1f} мм")
        print(f"   Температура: {weather['temperature']:.1f}°C")
        
        results.append({
            'date': date_str,
            **road_stats,
            **weather
        })
    
    # 3. Сохранение результатов
    import pandas as pd
    df = pd.DataFrame(results)
    
    if len(df) > 0:
        df.to_csv('nz_road_analysis.csv', index=False)
        print(f"\n✅ Результаты сохранены в 'nz_road_analysis.csv'")
        print(df.to_string())
        
        # 4. Визуализация
        if len(df) > 1:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Осадки vs Асфальт
            axes[0].scatter(df['precipitation'], df['асфальт'], alpha=0.6, s=100)
            axes[0].set_xlabel('Осадки (мм)')
            axes[0].set_ylabel('Асфальт (%)')
            axes[0].set_title('Корреляция: осадки → асфальт')
            axes[0].grid(True, alpha=0.3)
            
            # Динамика
            axes[1].plot(df['date'], df['асфальт'], 'o-', label='Асфальт', linewidth=2, markersize=8)
            axes[1].plot(df['date'], df['precipitation'], 's-', label='Осадки', linewidth=2, markersize=8)
            axes[1].set_xlabel('Дата')
            axes[1].set_ylabel('Значение')
            axes[1].set_title('Динамика асфальта и осадков')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig('nz_analysis_chart.png', dpi=150)
            plt.show()
            
            print("✅ График сохранен в 'nz_analysis_chart.png'")
    else:
        print("❌ Нет данных для анализа")
    
    print("\n✅ АНАЛИЗ ЗАВЕРШЕН!")


if __name__ == "__main__":
    main()