# complete_road_analysis_pipeline.py - ПОЛНЫЙ АНАЛИЗ ДОРОГ, ПОГОДЫ И ИНЦИДЕНТОВ
import os
import json
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
import time

# ============ КЛАСС ДЛЯ РАБОТЫ С LINZ API ============
class LinzMosaicDownloader:
    """Загрузчик мозаик из LINZ"""
    
    def __init__(self, api_key, layer_id=120064, output_folder="linz_mosaics"):
        self.api_key = api_key
        self.layer_id = layer_id
        self.output_folder = output_folder
        Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    def deg_to_tile(self, lat, lon, zoom):
        """Конвертирует градусы в номера тайлов XYZ"""
        import math
        n = 2.0 ** zoom
        xtile = int((lon + 180.0) / 360.0 * n)
        ytile = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
        return xtile, ytile
    
    def download_tile(self, zoom, x, y):
        """Скачивает один тайл"""
        url = f"https://tiles-cdn.koordinates.com/services;key={self.api_key}/tiles/v4/layer={self.layer_id}/EPSG:3857/{zoom}/{x}/{y}.png"
        headers = {'User-Agent': 'Mozilla/5.0'}
        
        try:
            response = requests.get(url, headers=headers, timeout=30)
            if response.status_code == 200:
                return Image.open(BytesIO(response.content))
        except Exception as e:
            print(f"   Ошибка: {e}")
        return None
    
    def download_mosaic(self, lat, lon, zoom, radius, name):
        """Скачивает мозаику вокруг точки"""
        center_x, center_y = self.deg_to_tile(lat, lon, zoom)
        size = radius * 2 + 1
        
        print(f"   Загрузка мозаики {size}x{size} для {name}...")
        
        first_tile = self.download_tile(zoom, center_x, center_y)
        if not first_tile:
            print(f"   ❌ Не удалось загрузить центральный тайл для {name}")
            return None
        
        w, h = first_tile.size
        mosaic = Image.new('RGB', (w * size, h * size))
        
        success_count = 0
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                x, y = center_x + dx, center_y + dy
                tile = self.download_tile(zoom, x, y)
                if tile:
                    mosaic.paste(tile, ((dx + radius) * w, (dy + radius) * h))
                    success_count += 1
        
        if success_count > 0:
            filename = f"{self.output_folder}/{name}_zoom{zoom}_r{radius}.png"
            mosaic.save(filename)
            print(f"   ✅ Сохранено: {filename} ({success_count}/{size*size} тайлов)")
            return filename
        return None


# ============ КЛАСС ДЛЯ ПОГОДНЫХ ДАННЫХ ============
class WeatherDataCollector:
    """Сбор погодных данных"""
    
    def __init__(self, lat=-35.726, lon=174.320):
        self.lat = lat
        self.lon = lon
    
    def get_weather_for_date_range(self, start_date, end_date):
        """Получение погоды за период"""
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "start_date": start_date,
            "end_date": end_date,
            "daily": [
                "precipitation_sum",
                "rain_sum",
                "snowfall_sum",
                "temperature_2m_max",
                "temperature_2m_min",
                "wind_speed_10m_max",
                "weathercode"
            ]
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if 'daily' in data:
            df = pd.DataFrame(data['daily'])
            df['date'] = pd.to_datetime(df['time'])
            return df
        return pd.DataFrame()


# ============ КЛАСС ДЛЯ ДАННЫХ О ДОРОГАХ ============
class RoadAnalyzer:
    """Анализ дорожного покрытия"""
    
    def __init__(self):
        self.surface_ranges = {
            'асфальт': {
                'lower': np.array([50, 50, 50]),
                'upper': np.array([130, 130, 130]),
                'color': (100, 100, 100)
            },
            'грунт': {
                'lower': np.array([60, 50, 40]),
                'upper': np.array([110, 90, 70]),
                'color': (139, 69, 19)
            },
            'песок': {
                'lower': np.array([180, 170, 130]),
                'upper': np.array([235, 215, 185]),
                'color': (255, 220, 150)
            }
        }
    
    def analyze_image(self, image_path):
        """Анализ одного изображения"""
        try:
            img = np.array(Image.open(image_path))
            if len(img.shape) == 2:
                img = np.stack([img, img, img], axis=2)
            elif img.shape[2] == 4:
                img = img[:, :, :3]
            
            total = img.shape[0] * img.shape[1]
            results = {}
            
            for name, ranges in self.surface_ranges.items():
                mask = cv2.inRange(img, ranges['lower'], ranges['upper'])
                results[name] = (np.sum(mask > 0) / total) * 100
            
            return results
        except Exception as e:
            print(f"   Ошибка: {e}")
            return {'асфальт': 0, 'грунт': 0, 'песок': 0}


# ============ КЛАСС ДЛЯ ИНЦИДЕНТОВ (ДАННЫЕ ОТ NZTA/Waka Kotahi) ============
class IncidentCollector:
    """Сбор данных об инцидентах"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key
    
    def get_road_events(self, lat, lon, radius_km=5, start_date=None, end_date=None):
        """
        Получение дорожных событий
        Источник: Waka Kotahi предоставляет данные о инцидентах через несколько API
        """
        # Здесь реальный API Waka Kotahi
        # Пока создаем заглушку с примерными данными для демонстрации
        
        events = []
        
        # Пример данных о ДТП по дням недели
        sample_events = [
            {"date": "2024-01-11", "type": "crash", "severity": "moderate", "clearance_time_min": 45},
            {"date": "2024-02-05", "type": "breakdown", "severity": "minor", "clearance_time_min": 20},
            {"date": "2024-02-21", "type": "crash", "severity": "minor", "clearance_time_min": 35},
            {"date": "2024-03-08", "type": "road_work", "severity": "moderate", "clearance_time_min": 120},
            {"date": "2024-03-24", "type": "crash", "severity": "severe", "clearance_time_min": 90},
        ]
        
        for event in sample_events:
            if start_date and event["date"] < start_date:
                continue
            if end_date and event["date"] > end_date:
                continue
            events.append(event)
        
        return events


# ============ ОСНОВНОЙ ПАЙПЛАЙН ============

class RoadCorrelationAnalyzer:
    """Анализатор корреляций погоды, инцидентов и дорог"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.linz_downloader = LinzMosaicDownloader(api_key)
        self.weather_collector = WeatherDataCollector()
        self.road_analyzer = RoadAnalyzer()
        self.incident_collector = IncidentCollector()
        self.results_dir = "analysis_results"
        Path(self.results_dir).mkdir(exist_ok=True)
    
    def process_location(self, name, lat, lon, zoom=18, radius=2, start_date="2024-01-01", end_date="2024-03-31"):
        """
        Полный анализ одного местоположения
        """
        print(f"\n{'='*60}")
        print(f"📍 АНАЛИЗ: {name} ({lat}, {lon})")
        print(f"{'='*60}")
        
        # 1. Скачиваем снимок
        print(f"\n📸 Шаг 1: Скачивание спутникового снимка...")
        image_path = self.linz_downloader.download_mosaic(lat, lon, zoom, radius, name)
        
        if not image_path:
            print(f"   ❌ Не удалось скачать снимок для {name}")
            return None
        
        # 2. Анализ дорожного покрытия
        print(f"\n🎨 Шаг 2: Анализ дорожного покрытия...")
        road_stats = self.road_analyzer.analyze_image(image_path)
        print(f"   Асфальт: {road_stats['асфальт']:.2f}%")
        print(f"   Грунт: {road_stats['грунт']:.2f}%")
        print(f"   Песок: {road_stats['песок']:.2f}%")
        
        # 3. Сбор погодных данных за период
        print(f"\n🌤️ Шаг 3: Сбор погодных данных...")
        weather_df = self.weather_collector.get_weather_for_date_range(start_date, end_date)
        avg_precip = weather_df['precipitation_sum'].mean()
        avg_temp = weather_df['temperature_2m_max'].mean()
        print(f"   Средние осадки: {avg_precip:.1f} мм")
        print(f"   Средняя температура: {avg_temp:.1f}°C")
        
        # 4. Сбор инцидентов
        print(f"\n🚗 Шаг 4: Сбор данных об инцидентах...")
        events = self.incident_collector.get_road_events(lat, lon, start_date=start_date, end_date=end_date)
        print(f"   Всего инцидентов: {len(events)}")
        
        # 5. Анализ времени устранения по дням недели
        print(f"\n📊 Шаг 5: Анализ времени устранения...")
        clearance_by_day = defaultdict(list)
        for event in events:
            date = datetime.strptime(event['date'], '%Y-%m-%d')
            day_name = date.strftime('%A')
            clearance_by_day[day_name].append(event['clearance_time_min'])
        
        avg_clearance = {}
        for day, times in clearance_by_day.items():
            avg_clearance[day] = np.mean(times)
            print(f"   {day}: {np.mean(times):.0f} мин (на основе {len(times)} событий)")
        
        return {
            'name': name,
            'coordinates': (lat, lon),
            'image_path': image_path,
            'road_stats': road_stats,
            'weather': {'avg_precipitation': avg_precip, 'avg_temperature': avg_temp},
            'incidents': {'total': len(events), 'by_day': clearance_by_day},
            'avg_clearance_time': avg_clearance
        }
    
    def process_location_list(self, locations, zoom=18, radius=2):
        """
        Анализ нескольких местоположений
        """
        results = []
        
        for name, (lat, lon) in locations.items():
            result = self.process_location(name, lat, lon, zoom, radius)
            if result:
                results.append(result)
        
        return results
    
    def generate_correlation_report(self, results):
        """
        Генерация отчета о корреляциях
        """
        print(f"\n{'='*60}")
        print("📊 КОРРЕЛЯЦИОННЫЙ АНАЛИЗ")
        print(f"{'='*60}")
        
        df = pd.DataFrame([{
            'location': r['name'],
            'asphalt_pct': r['road_stats']['асфальт'],
            'dirt_pct': r['road_stats']['грунт'],
            'sand_pct': r['road_stats']['песок'],
            'precipitation': r['weather']['avg_precipitation'],
            'temperature': r['weather']['avg_temperature'],
            'incident_count': r['incidents']['total']
        } for r in results])
        
        print("\n📋 Сводная таблица:")
        print(df.to_string(index=False))
        
        # Сохраняем результаты
        df.to_csv(f"{self.results_dir}/correlation_data.csv", index=False)
        
        # Визуализация
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Осадки vs Асфальт
        axes[0, 0].bar(df['location'], df['precipitation'], alpha=0.6, label='Осадки (мм)')
        axes[0, 0].set_ylabel('Осадки (мм)')
        axes[0, 0].set_title('Осадки по локациям')
        axes[0, 0].tick_params(rotation=45)
        
        # 2. Асфальт vs Инциденты
        axes[0, 1].bar(df['location'], df['asphalt_pct'], alpha=0.6, color='gray', label='Асфальт (%)')
        axes[0, 1].set_ylabel('Асфальт (%)')
        axes[0, 1].set_title('Процент асфальта')
        axes[0, 1].tick_params(rotation=45)
        
        # 3. Инциденты по локациям
        axes[1, 0].bar(df['location'], df['incident_count'], alpha=0.6, color='red')
        axes[1, 0].set_ylabel('Количество инцидентов')
        axes[1, 0].set_title('Инциденты за период')
        axes[1, 0].tick_params(rotation=45)
        
        # 4. Корреляция осадки-асфальт
        axes[1, 1].scatter(df['precipitation'], df['asphalt_pct'], s=100, alpha=0.6)
        for _, row in df.iterrows():
            axes[1, 1].annotate(row['location'], (row['precipitation'], row['asphalt_pct']))
        axes[1, 1].set_xlabel('Осадки (мм)')
        axes[1, 1].set_ylabel('Асфальт (%)')
        axes[1, 1].set_title('Корреляция: осадки → асфальт')
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/correlation_report.png", dpi=150)
        plt.show()
        
        return df


# ============ ЗАПУСК ============

if __name__ == "__main__":
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    API_KEY = "a409431e01c2452c8ff46173aecec35c"
    
    # Определяем локации для анализа
    # Можно менять координаты для разных точек
    locations = {
        "Whangarei_Center": (-35.726, 174.320),
        "Whangarei_Port": (-35.750, 174.330),
        "Whangarei_South": (-35.740, 174.315),
        "Whangarei_North": (-35.710, 174.325),
        # Добавьте другие города при необходимости
        # "Kerikeri": (-35.227, 173.948),
        # "Paihia": (-35.283, 174.091),
    }
    
    # Создаем анализатор
    analyzer = RoadCorrelationAnalyzer(API_KEY)
    
    # Анализируем все локации
    results = analyzer.process_location_list(
        locations=locations,
        zoom=18,
        radius=1  # 3x3 тайла для быстрого теста
    )
    
    # Генерируем отчет
    if results:
        df = analyzer.generate_correlation_report(results)
        
        print("\n" + "="*60)
        print("✅ АНАЛИЗ ЗАВЕРШЕН!")
        print("="*60)
        print(f"\n📁 Результаты сохранены в папку: {analyzer.results_dir}")
        print("   • correlation_data.csv - данные для анализа")
        print("   • correlation_report.png - визуализация")
        print("\n📸 Отдельные снимки локаций сохранены в: linz_mosaics/")
    else:
        print("\n❌ Не удалось проанализировать ни одну локацию")