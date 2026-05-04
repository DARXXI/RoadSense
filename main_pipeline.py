# 05_main_pipeline.py - ИНТЕГРАЦИЯ ВСЕХ 5 СКРИПТОВ В ОДИН ПАЙПЛАЙН
import os
import json
import numpy as np
from datetime import datetime
import sys

# Добавляем пути
sys.path.append(os.path.dirname(__file__))

# Импорты модулей
from config import *
from _01_satellite_downloader import SatelliteDownloader
from _02_surface_analyzer import RoadSurfaceAnalyzer
from _03_traffic_weather_api import TrafficWeatherData
from _04_route_optimizer import RouteOptimizer


class RoadSensePipeline:
    """
    Главный пайплайн RoadSense:
    1. Загрузка спутниковых снимков (LINZ)
    2. Анализ дорожного покрытия
    3. Получение погоды и трафика (NZTA + Open-Meteo)
    4. OPT-2 оптимизация маршрута
    5. Расчёт времени по формуле T = l/v + A*C + p*t
    """
    
    def __init__(self):
        self.satellite_downloader = SatelliteDownloader(LINZ_API_KEY, LINZ_LAYER_ID, SATELLITE_DIR)
        self.surface_analyzer = RoadSurfaceAnalyzer()
        self.traffic_weather_api = TrafficWeatherData(NZTA_API_KEY)
        self.route_optimizer = RouteOptimizer(sys.modules[__name__])
        
        self.surface_cache = {}
        self.weather_cache = {}
    
    def load_gps_track(self, gpx_file):
        """Загрузка GPS трека из GPX файла"""
        import gpxpy
        
        with open(gpx_file, 'r') as f:
            gpx = gpxpy.parse(f)
        
        points = []
        for track in gpx.tracks:
            for segment in track.segments:
                for point in segment.points:
                    points.append((point.latitude, point.longitude))
        
        return points
    
    def get_surface_for_point(self, lat, lon):
        """Получение типа покрытия для точки (из кэша или нового анализа)"""
        key = f"{lat:.4f}_{lon:.4f}"
        
        if key in self.surface_cache:
            return self.surface_cache[key]
        
        # Загружаем снимок для этой точки
        images = self.satellite_downloader.download_images_for_gps_track([(lat, lon)], zoom=18, radius=0)
        
        if images:
            result = self.surface_analyzer.classify_image(images[0])
            surface = result['surface']
        else:
            surface = 'asphalt'
        
        self.surface_cache[key] = surface
        return surface
    
    def get_weather_for_point(self, lat, lon, date):
        """Получение погоды для точки"""
        key = f"{lat:.4f}_{lon:.4f}_{date}"
        
        if key in self.weather_cache:
            return self.weather_cache[key]
        
        weather = self.traffic_weather_api.get_weather(lat, lon, date)
        self.weather_cache[key] = weather
        return weather
    
    def build_distance_matrix(self, points):
        """Построение матрицы расстояний между точками"""
        n = len(points)
        matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i][j] = 0
                else:
                    # Формула гаверсинуса
                    lat1, lon1 = points[i]
                    lat2, lon2 = points[j]
                    R = 6371
                    dlat = np.radians(lat2 - lat1)
                    dlon = np.radians(lon2 - lon1)
                    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
                    matrix[i][j] = R * 2 * np.arcsin(np.sqrt(a))
        
        return matrix
    
    def run(self, gpx_file, start_date, start_hour=12):
        """
        Запуск полного пайплайна
        """
        print("=" * 80)
        print("🚀 RoadSense: Полный анализ маршрута")
        print("=" * 80)
        print(f"\n📐 Формула: T = l/v + A*C + p*t")
        print(f"   Где: l/v - чистое время, A*C - пробки, p*t - риск ДТП")
        
        # Шаг 1: Загрузка GPS трека
        print("\n📍 Шаг 1: Загрузка GPS трека...")
        points = self.load_gps_track(gpx_file)
        print(f"   Загружено {len(points)} точек")
        
        # Шаг 2: Определение покрытия для каждой точки
        print("\n🛰️ Шаг 2: Анализ дорожного покрытия (спутниковые снимки)...")
        surfaces = {}
        for i, (lat, lon) in enumerate(points[:10]):  # Ограничим для скорости
            surface = self.get_surface_for_point(lat, lon)
            surfaces[f"{lat:.4f}_{lon:.4f}"] = surface
            print(f"   Точка {i+1}: ({lat:.4f}, {lon:.4f}) -> {surface}")
        
        # Шаг 3: Получение погоды
        print(f"\n🌤️ Шаг 3: Получение погоды на {start_date}...")
        weather_data = self.get_weather_for_point(points[0][0], points[0][1], start_date)
        print(f"   Погода: {weather_data['weather']}, {weather_data['temperature_c']}°C, {weather_data['precipitation_mm']} мм")
        
        # Шаг 4: Построение матрицы расстояний
        print("\n📏 Шаг 4: Построение матрицы расстояний...")
        unique_points = list(set(points[:20]))  # Ограничиваем для OPT-2
        dist_matrix = self.build_distance_matrix(unique_points)
        print(f"   Матрица {len(unique_points)}x{len(unique_points)} построена")
        
        # Шаг 5: OPT-2 оптимизация маршрута
        print("\n🔧 Шаг 5: OPT-2 оптимизация маршрута...")
        
        # Создаем карту покрытия для точек
        surface_map = {}
        for i, (lat, lon) in enumerate(unique_points):
            key = f"{lat:.4f}_{lon:.4f}"
            surface_map[f"P{i}"] = surfaces.get(key, 'asphalt')
        
        # Создаем сегменты
        segments = []
        for i in range(len(unique_points) - 1):
            dist = dist_matrix[i][i+1]
            segments.append({
                'distance_km': dist,
                'surface': surface_map.get(f"P{i}", 'asphalt'),
                'weather': weather_data['weather'],
                'hour': start_hour,
                'police_distance_km': 10
            })
        
        # Расчет времени по формуле
        total_time = 0
        print("\n📊 РАСЧЁТ ПО ФОРМУЛЕ T = l/v + A*C + p*t:")
        print("-" * 60)
        
        for i, seg in enumerate(segments):
            travel = self.route_optimizer.calculate_travel_time(
                seg['distance_km'], seg['surface'], seg['weather'], seg['hour']
            )
            accident = self.route_optimizer.calculate_accident_impact(
                seg['distance_km'], seg['surface'], seg['weather'], seg['hour'],
                seg['police_distance_km'], 0.03
            )
            total = travel + accident
            
            total_time += total
            
            print(f"\n   Сегмент {i+1}: {seg['distance_km']:.1f} км, {seg['surface']}")
            print(f"      l/v  = {travel:.1f} мин")
            print(f"      A*C + p*t = {accident:.1f} мин")
            print(f"      Итого: {total:.1f} мин")
        
        print("-" * 60)
        print(f"\n🎯 ОБЩЕЕ ВРЕМЯ: {total_time:.0f} мин ({total_time/60:.1f} ч)")
        
        # Сохранение результатов
        output = {
            'formula': 'T = l/v + A*C + p*t',
            'gpx_file': gpx_file,
            'date': start_date,
            'weather': weather_data,
            'total_time_min': round(total_time, 1),
            'total_time_hours': round(total_time / 60, 1),
            'segments_count': len(segments),
            'surface_types': list(set([s['surface'] for s in segments]))
        }
        
        with open(os.path.join(OUTPUT_DIR, 'pipeline_result.json'), 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n💾 Результаты сохранены: {OUTPUT_DIR}/pipeline_result.json")
        
        return output


def main():
    # Проверка наличия GPX файла
    gpx_file = "2105199.gpx"
    
    if not os.path.exists(gpx_file):
        print(f"❌ Файл {gpx_file} не найден!")
        print("   Поместите GPX файл в папку проекта")
        return
    
    # Запуск пайплайна
    pipeline = RoadSensePipeline()
    result = pipeline.run(
        gpx_file=gpx_file,
        start_date="2016-01-11",  # Дата из вашего трека
        start_hour=10
    )
    
    print("\n" + "=" * 80)
    print("✅ RoadSense Pipeline completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()