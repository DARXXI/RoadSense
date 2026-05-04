# gpx_satellite_weather_analysis_fixed.py - ИСПРАВЛЕННАЯ ВЕРСИЯ
import gpxpy
import gpxpy.gpx
import folium
import requests
from PIL import Image
from io import BytesIO
import math
import os
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# ============ КОНФИГУРАЦИЯ ============
API_KEY = "a409431e01c2452c8ff46173aecec35c"
LAYER_ID = 121752  # Auckland 0.075m Urban Aerial Photos

GPX_FILE = "2105199.gpx"  # Ваш файл

class RouteAnalyzer:
    def __init__(self, gpx_file):
        self.gpx_file = gpx_file
        self.points = []
        self.load_gpx()
    
    def load_gpx(self):
        """Загрузка GPX файла"""
        print(f"📂 Загрузка GPX: {self.gpx_file}")
        
        with open(self.gpx_file, 'r') as f:
            gpx = gpxpy.parse(f)
        
        for track in gpx.tracks:
            for segment in track.segments:
                for point in segment.points:
                    self.points.append({
                        'lat': point.latitude,
                        'lon': point.longitude,
                        'time': point.time,
                        'elevation': point.elevation
                    })
        
        print(f"   ✅ Загружено {len(self.points)} точек")
        return self.points
    
    def get_route_bounds(self):
        """Получение границ маршрута"""
        lats = [p['lat'] for p in self.points]
        lons = [p['lon'] for p in self.points]
        
        return {
            'min_lat': min(lats), 'max_lat': max(lats),
            'min_lon': min(lons), 'max_lon': max(lons),
            'center_lat': sum(lats)/len(lats),
            'center_lon': sum(lons)/len(lons)
        }
    
    def sample_points_along_route(self, num_samples=15):
        """Выборка точек вдоль маршрута"""
        if len(self.points) <= num_samples:
            return self.points
        
        indices = np.linspace(0, len(self.points)-1, num_samples, dtype=int)
        return [self.points[i] for i in indices]
    
    def get_weather_for_point(self, lat, lon, date):
        """Погода для точки"""
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": date.strftime('%Y-%m-%d'),
            "end_date": date.strftime('%Y-%m-%d'),
            "hourly": ["temperature_2m", "precipitation", "wind_speed_10m"]
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if 'hourly' in data and data['hourly']:
                temps = [t for t in data['hourly']['temperature_2m'] if t is not None]
                precip = [p for p in data['hourly']['precipitation'] if p is not None]
                winds = [w for w in data['hourly']['wind_speed_10m'] if w is not None]
                
                return {
                    'temperature': sum(temps)/len(temps) if temps else None,
                    'precipitation': sum(precip)/len(precip) if precip else 0,
                    'wind_speed': sum(winds)/len(winds) if winds else None
                }
        except Exception as e:
            pass
        return None
    
    def create_map(self, output_file="gpx_track_map.html"):
        """Создание карты"""
        if not self.points:
            return
        
        center_lat = sum(p['lat'] for p in self.points) / len(self.points)
        center_lon = sum(p['lon'] for p in self.points) / len(self.points)
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=8)
        
        coordinates = [[p['lat'], p['lon']] for p in self.points]
        
        folium.PolyLine(coordinates, color='red', weight=3, opacity=0.8).add_to(m)
        
        folium.Marker([self.points[0]['lat'], self.points[0]['lon']], 
                      popup="Start", icon=folium.Icon(color='green')).add_to(m)
        folium.Marker([self.points[-1]['lat'], self.points[-1]['lon']], 
                      popup="End", icon=folium.Icon(color='red')).add_to(m)
        
        m.save(output_file)
        print(f"   ✅ Карта сохранена: {output_file}")


class SatelliteImageDownloader:
    """Загрузчик спутниковых снимков LINZ"""
    
    def __init__(self, api_key, layer_id):
        self.api_key = api_key
        self.layer_id = layer_id
        self.cache = {}
    
    def deg_to_tile(self, lat, lon, zoom):
        n = 2.0 ** zoom
        xtile = int((lon + 180.0) / 360.0 * n)
        ytile = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
        return xtile, ytile
    
    def download_tile(self, lat, lon, zoom=17):
        """Скачивание тайла для координат"""
        x, y = self.deg_to_tile(lat, lon, zoom)
        cache_key = f"{zoom}_{x}_{y}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        url = f"https://tiles-cdn.koordinates.com/services;key={self.api_key}/tiles/v4/layer={self.layer_id}/EPSG:3857/{zoom}/{x}/{y}.png"
        
        try:
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                self.cache[cache_key] = img
                return img
        except Exception as e:
            pass
        return None


class RoadSurfaceAnalyzer:
    """Анализ дорожного покрытия - ИСПРАВЛЕННАЯ ВЕРСИЯ"""
    
    def __init__(self):
        self.surface_colors = {
            'асфальт': ([50, 50, 50], [130, 130, 130]),
            'грунт': ([60, 50, 40], [110, 90, 70]),
            'песок': ([180, 170, 130], [235, 215, 185])
        }
    
    def _ensure_rgb(self, img):
        """Конвертирует изображение в RGB формат"""
        if img.mode == 'P':
            img = img.convert('RGB')
        elif img.mode == 'L':
            img = img.convert('RGB')
        elif img.mode == 'RGBA':
            img = img.convert('RGB')
        return img
    
    def analyze_image(self, img):
        """Анализ изображения - ИСПРАВЛЕННАЯ ВЕРСИЯ"""
        import cv2
        import numpy as np
        
        # Конвертируем в RGB
        img = self._ensure_rgb(img)
        
        # Конвертируем в numpy
        img_np = np.array(img)
        
        total = img_np.shape[0] * img_np.shape[1]
        results = {}
        
        for name, (lower, upper) in self.surface_colors.items():
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(img_np, lower, upper)
            results[name] = (np.sum(mask > 0) / total) * 100
        
        return results


# ============ ОСНОВНАЯ ФУНКЦИЯ ============

def main():
    print("=" * 60)
    print("🚗 АНАЛИЗ МАРШРУТА: Тонгариро → Окленд")
    print("=" * 60)
    
    # 1. Загрузка трека
    route = RouteAnalyzer(GPX_FILE)
    bounds = route.get_route_bounds()
    
    print(f"\n🗺️ Границы маршрута:")
    print(f"   Широта: {bounds['min_lat']:.4f} → {bounds['max_lat']:.4f}")
    print(f"   Долгота: {bounds['min_lon']:.4f} → {bounds['max_lon']:.4f}")
    
    # 2. Создание карты
    route.create_map()
    
    # 3. Выбор точек для анализа
    sample_points = route.sample_points_along_route(num_samples=10)
    
    # 4. Инициализация
    downloader = SatelliteImageDownloader(API_KEY, LAYER_ID)
    road_analyzer = RoadSurfaceAnalyzer()
    
    # 5. Сбор данных
    results = []
    
    print(f"\n📸 Анализ {len(sample_points)} точек вдоль маршрута...")
    
    for i, point in enumerate(sample_points):
        print(f"\n   Точка {i+1}/{len(sample_points)}: {point['lat']:.4f}, {point['lon']:.4f}")
        
        # Спутниковый снимок
        img = downloader.download_tile(point['lat'], point['lon'], zoom=16)
        if img:
            surface = road_analyzer.analyze_image(img)
            print(f"      🛣️ Асфальт: {surface['асфальт']:.1f}%, Грунт: {surface['грунт']:.1f}%")
        else:
            surface = {'асфальт': None, 'грунт': None, 'песок': None}
            print(f"      ❌ Нет снимка")
        
        # Погода
        weather = None
        if point['time']:
            weather = route.get_weather_for_point(point['lat'], point['lon'], point['time'])
            if weather and weather['temperature']:
                print(f"      🌡️ {weather['temperature']:.1f}°C, ☔ {weather['precipitation']:.1f}мм")
        
        results.append({
            'lat': point['lat'],
            'lon': point['lon'],
            'elevation': point['elevation'] or 0,
            'time': str(point['time']) if point['time'] else None,
            'asphalt_pct': surface['асфальт'],
            'dirt_pct': surface['грунт'],
            'temperature': weather['temperature'] if weather else None,
            'precipitation': weather['precipitation'] if weather else None
        })
        
        time.sleep(0.3)
    
    # 6. Сохранение
    df = pd.DataFrame(results)
    df.to_csv('route_analysis_results.csv', index=False)
    print(f"\n✅ Результаты сохранены: route_analysis_results.csv")
    print(df.to_string())
    
    # 7. Визуализация
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Высота
    axes[0, 0].plot(range(len(results)), [r['elevation'] for r in results], 'g-', linewidth=2)
    axes[0, 0].set_ylabel('Высота (м)')
    axes[0, 0].set_title('Профиль высоты')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Асфальт
    asphalt_vals = [r['asphalt_pct'] if r['asphalt_pct'] else 0 for r in results]
    axes[0, 1].bar(range(len(results)), asphalt_vals, color='gray', alpha=0.7)
    axes[0, 1].set_ylabel('Асфальт (%)')
    axes[0, 1].set_title('Дорожное покрытие')
    axes[0, 1].set_ylim(0, 100)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Температура
    temps = [r['temperature'] for r in results if r['temperature']]
    if temps:
        axes[1, 0].plot(range(len(temps)), temps, 'r-o', linewidth=2)
        axes[1, 0].set_ylabel('Температура (°C)')
        axes[1, 0].set_title('Температура')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Корреляция
    valid = [(r['elevation'], r['asphalt_pct']) for r in results if r['asphalt_pct']]
    if valid:
        elev, asphalt = zip(*valid)
        axes[1, 1].scatter(elev, asphalt, alpha=0.7, s=80)
        axes[1, 1].set_xlabel('Высота (м)')
        axes[1, 1].set_ylabel('Асфальт (%)')
        axes[1, 1].set_title('Корреляция: высота → асфальт')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('route_analysis.png', dpi=150)
    plt.show()
    
    print("\n✅ АНАЛИЗ ЗАВЕРШЕН!")


if __name__ == "__main__":
    # Установка библиотек при необходимости
    # pip install gpxpy folium opencv-python pillow requests pandas numpy matplotlib
    
    main()