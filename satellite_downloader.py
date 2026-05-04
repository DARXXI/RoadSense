# 01_satellite_downloader.py - Загрузка спутниковых снимков LINZ (100 шт)
import requests
from PIL import Image
from io import BytesIO
import math
import os
from datetime import datetime
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

class SatelliteDownloader:
    """Загрузчик спутниковых снимков для Новой Зеландии"""
    
    def __init__(self, api_key, layer_id, output_dir):
        self.api_key = api_key
        self.layer_id = layer_id
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def deg_to_tile(self, lat, lon, zoom):
        n = 2.0 ** zoom
        xtile = int((lon + 180.0) / 360.0 * n)
        ytile = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
        return xtile, ytile
    
    def download_tile(self, zoom, x, y):
        url = f"https://tiles-cdn.koordinates.com/services;key={self.api_key}/tiles/v4/layer={self.layer_id}/EPSG:3857/{zoom}/{x}/{y}.png"
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                return Image.open(BytesIO(response.content))
        except Exception as e:
            print(f"   Ошибка: {e}")
        return None
    
    def download_images_for_gps_track(self, gps_points, zoom=19, radius=1):
        """
        Загрузка снимков для всех точек GPS трека
        """
        print(f"\n🛰️ Загрузка {len(gps_points)} снимков для GPS трека...")
        
        downloaded = []
        for i, (lat, lon) in enumerate(gps_points):
            center_x, center_y = self.deg_to_tile(lat, lon, zoom)
            
            # Создаем мозаику 3x3
            size = radius * 2 + 1
            first_tile = self.download_tile(zoom, center_x, center_y)
            if not first_tile:
                continue
            
            w, h = first_tile.size
            mosaic = Image.new('RGB', (w * size, h * size))
            
            success = 0
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    x, y = center_x + dx, center_y + dy
                    tile = self.download_tile(zoom, x, y)
                    if tile:
                        mosaic.paste(tile, ((dx + radius) * w, (dy + radius) * h))
                        success += 1
            
            if success > 0:
                filename = f"{self.output_dir}/point_{i:03d}_zoom{zoom}.png"
                mosaic.save(filename)
                downloaded.append(filename)
                print(f"   ✅ Точка {i+1}: {filename}")
        
        return downloaded


def download_batch(locations, zoom=19, radius=1):
    """Загрузка 100 снимков для заданных координат"""
    from config import LINZ_API_KEY, LINZ_LAYER_ID, SATELLITE_DIR
    
    downloader = SatelliteDownloader(LINZ_API_KEY, LINZ_LAYER_ID, SATELLITE_DIR)
    return downloader.download_images_for_gps_track(locations, zoom, radius)


if __name__ == "__main__":
    # Тестовые координаты (центр Окленда + случайные)
    test_locations = [(-36.8485, 174.7633) for _ in range(5)]
    download_batch(test_locations, zoom=19, radius=1)