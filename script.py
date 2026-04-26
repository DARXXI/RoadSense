# road_surface_segments_fixed.py - СТАРЫЕ ДИАПАЗОНЫ
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
from pathlib import Path
import json
from collections import defaultdict

class RoadSurfaceSegmentation:
    def __init__(self):
        # СТАРЫЕ цветовые диапазоны (которые были в начале)
        self.surface_colors = {
           'асфальт/гравий': {
                'ranges': [
                    ([70, 70, 70], [130, 130, 130]),    # асфальт
                    ([90, 90, 90], [150, 150, 150]),    # светлый асфальт
                    ([50, 50, 50], [100, 100, 100]),    # темный асфальт
                    ([100, 100, 100], [160, 160, 160]), # гравий
                    ([80, 80, 80], [130, 130, 130]),    # темный гравий
                ],
                'color': (100, 100, 100)
            },
            # 'брусчатка': {
            #     'ranges': [
            #         ([110, 100, 80], [170, 150, 120])
            #     ],  
            #     'color': (140, 120, 100)
            # },
            'грунт/грязь': {
                'ranges': [
                    ([60, 50, 40], [110, 90, 70]),
                    ([80, 70, 50], [130, 110, 90]),
                    ([100, 85, 65], [150, 125, 100]),
                ],
                'color': (80, 70, 55)
            },
            'песок': {
                'ranges': [
                    ([180, 170, 130], [235, 215, 185]),
                    ([200, 180, 140], [255, 225, 195]),
                    ([160, 150, 110], [210, 190, 160]),
                ],
                'color': (200, 180, 140)
            }
        }
    
    def get_geodeep_mask(self, image_path):
        """Получение маски дорог через geodeep"""
        from geodeep import segment
        
        print("🛰️  Получение маски дорог через geodeep...")
        mask = segment(image_path, "roads")
        mask = (mask > 0).astype(np.uint8) * 255
        
        print(f"✅ Маска получена. Дорог: {(mask > 0).sum() / mask.size * 100:.2f}%")
        return mask
    
    def load_image(self, image_path):
        """Загрузка оригинального изображения"""
        import imageio
        image = imageio.imread(image_path)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = image[:, :, :3]
        return image
    
    def find_through_and_dead_ends(self, mask):
        """Находит сквозные дороги и тупики"""
        print("🔍 Поиск сквозных дорог и тупиков...")
        
        num_labels, labeled = cv2.connectedComponents(mask)
        
        through_mask = np.zeros_like(mask)
        dead_ends_mask = np.zeros_like(mask)
        
        for label_id in range(1, num_labels):
            component_mask = (labeled == label_id).astype(np.uint8) * 255
            
            touches_boundary = (
                np.any(component_mask[0, :]) or
                np.any(component_mask[-1, :]) or
                np.any(component_mask[:, 0]) or
                np.any(component_mask[:, -1])
            )
            
            if touches_boundary:
                through_mask = cv2.bitwise_or(through_mask, component_mask)
            else:
                dead_ends_mask = cv2.bitwise_or(dead_ends_mask, component_mask)
        
        print(f"   Сквозные дороги: {np.sum(through_mask > 0)} px")
        print(f"   Тупики: {np.sum(dead_ends_mask > 0)} px")
        
        return through_mask, dead_ends_mask
    
    def classify_surfaces(self, image, road_mask):
        """
        Классификация типов покрытия (старые диапазоны)
        """
        print("\n🎨 Классификация типов покрытия...")
        
        height, width = image.shape[:2]
        surface_map = np.zeros((height, width), dtype=np.uint8)
        
        surface_names = list(self.surface_colors.keys())
        
        for surface_idx, surface_name in enumerate(surface_names, 1):
            surface_mask = np.zeros((height, width), dtype=np.uint8)
            
            for lower, upper in self.surface_colors[surface_name]['ranges']:
                lower = np.array(lower, dtype=np.uint8)
                upper = np.array(upper, dtype=np.uint8)
                
                range_mask = cv2.inRange(image, lower, upper)
                range_mask = cv2.bitwise_and(range_mask, road_mask)
                surface_mask = cv2.bitwise_or(surface_mask, range_mask)
            
            # Морфологическая очистка
            kernel = np.ones((7, 7), np.uint8)
            surface_mask = cv2.morphologyEx(surface_mask, cv2.MORPH_CLOSE, kernel)
            surface_mask = cv2.morphologyEx(surface_mask, cv2.MORPH_OPEN, kernel)
            
            surface_map[surface_mask > 0] = surface_idx
            
            pixels = np.sum(surface_mask > 0)
            if pixels > 0:
                road_pixels = np.sum(road_mask > 0)
                print(f"   {surface_name}: {pixels} px ({pixels/road_pixels*100:.1f}% дорог)")
        
        return surface_map, surface_names
    
    def create_surface_overlay(self, original_image, surface_map, surface_names):
        """
        Создание наложения типов покрытия
        """
        overlay = original_image.copy()
        
        # Яркие цвета для видимости
        display_colors = {
            'асфальт/гравий': (100, 100, 100),
            'брусчатка': (180, 140, 100),
            'грунт/грязь': (139, 69, 19),
            'песок': (255, 220, 150)
        }
        
        for idx, name in enumerate(surface_names, 1):
            color = display_colors.get(name, (100, 100, 100))
            mask = (surface_map == idx)
            if np.any(mask):
                overlay[mask] = (overlay[mask] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
        
        return overlay
    
    def create_surface_only(self, surface_map, surface_names):
        """
        Создание изображения только с типами покрытия
        """
        surface_only = np.zeros((surface_map.shape[0], surface_map.shape[1], 3), dtype=np.uint8)
        
        display_colors = {
            'асфальт/гравий': (100, 100, 100),
            'брусчатка': (180, 140, 100),
            'грунт/грязь': (139, 69, 19),
            'песок': (255, 220, 150)
        }
        
        for idx, name in enumerate(surface_names, 1):
            color = display_colors.get(name, (100, 100, 100))
            surface_only[surface_map == idx] = color
        
        return surface_only
    
    def calculate_statistics(self, through_mask, dead_ends_mask, surface_map, surface_names, total_pixels):
        """Расчет статистики"""
        all_roads_pixels = np.sum(through_mask > 0) + np.sum(dead_ends_mask > 0)
        
        stats = {
            'total_roads': {
                'pixels': int(all_roads_pixels),
                'percentage': float(all_roads_pixels / total_pixels * 100)
            },
            'through_roads': {
                'pixels': int(np.sum(through_mask > 0)),
                'percentage': float(np.sum(through_mask > 0) / total_pixels * 100)
            },
            'dead_ends': {
                'pixels': int(np.sum(dead_ends_mask > 0)),
                'percentage': float(np.sum(dead_ends_mask > 0) / total_pixels * 100)
            },
            'surfaces': {}
        }
        
        through_pixels = stats['through_roads']['pixels']
        for idx, surface_name in enumerate(surface_names, 1):
            surface_pixels = np.sum(surface_map == idx)
            if surface_pixels > 0:
                stats['surfaces'][surface_name] = {
                    'pixels': int(surface_pixels),
                    'percentage_of_image': float(surface_pixels / total_pixels * 100),
                    'percentage_of_through_roads': float(surface_pixels / through_pixels * 100) if through_pixels > 0 else 0
                }
        
        return stats
    
    def visualize_results(self, original, through_mask, dead_ends_mask, 
                         surface_map, surface_names, stats):
        """
        Визуализация результатов
        """
        # Создаем наложения
        overlay_through = original.copy()
        overlay_through[through_mask > 0] = [255, 80, 80]
        overlay_through = cv2.addWeighted(original, 0.5, overlay_through, 0.5, 0)
        
        overlay_all = original.copy()
        overlay_all[through_mask > 0] = [255, 80, 80]
        overlay_all[dead_ends_mask > 0] = [255, 255, 80]
        overlay_all = cv2.addWeighted(original, 0.5, overlay_all, 0.5, 0)
        
        overlay_surface = self.create_surface_overlay(original, surface_map, surface_names)
        surface_only = self.create_surface_only(surface_map, surface_names)
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # 1. Оригинал
        axes[0, 0].imshow(original)
        axes[0, 0].set_title('1. Оригинальный снимок', fontsize=14)
        axes[0, 0].axis('off')
        
        # 2. Сквозные дороги
        axes[0, 1].imshow(overlay_through)
        axes[0, 1].set_title(f'2. Сквозные дороги\n{stats["through_roads"]["percentage"]:.2f}%', fontsize=12)
        axes[0, 1].axis('off')
        
        # 3. Все дороги
        axes[0, 2].imshow(overlay_all)
        axes[0, 2].set_title(f'3. Все дороги (сквозные+тупики)\nвсего {stats["total_roads"]["percentage"]:.2f}%', fontsize=12)
        axes[0, 2].axis('off')
        
        # 4. ТИПЫ ПОКРЫТИЯ (отдельно)
        axes[1, 0].imshow(surface_only)
        axes[1, 0].set_title('4. ТИПЫ ПОКРЫТИЯ (только сегментация)', fontsize=14)
        axes[1, 0].axis('off')
        
        # 5. Наложение типов покрытия
        axes[1, 1].imshow(overlay_surface)
        axes[1, 1].set_title('5. Типы покрытия на оригинале', fontsize=14)
        axes[1, 1].axis('off')
        
        # 6. Статистика
        axes[1, 2].axis('off')
        
        stats_text = "📊 СТАТИСТИКА\n\n"
        stats_text += f"🚗 ВСЕГО ДОРОГ: {stats['total_roads']['percentage']:.2f}%\n"
        stats_text += f"   ├─ Сквозные: {stats['through_roads']['percentage']:.2f}%\n"
        stats_text += f"   └─ Тупики: {stats['dead_ends']['percentage']:.2f}%\n\n"
        stats_text += "🎨 ТИПЫ ПОКРЫТИЯ:\n"
        
        for surface, data in stats['surfaces'].items():
            if data['percentage_of_image'] > 0:
                stats_text += f"\n   {surface}:\n"
                stats_text += f"      • {data['percentage_of_image']:.2f}% снимка\n"
                stats_text += f"      • {data['percentage_of_through_roads']:.1f}% дорог"
        
        axes[1, 2].text(0.05, 0.95, stats_text, fontsize=10, 
                       verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        
        # Легенда
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, fc=np.array([100,100,100])/255, label='асфальт/гравий'),
            plt.Rectangle((0, 0), 1, 1, fc=np.array([180,140,100])/255, label='брусчатка'),
            plt.Rectangle((0, 0), 1, 1, fc=np.array([139,69,19])/255, label='грунт/грязь'),
            plt.Rectangle((0, 0), 1, 1, fc=np.array([255,220,150])/255, label='песок'),
            plt.Rectangle((0, 0), 1, 1, fc=np.array([255,80,80])/255, label='сквозные дороги'),
            plt.Rectangle((0, 0), 1, 1, fc=np.array([255,255,80])/255, label='тупики')
        ]
        
        fig.legend(handles=legend_elements, loc='lower center', 
                  ncol=6, fontsize=10, bbox_to_anchor=(0.5, -0.02))
        
        plt.suptitle('СЕГМЕНТАЦИЯ ДОРОГ ПО ТИПАМ ПОКРЫТИЯ (СТАРЫЕ ДИАПАЗОНЫ)', 
                    fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig('road_segmentation_result.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Сохраняем файлы
        cv2.imwrite('surface_only.png', cv2.cvtColor(surface_only, cv2.COLOR_RGB2BGR))
        cv2.imwrite('surface_overlay.png', cv2.cvtColor(overlay_surface, cv2.COLOR_RGB2BGR))
        cv2.imwrite('roads_overlay.png', cv2.cvtColor(overlay_through, cv2.COLOR_RGB2BGR))
        
        print("\n✅ Сохранены файлы:")
        print("   📍 surface_only.png - ТОЛЬКО типы покрытия")
        print("   📍 surface_overlay.png - типы покрытия на оригинале")
        print("   📍 roads_overlay.png - дороги на оригинале")
        
        return overlay_surface
    
    def run(self, image_path):
        """Запуск анализа"""
        print("="*60)
        print("🚀 СЕГМЕНТАЦИЯ ДОРОГ ПО ТИПАМ ПОКРЫТИЯ")
        print("="*60)
        
        # 1. Загрузка
        original = self.load_image(image_path)
        geodeep_mask = self.get_geodeep_mask(image_path)
        
        # 2. Находим сквозные дороги и тупики
        through_mask, dead_ends_mask = self.find_through_and_dead_ends(geodeep_mask)
        
        # 3. Классификация типов покрытия
        surface_map, surface_names = self.classify_surfaces(original, through_mask)
        
        # 4. Статистика
        total_pixels = original.shape[0] * original.shape[1]
        stats = self.calculate_statistics(through_mask, dead_ends_mask, surface_map, surface_names, total_pixels)
        
        # 5. Визуализация
        overlay = self.visualize_results(original, through_mask, dead_ends_mask, 
                                         surface_map, surface_names, stats)
        
        # 6. Итоговый вывод
        print("\n" + "="*60)
        print("📊 ИТОГОВАЯ СТАТИСТИКА")
        print("="*60)
        print(f"📐 Всего дорог: {stats['total_roads']['percentage']:.2f}%")
        print(f"   ├─ Сквозные: {stats['through_roads']['percentage']:.2f}%")
        print(f"   └─ Тупики: {stats['dead_ends']['percentage']:.2f}%")
        
        print("\n🎨 Типы покрытия на сквозных дорогах:")
        for surface, data in stats['surfaces'].items():
            if data['percentage_of_image'] > 0:
                print(f"   • {surface}: {data['percentage_of_image']:.2f}% снимка")
        
        return stats, surface_map, through_mask

# Запуск
if __name__ == "__main__":
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    #image_path = r"C:\Users\Arthur\Documents\RoadSense\satellite_image.tiff"
    image_path = r"C:\Users\Arthur\Documents\RoadSense\Снимки\Screenshot 2026-04-26 033207.png"

    analyzer = RoadSurfaceSegmentation()
    
    try:
        stats, surface_map, road_mask = analyzer.run(image_path)
        print("\n✅ АНАЛИЗ ЗАВЕРШЕН!")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()