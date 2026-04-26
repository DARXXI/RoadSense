# satellite_road_detector_fixed.py - ВЕРСИЯ БЕЗ KORNIA
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import requests
from io import BytesIO

class SatelliteRoadDetectorPro:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Устройство: {self.device}")
        
        # Загружаем модель
        self.model = self._create_custom_model()
        
    def _create_custom_model(self):
        """
        Создаем простую, но эффективную модель для дорог
        """
        import torchvision.models as models
        
        class RoadDetector(nn.Module):
            def __init__(self):
                super().__init__()
                # Используем ResNet34 как бэкбон
                resnet = models.resnet34(pretrained=True)
                self.encoder = nn.Sequential(*list(resnet.children())[:-2])
                
                # Декодер для сегментации
                self.decoder = nn.Sequential(
                    nn.Conv2d(512, 256, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(256, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(128, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(64, 1, 1),
                    nn.Sigmoid()
                )
                
            def forward(self, x):
                features = self.encoder(x)
                # Апсемплинг до размера входа
                features = nn.functional.interpolate(
                    features, 
                    size=(x.shape[2], x.shape[3]), 
                    mode='bilinear', 
                    align_corners=False
                )
                return self.decoder(features)
        
        model = RoadDetector().to(self.device)
        model.eval()
        print("✅ Создана кастомная модель (использует веса ImageNet)")
        
        # Пытаемся загрузить веса если есть
        weights_path = "road_detector_weights.pth"
        if Path(weights_path).exists():
            model.load_state_dict(torch.load(weights_path, map_location=self.device))
            print("✅ Загружены сохраненные веса")
        
        return model
    
    def preprocess_satellite_image(self, image_path):
        """
        Специальная предобработка для спутниковых снимков
        """
        # Загрузка
        if isinstance(image_path, str):
            if image_path.startswith('http'):
                response = requests.get(image_path)
                img = Image.open(BytesIO(response.content))
            else:
                img = Image.open(image_path)
        else:
            img = image_path
        
        # Конвертируем в RGB если нужно
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Конвертируем в numpy
        img_np = np.array(img)
        
        # 1. Нормализация (растяжка гистограммы для спутниковых снимков)
        for i in range(3):
            img_np[:,:,i] = cv2.normalize(img_np[:,:,i], None, 0, 255, cv2.NORM_MINMAX)
        
        # 2. Улучшение контраста (CLAHE)
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        img_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return Image.fromarray(img_np), img_np
    
    def detect_roads_advanced(self, image_path):
        """
        Продвинутая детекция дорог
        """
        # Предобработка
        pil_img, img_np = self.preprocess_satellite_image(image_path)
        original_size = pil_img.size[::-1]  # (H, W)
        
        # Подготовка для нейросети
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(pil_img).unsqueeze(0).to(self.device)
        
        # Инференс
        with torch.no_grad():
            output = self.model(input_tensor)
            
            # Ресайз обратно
            if len(output.shape) == 4:
                output = torch.nn.functional.interpolate(
                    output,
                    size=(original_size[0], original_size[1]),
                    mode='bilinear',
                    align_corners=False
                ).squeeze()
            else:
                output = torch.nn.functional.interpolate(
                    output.unsqueeze(0),
                    size=(original_size[0], original_size[1]),
                    mode='bilinear',
                    align_corners=False
                ).squeeze()
        
        # Преобразуем в маску
        if output.shape != (original_size[0], original_size[1]):
            output = output.squeeze()
        
        road_mask = (output.cpu().numpy() > 0.5).astype(np.uint8) * 255
        
        # Дополнительная постобработка
        road_mask = self._postprocess_mask(road_mask, img_np)
        
        return road_mask, img_np
    
    def _postprocess_mask(self, mask, original_img):
        """
        Постобработка маски для улучшения результатов
        """
        # Морфологическая очистка
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Удаление маленьких объектов
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < 100:  # Удаляем объекты меньше 100 пикселей
                cv2.drawContours(mask, [cnt], -1, 0, -1)
        
        # Традиционная детекция дорог (как дополнение к нейросети)
        gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
        
        # Поиск прямых линий (дороги часто прямые)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                                minLineLength=100, maxLineGap=50)
        
        if lines is not None:
            line_mask = np.zeros_like(mask)
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_mask, (x1, y1), (x2, y2), 255, 5)
            
            # Комбинируем маски
            mask = cv2.bitwise_or(mask, line_mask)
        
        return mask
    
    def calculate_road_metrics(self, road_mask, original_img):
        """
        Расширенная статистика дорог
        """
        # Основная статистика
        total_pixels = road_mask.size
        road_pixels = np.sum(road_mask > 0)
        road_percentage = (road_pixels / total_pixels) * 100 if total_pixels > 0 else 0
        
        # Анализ контуров дорог
        contours, _ = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            lengths = [cv2.arcLength(cnt, False) for cnt in contours]
            areas = [cv2.contourArea(cnt) for cnt in contours]
            
            # Оценка ширины дороги
            widths = []
            for cnt in contours:
                if len(cnt) > 0:
                    rect = cv2.minAreaRect(cnt)
                    width = min(rect[1])
                    widths.append(width)
            
            avg_width = np.mean(widths) if widths else 0
            avg_length = np.mean(lengths) if lengths else 0
            avg_area = np.mean(areas) if areas else 0
        else:
            avg_length = 0
            avg_area = 0
            avg_width = 0
        
        return {
            'road_percentage': road_percentage,
            'road_pixels': road_pixels,
            'total_pixels': total_pixels,
            'num_road_segments': len(contours),
            'avg_segment_length': avg_length,
            'avg_segment_area': avg_area,
            'avg_road_width_pixels': avg_width,
            'road_density': road_percentage / 100
        }
    
    def visualize_results(self, original_img, road_mask, metrics):
        """
        Визуализация с дополнительной информацией
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Оригинал
        axes[0, 0].imshow(original_img)
        axes[0, 0].set_title('Спутниковый снимок', fontsize=14)
        axes[0, 0].axis('off')
        
        # Маска дорог
        axes[0, 1].imshow(road_mask, cmap='hot')
        axes[0, 1].set_title(f'Детекция дорог ({metrics["road_percentage"]:.2f}%)', fontsize=14)
        axes[0, 1].axis('off')
        
        # Наложение
        overlay = original_img.copy()
        overlay[road_mask > 0] = [255, 0, 0]  # Красные дороги
        alpha = 0.6
        overlay = cv2.addWeighted(original_img, 1-alpha, overlay, alpha, 0)
        axes[1, 0].imshow(overlay)
        axes[1, 0].set_title('Дороги (красные)', fontsize=14)
        axes[1, 0].axis('off')
        
        # Статистика
        axes[1, 1].axis('off')
        stats_text = f"""
        📊 СТАТИСТИКА ДОРОГ:
        
        🛣️  Процент дорог: {metrics['road_percentage']:.2f}%
        🎯 Плотность дорог: {metrics['road_density']:.3f}
        
        🔢 Сегментов: {metrics['num_road_segments']}
        📏 Средняя длина: {metrics['avg_segment_length']:.1f} px
        📐 Средняя ширина: {metrics['avg_road_width_pixels']:.1f} px
        📐 Средняя площадь: {metrics['avg_segment_area']:.0f} px²
        
        📌 Всего пикселей дорог: {metrics['road_pixels']:,}
        """
        
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                        fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('satellite_roads_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("\n✅ Результат сохранен как 'satellite_roads_analysis.png'")
    
    def run(self, image_path):
        """
        Запуск полного анализа
        """
        print(f"🛰️  Анализ спутникового снимка: {image_path}")
        
        try:
            # Детекция дорог
            road_mask, original_img = self.detect_roads_advanced(image_path)
            
            # Расчет метрик
            metrics = self.calculate_road_metrics(road_mask, original_img)
            
            # Вывод результатов
            print("\n" + "="*60)
            print("📊 РЕЗУЛЬТАТЫ ДЕТЕКЦИИ ДОРОГ НА СПУТНИКОВОМ СНИМКЕ")
            print("="*60)
            print(f"📐 Процент дорог: {metrics['road_percentage']:.2f}%")
            print(f"🔢 Количество дорожных сегментов: {metrics['num_road_segments']}")
            
            if metrics['num_road_segments'] > 0:
                print(f"📏 Средняя длина сегментов: {metrics['avg_segment_length']:.1f} пикселей")
                print(f"🛣️  Средняя ширина дороги: {metrics['avg_road_width_pixels']:.1f} пикселей")
                
                # Оценка ширины в метрах (для большинства спутников 1 пиксель ≈ 0.5-10 м)
                estimated_width_m = metrics['avg_road_width_pixels'] * 0.5
                print(f"📐 Ориентировочная ширина: ~{estimated_width_m:.1f} метров")
            else:
                print("⚠️ Дороги не обнаружены.")
                print("\nВозможные причины:")
                print("  1. Слишком низкое разрешение снимка")
                print("  2. Снимок не содержит дорог (лес, пустыня, вода)")
                print("  3. Требуется улучшить предобработку изображения")
                print("\n📌 Рекомендации:")
                print("  - Используйте снимки с разрешением не ниже 1 м/пиксель")
                print("  - Попробуйте обрезать снимок до области с предполагаемыми дорогами")
                print("  - Увеличьте контраст и яркость изображения")
            
            # Визуализация
            self.visualize_results(original_img, road_mask, metrics)
            
            return road_mask, metrics
            
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            print("\n🔄 Пробуем альтернативный метод...")
            return self._fallback_detection(image_path)
    
    def _fallback_detection(self, image_path):
        """
        Fallback метод если нейросеть не сработала
        """
        print("Использую классические методы компьютерного зрения...")
        
        # Загрузка
        img = cv2.imread(str(image_path))
        if img is None:
            pil_img = Image.open(image_path)
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Улучшение изображения
        gray = cv2.equalizeHist(gray)
        
        # Поиск прямых линий (характерно для дорог)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                                minLineLength=200, maxLineGap=50)
        
        mask = np.zeros_like(gray)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(mask, (x1, y1), (x2, y2), 255, 8)
        
        # Поиск однородных областей (дороги обычно однородные)
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        _, homogeneous = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Комбинируем результаты
        final_mask = cv2.bitwise_or(mask, homogeneous)
        
        # Морфологическая очистка
        kernel = np.ones((5,5), np.uint8)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
        
        total_pixels = final_mask.size
        road_pixels = np.sum(final_mask > 0)
        road_percentage = (road_pixels / total_pixels) * 100
        
        metrics = {
            'road_percentage': road_percentage,
            'num_road_segments': len(cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]),
            'road_pixels': road_pixels,
            'total_pixels': total_pixels,
            'avg_segment_length': 0,
            'avg_segment_area': 0,
            'avg_road_width_pixels': 0,
            'road_density': road_percentage / 100
        }
        
        print(f"\n📊 Результаты классического метода:")
        print(f"Процент дорог: {road_percentage:.2f}%")
        print(f"Дорожных сегментов: {metrics['num_road_segments']}")
        
        self.visualize_results(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), final_mask, metrics)
        
        return final_mask, metrics

# Запуск
if __name__ == "__main__":
    # Установка кодировки для Windows
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    detector = SatelliteRoadDetectorPro()
    
    # Ваш файл
    image_path = r"C:\Users\Arthur\Documents\RoadSense\satellite_image.tiff"
    
    # Проверка существования файла
    if not Path(image_path).exists():
        print(f"❌ Файл не найден: {image_path}")
        image_path = input("Введите правильный путь к изображению: ").strip()
    
    road_mask, metrics = detector.run(image_path)
    
    # Сохраняем маску
    if road_mask is not None:
        cv2.imwrite('road_mask.png', road_mask)
        print("\n💾 Маска дорог сохранена как 'road_mask.png'")
    
    # Сохраняем статистику в файл
    with open('road_analysis_results.txt', 'w', encoding='utf-8') as f:
        f.write("РЕЗУЛЬТАТЫ АНАЛИЗА ДОРОГ\n")
        f.write("="*40 + "\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    
    print("💾 Статистика сохранена в 'road_analysis_results.txt'")