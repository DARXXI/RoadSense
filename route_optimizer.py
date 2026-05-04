# 04_route_optimizer.py - OPT-2 оптимизация маршрута с формулой T = l/v + A*C + p*t
import numpy as np
import random
import json
import sys

class RouteOptimizer:
    """
    OPT-2 оптимизатор маршрута с формулой:
    T = l/v + A*C + p*t
    """
    
    def __init__(self, config):
        self.v0 = config.V0
        self.k_surf = config.K_SURF
        self.k_weather = config.K_WEATHER
        self.k_time = config.K_TIME
    
    def get_time_category(self, hour):
        if 7 <= hour <= 9 or 16 <= hour <= 18:
            return 'peak_hour'
        elif hour < 6 or hour > 22:
            return 'night'
        return 'day'
    
    def calculate_travel_time(self, distance_km, surface, weather, hour):
        """l/v - чистое время движения"""
        time_cat = self.get_time_category(hour)
        speed = (self.v0 * 
                 self.k_surf.get(surface, 1.0) * 
                 self.k_weather.get(weather, 1.0) * 
                 self.k_time.get(time_cat, 1.0))
        return (distance_km / speed) * 60
    
    def calculate_accident_impact(self, distance_km, surface, weather, hour, 
                                   police_distance_km, active_accident_prob=0.03):
        """A*C + p*t - влияние аварий"""
        # A*C - активные аварии
        if random.random() < active_accident_prob:
            jam_delay = random.uniform(8, 25)
        else:
            jam_delay = 0
        
        # p - вероятность нового ДТП
        base_risk = 0.003
        weather_risk = 1.4 if weather in ['rain', 'heavy_rain'] else 1.0
        surface_risk = 1.0 if surface == 'asphalt' else (1.3 if surface == 'gravel' else 1.5)
        time_risk = 1.5 if (7 <= hour <= 9 or 16 <= hour <= 18) else 1.0
        risk = base_risk * weather_risk * surface_risk * time_risk
        risk = min(0.08, risk)
        
        # t - время устранения
        clearance = 37 * (1 + police_distance_km / 30) * (1.25 if weather in ['rain', 'heavy_rain'] else 1.0)
        
        expected_delay = risk * (clearance / 60)
        
        return jam_delay + expected_delay
    
    def segment_time(self, segment):
        """Полное время для одного сегмента по формуле"""
        travel = self.calculate_travel_time(
            segment['distance_km'],
            segment.get('surface', 'asphalt'),
            segment.get('weather', 'fine'),
            segment.get('hour', 12)
        )
        
        accident = self.calculate_accident_impact(
            segment['distance_km'],
            segment.get('surface', 'asphalt'),
            segment.get('weather', 'fine'),
            segment.get('hour', 12),
            segment.get('police_distance_km', 10),
            segment.get('active_accident_prob', 0.03)
        )
        
        return travel + accident
    
    def route_total_time(self, route):
        """Общее время маршрута"""
        return sum(self.segment_time(seg) for seg in route)
    
    def opt_2_improve(self, route, max_iterations=100):
        """
        OPT-2 улучшение маршрута
        Меняет местами сегменты для уменьшения общего времени
        """
        best_route = route.copy()
        best_time = self.route_total_time(best_route)
        improved = True
        iterations = 0
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            for i in range(len(best_route) - 1):
                for j in range(i + 1, len(best_route)):
                    # Пробуем поменять местами
                    new_route = best_route.copy()
                    new_route[i], new_route[j] = new_route[j], new_route[i]
                    new_time = self.route_total_time(new_route)
                    
                    if new_time < best_time:
                        best_route = new_route
                        best_time = new_time
                        improved = True
        
        return best_route, best_time
    
    def optimize_delivery_route(self, points, distances_matrix, surface_map, weather, hour):
        """
        Оптимизация маршрута доставки между точками
        """
        n = len(points)
        route = list(range(n))
        
        # Создаем сегменты с данными
        segments = []
        for i in range(n - 1):
            from_idx, to_idx = i, i + 1
            segments.append({
                'from': points[from_idx],
                'to': points[to_idx],
                'distance_km': distances_matrix[from_idx][to_idx],
                'surface': surface_map.get(points[to_idx], 'asphalt'),
                'weather': weather,
                'hour': hour,
                'police_distance_km': 10
            })
        
        # OPT-2 оптимизация
        best_segments, total_time = self.opt_2_improve(segments)
        
        return {
            'route': [s['from'] for s in best_segments] + [best_segments[-1]['to']],
            'total_time_min': round(total_time, 1),
            'total_time_hours': round(total_time / 60, 1),
            'segments': best_segments
        }


if __name__ == "__main__":
    import config
    optimizer = RouteOptimizer(config)
    
    # Тест
    test_segment = {'distance_km': 10, 'surface': 'asphalt', 'weather': 'fine', 'hour': 12}
    print(f"Время: {optimizer.segment_time(test_segment):.1f} мин")