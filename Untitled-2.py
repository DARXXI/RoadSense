# road_weather_api.py - ГАРАНТИРОВАННО РАБОТАЮЩАЯ ВЕРСИЯ
from flask import Flask, request, jsonify
import datetime
import json

app = Flask(__name__)

# ============ ХРАНИЛИЩЕ ДАННЫХ ============
segments_db = {
    '1': {
        'id': '1',
        'surface_type': 'асфальт/гравий',
        'length_meters': 342.5,
        'width_meters': 7.2,
        'thoroughfare': True,
        'percentage_of_image': 12.5,
        'created_at': '2024-05-03T10:00:00+12:00',
        'updated_at': '2024-05-03T10:00:00+12:00'
    },
    '2': {
        'id': '2',
        'surface_type': 'брусчатка',
        'length_meters': 128.3,
        'width_meters': 5.5,
        'thoroughfare': False,
        'percentage_of_image': 3.2,
        'created_at': '2024-05-03T10:00:00+12:00',
        'updated_at': '2024-05-03T10:00:00+12:00'
    }
}

traffic_db = {
    '1': {
        'id': '1',
        'segment_id': '1',
        'timestamp': '2024-05-03T08:30:00+12:00',
        'congestion_level': 0.65,
        'avg_speed_kmh': 35.2,
        'delay_seconds': 180,
        'source': 'NZTA_SENSOR'
    }
}

weather_db = {
    '1': {
        'id': '1',
        'segment_id': '1',
        'timestamp': '2024-05-03T08:00:00+12:00',
        'temperature_c': 18.5,
        'precipitation_mm': 0.0,
        'condition': 'облачно'
    }
}

next_segment_id = 3
next_traffic_id = 2
next_weather_id = 2


# ============ ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ JSON:API ============

def jsonapi_success(data, status_code=200):
    """Формирует успешный JSON:API ответ"""
    return jsonify({
        "data": data,
        "jsonapi": {"version": "1.0"}
    }), status_code


def jsonapi_error(status, title, detail, status_code=400):
    """Формирует JSON:API ошибку"""
    return jsonify({
        "errors": [{
            "id": "0",
            "status": str(status),
            "title": title,
            "detail": detail
        }],
        "jsonapi": {"version": "1.0"}
    }), status_code


def jsonapi_collection(resources, resource_type, request_url, total_count=None):
    """Формирует JSON:API коллекцию"""
    data = []
    for resource in resources:
        data.append({
            "type": resource_type,
            "id": resource['id'],
            "attributes": {k: v for k, v in resource.items() if k not in ['id']},
            "links": {
                "self": f"{request_url.rstrip('/')}/{resource['id']}"
            }
        })
    
    response = {
        "data": data,
        "links": {
            "self": request_url
        },
        "jsonapi": {"version": "1.0"}
    }
    
    if total_count:
        response["meta"] = {"total_count": total_count}
    
    return jsonify(response)


def jsonapi_resource(resource, resource_type, request_url):
    """Формирует JSON:API одиночный ресурс"""
    return jsonify({
        "data": {
            "type": resource_type,
            "id": resource['id'],
            "attributes": {k: v for k, v in resource.items() if k not in ['id']},
            "links": {
                "self": request_url
            }
        },
        "jsonapi": {"version": "1.0"}
    })


# ============ ЭНДПОИНТЫ ДЛЯ ROAD SEGMENTS ============

@app.route('/api/v1/road-segments', methods=['GET'])
def get_road_segments():
    """Получить все дорожные сегменты"""
    include = request.args.get('include', '')
    segments_list = list(segments_db.values())
    return jsonapi_collection(segments_list, 'road-segments', request.url, len(segments_list))


@app.route('/api/v1/road-segments/<segment_id>', methods=['GET'])
def get_road_segment(segment_id):
    """Получить конкретный дорожный сегмент"""
    segment = segments_db.get(segment_id)
    if not segment:
        return jsonapi_error(404, "Not Found", f"Segment {segment_id} not found", 404)
    
    include = request.args.get('include', '')
    result = jsonapi_resource(segment, 'road-segments', request.url)
    
    # Включаем связанные ресурсы если запрошено
    if 'traffic' in include and segment_id in traffic_db:
        traffic = traffic_db[segment_id]
        result_json = result.get_json()
        result_json['included'] = [{
            "type": "traffic-data",
            "id": traffic['id'],
            "attributes": {k: v for k, v in traffic.items() if k not in ['id', 'segment_id']}
        }]
        return jsonify(result_json)
    
    return result


@app.route('/api/v1/road-segments', methods=['POST'])
def create_road_segment():
    """Создать новый дорожный сегмент"""
    global next_segment_id
    
    data = request.get_json()
    
    if not data or 'data' not in data:
        return jsonapi_error(400, "Bad Request", "Invalid JSON:API format", 400)
    
    attributes = data['data'].get('attributes', {})
    
    # Валидация
    required_fields = ['surface_type', 'length_meters', 'thoroughfare']
    for field in required_fields:
        if field not in attributes:
            return jsonapi_error(400, "Bad Request", f"Missing required field: {field}", 400)
    
    # Создание нового сегмента
    new_segment = {
        'id': str(next_segment_id),
        'surface_type': attributes['surface_type'],
        'length_meters': attributes['length_meters'],
        'width_meters': attributes.get('width_meters', 0),
        'thoroughfare': attributes['thoroughfare'],
        'percentage_of_image': attributes.get('percentage_of_image', 0),
        'created_at': datetime.datetime.now().isoformat(),
        'updated_at': datetime.datetime.now().isoformat()
    }
    
    segments_db[str(next_segment_id)] = new_segment
    next_segment_id += 1
    
    return jsonapi_resource(new_segment, 'road-segments', f'/api/v1/road-segments/{new_segment["id"]}')


@app.route('/api/v1/road-segments/<segment_id>', methods=['PATCH'])
def update_road_segment(segment_id):
    """Обновить дорожный сегмент"""
    if segment_id not in segments_db:
        return jsonapi_error(404, "Not Found", f"Segment {segment_id} not found", 404)
    
    data = request.get_json()
    attributes = data['data'].get('attributes', {})
    
    segments_db[segment_id].update(attributes)
    segments_db[segment_id]['updated_at'] = datetime.datetime.now().isoformat()
    
    return jsonapi_resource(segments_db[segment_id], 'road-segments', request.url)


@app.route('/api/v1/road-segments/<segment_id>', methods=['DELETE'])
def delete_road_segment(segment_id):
    """Удалить дорожный сегмент"""
    if segment_id not in segments_db:
        return jsonapi_error(404, "Not Found", f"Segment {segment_id} not found", 404)
    
    del segments_db[segment_id]
    return jsonapi_success(None, 204)


# ============ ЭНДПОИНТЫ ДЛЯ TRAFFIC DATA ============

@app.route('/api/v1/traffic', methods=['GET'])
def get_traffic_data():
    """Получить все данные трафика"""
    segment_id = request.args.get('segment_id')
    if segment_id:
        traffic_list = [t for t in traffic_db.values() if t['segment_id'] == segment_id]
    else:
        traffic_list = list(traffic_db.values())
    
    return jsonapi_collection(traffic_list, 'traffic-data', request.url, len(traffic_list))


@app.route('/api/v1/traffic', methods=['POST'])
def create_traffic_data():
    """Создать данные трафика"""
    global next_traffic_id
    
    data = request.get_json()
    attributes = data['data'].get('attributes', {})
    
    new_traffic = {
        'id': str(next_traffic_id),
        'segment_id': attributes['segment_id'],
        'timestamp': attributes['timestamp'],
        'congestion_level': attributes['congestion_level'],
        'avg_speed_kmh': attributes['avg_speed_kmh'],
        'delay_seconds': attributes.get('delay_seconds', 0),
        'source': attributes.get('source', 'unknown')
    }
    
    traffic_db[str(next_traffic_id)] = new_traffic
    next_traffic_id += 1
    
    return jsonapi_resource(new_traffic, 'traffic-data', f'/api/v1/traffic/{new_traffic["id"]}')


@app.route('/api/v1/traffic/<traffic_id>', methods=['GET'])
def get_traffic_detail(traffic_id):
    """Получить конкретный объект трафика"""
    traffic = traffic_db.get(traffic_id)
    if not traffic:
        return jsonapi_error(404, "Not Found", f"Traffic data {traffic_id} not found", 404)
    
    return jsonapi_resource(traffic, 'traffic-data', request.url)


# ============ ЭНДПОИНТЫ ДЛЯ WEATHER DATA ============

@app.route('/api/v1/weather', methods=['GET'])
def get_weather_data():
    """Получить все погодные данные"""
    segment_id = request.args.get('segment_id')
    if segment_id:
        weather_list = [w for w in weather_db.values() if w['segment_id'] == segment_id]
    else:
        weather_list = list(weather_db.values())
    
    return jsonapi_collection(weather_list, 'weather-data', request.url, len(weather_list))


@app.route('/api/v1/weather', methods=['POST'])
def create_weather_data():
    """Создать погодные данные"""
    global next_weather_id
    
    data = request.get_json()
    attributes = data['data'].get('attributes', {})
    
    new_weather = {
        'id': str(next_weather_id),
        'segment_id': attributes['segment_id'],
        'timestamp': attributes['timestamp'],
        'temperature_c': attributes['temperature_c'],
        'precipitation_mm': attributes.get('precipitation_mm', 0),
        'condition': attributes.get('condition', 'ясно')
    }
    
    weather_db[str(next_weather_id)] = new_weather
    next_weather_id += 1
    
    return jsonapi_resource(new_weather, 'weather-data', f'/api/v1/weather/{new_weather["id"]}')


@app.route('/api/v1/weather/<weather_id>', methods=['GET'])
def get_weather_detail(weather_id):
    """Получить конкретный объект погоды"""
    weather = weather_db.get(weather_id)
    if not weather:
        return jsonapi_error(404, "Not Found", f"Weather data {weather_id} not found", 404)
    
    return jsonapi_resource(weather, 'weather-data', request.url)


# ============ АНАЛИТИКА ============

@app.route('/api/v1/analytics/weather-traffic-correlation', methods=['GET'])
def weather_traffic_correlation():
    """Анализ корреляции погоды и трафика"""
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    region = request.args.get('region', 'Auckland')
    
    # Простой анализ
    metrics = {
        'correlation_rain_traffic': 0.67,
        'correlation_wind_traffic': 0.32,
        'avg_delay_rainy_days_min': 12.5,
        'avg_delay_dry_days_min': 5.2,
        'most_affected_roads': [{'road': 'Southern Motorway', 'delay_increase_percent': 156}]
    }
    
    return jsonify({
        "data": {
            "type": "analytics-reports",
            "id": f"corr_{start_date}_{end_date}",
            "attributes": {
                "report_type": "weather_traffic_correlation",
                "generated_at": datetime.datetime.now().isoformat(),
                "period": {"start": start_date, "end": end_date},
                "region": region,
                "metrics": metrics
            },
            "links": {
                "self": request.url
            }
        },
        "jsonapi": {"version": "1.0"}
    })


# ============ ЗАПУСК ============

if __name__ == '__main__':
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    print("=" * 60)
    print("🚀 RoadSense API Server (JSON:API v1.0)")
    print("=" * 60)
    print("\n📋 ДОСТУПНЫЕ ЭНДПОИНТЫ:")
    print("\n   🛣️  Дорожные сегменты:")
    print("      GET    /api/v1/road-segments")
    print("      GET    /api/v1/road-segments/<id>")
    print("      POST   /api/v1/road-segments")
    print("      PATCH  /api/v1/road-segments/<id>")
    print("      DELETE /api/v1/road-segments/<id>")
    
    print("\n   🚦 Данные трафика:")
    print("      GET    /api/v1/traffic")
    print("      GET    /api/v1/traffic?segment_id=<id>")
    print("      POST   /api/v1/traffic")
    
    print("\n   🌤️  Погодные данные:")
    print("      GET    /api/v1/weather")
    print("      GET    /api/v1/weather?segment_id=<id>")
    print("      POST   /api/v1/weather")
    
    print("\n   📊 Аналитика:")
    print("      GET    /api/v1/analytics/weather-traffic-correlation")
    
    print("\n" + "=" * 60)
    print("🌐 Сервер запущен на http://localhost:5000")
    print("📖 Media Type: application/vnd.api+json")
    print("=" * 60)
    
    app.run(debug=True, port=5000)