"""
Configuración del proyecto de predicción y optimización de incendios forestales
"""
import os
from pathlib import Path

# Directorios base
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Crear directorios si no existen
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# URLs de fuentes de datos (ejemplos - adaptar según disponibilidad real)
DATA_SOURCES = {
    "conaf_incendios": "https://www.conaf.cl/incendios-forestales/incendios-forestales-en-chile/",
    "firms_nasa": "https://firms.modaps.eosdis.nasa.gov/api/",
    "cr2_clima": "http://www.cr2.cl/datos-de-clima-y-descargas/",
    "ide_chile": "https://www.ide.cl/index.php/descarga-de-capas/capas-de-incendios"
}

# Parámetros del modelo
MODEL_CONFIG = {
    "target_variable": "incendio_ocurrencia",  # o "area_quemada", "numero_incendios"
    "prediction_horizon_days": 1,  # Predicción para el día siguiente
    "temporal_features": {
        "lags_days": [1, 3, 7, 14, 30],  # Días de atraso para features temporales
        "seasonal_features": True,
        "day_of_week": True,
        "month": True
    },
    "train_test_split": {
        "test_size": 0.2,
        "validation_size": 0.1,
        "temporal_split": True  # Usar split temporal en lugar de aleatorio
    }
}

# Parámetros de optimización
OPTIMIZATION_CONFIG = {
    "max_brigades": 50,  # Número máximo de brigadas disponibles
    "max_bases": 20,  # Número máximo de bases posibles
    "response_time_threshold_minutes": 60,  # Tiempo máximo de respuesta en minutos
    "objective": "minimize_damage",  # "minimize_damage" o "minimize_response_time"
    "risk_weight": 1.0,  # Peso del riesgo en la función objetivo
    "severity_weight": 1.5,  # Peso de la severidad esperada
    "solver": "PULP_CBC_CMD"  # Solver para optimización (PULP_CBC_CMD, HiGHS, etc.)
}

# Parámetros geográficos
GEO_CONFIG = {
    "unit_analysis": "comuna",  # "comuna" o "grid"
    "grid_resolution_km": 1.0,  # Resolución en km si se usa grid
    "coordinate_system": "EPSG:4326",  # WGS84
    "projected_crs": "EPSG:32719"  # UTM Zone 19S para Chile
}

# Features del modelo
FEATURES = {
    "climatic": [
        "temperatura_maxima",
        "temperatura_minima",
        "humedad_relativa",
        "velocidad_viento",
        "precipitacion",
        "indice_sequia",
        "deficit_hidrico"
    ],
    "vegetation": [
        "ndvi",
        "tipo_cobertura",
        "densidad_vegetacion",
        "biomasa"
    ],
    "topographic": [
        "elevacion",
        "pendiente",
        "orientacion"
    ],
    "socioeconomic": [
        "densidad_poblacion",
        "distancia_caminos",
        "viviendas_interfaz",
        "actividad_economica"
    ],
    "historical": [
        "incendios_previos_mes",
        "incendios_previos_anio",
        "tiempo_ultimo_incendio"
    ]
}

# Configuración del dashboard
DASHBOARD_CONFIG = {
    "title": "Sistema de Predicción y Optimización de Recursos para Incendios Forestales - Chile",
    "map_center": [-35.6751, -71.5430],  # Centro de Chile (coordenadas aproximadas)
    "map_zoom": 6,
    "update_frequency_hours": 24  # Frecuencia de actualización de datos
}

