"""
Utilidades y funciones auxiliares para el proyecto
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def calculate_risk_category(risk_score: float, 
                           thresholds: List[float] = [0.3, 0.6]) -> str:
    """
    Calcula categoría de riesgo basada en score
    
    Args:
        risk_score: Score de riesgo (0-1)
        thresholds: Umbrales para categorías [bajo, medio, alto]
    
    Returns:
        Categoría de riesgo: 'Bajo', 'Medio', o 'Alto'
    """
    if risk_score < thresholds[0]:
        return 'Bajo'
    elif risk_score < thresholds[1]:
        return 'Medio'
    else:
        return 'Alto'


def calculate_travel_time(lat1: float, lon1: float, 
                         lat2: float, lon2: float,
                         speed_kmh: float = 60) -> float:
    """
    Calcula tiempo de viaje entre dos puntos usando distancia euclidiana
    
    Args:
        lat1, lon1: Coordenadas punto 1
        lat2, lon2: Coordenadas punto 2
        speed_kmh: Velocidad promedio en km/h
    
    Returns:
        Tiempo de viaje en minutos
    """
    # Distancia euclidiana aproximada
    # 1 grado latitud ≈ 111 km
    # 1 grado longitud ≈ 111 km * cos(latitud)
    distance_km = np.sqrt(
        (lat1 - lat2)**2 * 111**2 +
        (lon1 - lon2)**2 * 111**2 * np.cos(np.radians((lat1 + lat2) / 2))
    )
    
    # Tiempo en minutos
    time_minutes = (distance_km / speed_kmh) * 60
    
    return time_minutes


def aggregate_by_temporada(df: pd.DataFrame, 
                          date_col: str = 'fecha') -> pd.DataFrame:
    """
    Agrega columna de temporada de incendios
    
    Args:
        df: DataFrame con columna de fecha
        date_col: Nombre de columna de fecha
    
    Returns:
        DataFrame con columna 'temporada' agregada
    """
    df = df.copy()
    
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df['mes'] = df[date_col].dt.month
    
    # Temporada alta: dic-mar (12, 1, 2, 3)
    # Temporada media: abr-may, oct-nov (4, 5, 10, 11)
    # Temporada baja: jun-sep (6, 7, 8, 9)
    
    def get_temporada(mes):
        if mes in [12, 1, 2, 3]:
            return 'Alta'
        elif mes in [4, 5, 10, 11]:
            return 'Media'
        else:
            return 'Baja'
    
    df['temporada'] = df['mes'].apply(get_temporada)
    
    return df


def calculate_expected_damage(risk_prob: float, 
                             severity_ha: float = None,
                             population: float = None,
                             weight_ha: float = 1.0,
                             weight_pop: float = 1.5) -> float:
    """
    Calcula daño esperado = riesgo × severidad
    
    Args:
        risk_prob: Probabilidad de incendio (0-1)
        severity_ha: Severidad en hectáreas esperadas
        population: Población afectada esperada
        weight_ha: Peso de hectáreas
        weight_pop: Peso de población
    
    Returns:
        Daño esperado (score normalizado)
    """
    if severity_ha is None:
        severity_ha = risk_prob * 1000  # Estimación por defecto
    
    damage = risk_prob * (
        weight_ha * severity_ha +
        (weight_pop * population if population is not None else 0)
    )
    
    return damage


def validate_data(df: pd.DataFrame, 
                 required_cols: List[str],
                 data_type: str = "panel") -> bool:
    """
    Valida que un DataFrame tenga las columnas requeridas
    
    Args:
        df: DataFrame a validar
        required_cols: Lista de columnas requeridas
        data_type: Tipo de datos (para mensajes de error)
    
    Returns:
        True si válido, False si no
    """
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        logger.error(f"Faltan columnas en {data_type}: {missing_cols}")
        return False
    
    logger.info(f"Datos {data_type} válidos: {len(df)} filas, {len(df.columns)} columnas")
    return True


def get_feature_importance_summary(feature_importance: pd.DataFrame,
                                  top_n: int = 10) -> pd.DataFrame:
    """
    Obtiene resumen de importancia de features
    
    Args:
        feature_importance: DataFrame con 'feature' e 'importance'
        top_n: Número de top features a retornar
    
    Returns:
        DataFrame con top N features
    """
    if feature_importance is None or len(feature_importance) == 0:
        return pd.DataFrame()
    
    # Ordenar por importancia
    top_features = feature_importance.sort_values('importance', ascending=False).head(top_n)
    
    # Normalizar importancia
    top_features['importance_normalized'] = (
        top_features['importance'] / top_features['importance'].sum()
    )
    
    # Categorizar features
    def categorize_feature(feature):
        if 'temp' in feature.lower() or 'clima' in feature.lower():
            return 'Climático'
        elif 'historico' in feature.lower() or 'lag' in feature.lower():
            return 'Histórico'
        elif 'comuna' in feature.lower():
            return 'Geográfico'
        elif 'temporada' in feature.lower() or 'mes' in feature.lower():
            return 'Temporal'
        else:
            return 'Otro'
    
    top_features['categoria'] = top_features['feature'].apply(categorize_feature)
    
    return top_features

