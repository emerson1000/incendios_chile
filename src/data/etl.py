"""
Módulo ETL para ingesta y transformación de datos de incendios forestales
"""
import pandas as pd
import numpy as np
from pathlib import Path
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import logging

try:
    import geopandas as gpd
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, GEO_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FireDataETL:
    """Clase principal para ETL de datos de incendios forestales"""
    
    def __init__(self, raw_dir: Path = RAW_DATA_DIR, processed_dir: Path = PROCESSED_DATA_DIR):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Importar downloader
        try:
            from src.data.downloaders import DataDownloader
            self.downloader = DataDownloader(raw_dir=raw_dir)
            self.has_downloader = True
        except ImportError:
            logger.warning("No se pudo importar DataDownloader. No se intentarán descargas automáticas.")
            self.has_downloader = False
        
    def load_conaf_data(self, file_path: Optional[str] = None, 
                       try_download: bool = True) -> pd.DataFrame:
        """
        Carga datos históricos de incendios de CONAF
        
        Estrategia:
        1. Si se proporciona file_path y existe, lo carga
        2. Si no, busca archivos en data/raw/
        3. Si no encuentra, intenta descargar datos reales
        4. Solo como último recurso genera datos sintéticos
        """
        # 1. Intentar cargar desde archivo específico
        if file_path and Path(file_path).exists():
            logger.info(f"Cargando datos CONAF desde {file_path}")
            df = pd.read_csv(file_path, encoding='utf-8')
        else:
            # 2. Buscar archivos en data/raw/
            raw_files = list(self.raw_dir.glob("conaf*.csv"))
            raw_files.extend(list(self.raw_dir.glob("incendio*.csv")))
            raw_files.extend(list(self.raw_dir.glob("firms*.csv")))
            
            if raw_files:
                latest_file = max(raw_files, key=lambda p: p.stat().st_mtime)
                logger.info(f"Encontrado archivo de incendios: {latest_file}")
                df = pd.read_csv(latest_file, encoding='utf-8')
            elif try_download and self.has_downloader:
                # 3. Intentar descargar datos reales
                logger.info("No se encontraron archivos locales. Intentando descargar datos reales...")
                df = self.downloader.download_conaf_data()
                
                if df is None or len(df) == 0:
                    logger.warning("No se pudieron descargar datos reales. Intentando NASA FIRMS...")
                    # Intentar NASA FIRMS como alternativa
                    firms_data = self.downloader.download_nasa_firms_data(
                        country='Chile',
                        start_date='2020-01-01'
                    )
                    if firms_data is not None and len(firms_data) > 0:
                        df = self.downloader._convert_firms_to_conaf_format(firms_data)
                
                if df is None or len(df) == 0:
                    logger.error("No se pudieron obtener datos reales. Usando datos sintéticos solo para pruebas.")
                    logger.error("Por favor, descarga datos reales manualmente o verifica la conexión a internet.")
                    df = self._generate_synthetic_conaf_data()
                    logger.warning("⚠️ ADVERTENCIA: Usando datos sintéticos. Los resultados no serán reales.")
                else:
                    logger.info(f"✅ Datos reales descargados exitosamente: {len(df)} registros")
            else:
                # 4. Último recurso: datos sintéticos (solo para pruebas)
                logger.error("No se encontraron datos y no se intentó descargar.")
                logger.error("Por favor, proporciona datos reales o habilita try_download=True")
                raise FileNotFoundError(
                    "No se encontraron datos de incendios. "
                    "Por favor, descarga datos reales o proporciona file_path."
                )
        
        # Normalizar columnas
        df = self._normalize_conaf_data(df)
        return df
    
    def _generate_synthetic_conaf_data(self, years: List[int] = None) -> pd.DataFrame:
        """
        Genera datos sintéticos de incendios para desarrollo y pruebas
        Basado en estadísticas reales de Chile
        """
        if years is None:
            years = list(range(2002, 2024))
        
        # Comunas con mayor riesgo (basadas en datos reales)
        comunas_alto_riesgo = [
            "Valparaíso", "Viña del Mar", "Quilpué", "Villa Alemana", "La Ligua",
            "Petorca", "Los Andes", "San Felipe", "Rancagua", "San Fernando",
            "Talca", "Linares", "Parral", "Chillán", "Los Ángeles",
            "Temuco", "Villarrica", "Pucón", "Valdivia", "Osorno"
        ]
        
        all_incendios = []
        
        for year in years:
            # Temporada alta: diciembre a marzo
            for month in range(1, 13):
                if month in [12, 1, 2, 3]:  # Temporada alta
                    n_incendios = np.random.poisson(15)
                elif month in [4, 5, 10, 11]:  # Temporada media
                    n_incendios = np.random.poisson(5)
                else:  # Temporada baja
                    n_incendios = np.random.poisson(2)
                
                for _ in range(n_incendios):
                    # Seleccionar comuna (mayor probabilidad en alto riesgo)
                    if np.random.random() < 0.6:
                        comuna = np.random.choice(comunas_alto_riesgo)
                    else:
                        comuna = f"Comuna_{np.random.randint(1, 100)}"
                    
                    # Generar fecha aleatoria en el mes
                    try:
                        fecha = pd.Timestamp(year=year, month=month, 
                                           day=np.random.randint(1, 29))
                    except:
                        fecha = pd.Timestamp(year=year, month=month, day=1)
                    
                    # Área quemada (hectáreas) - distribución log-normal
                    area_quemada = np.random.lognormal(mean=4, sigma=1.5)
                    area_quemada = min(area_quemada, 10000)  # Máximo razonable
                    
                    # Duración del incendio (días)
                    duracion = max(1, int(np.random.gamma(2, 2)))
                    
                    # Personas afectadas
                    personas_afectadas = int(area_quemada * np.random.uniform(0.1, 2))
                    
                    incendio = {
                        "fecha_inicio": fecha,
                        "comuna": comuna,
                        "region": self._get_region(comuna),
                        "area_quemada_ha": area_quemada,
                        "duracion_dias": duracion,
                        "personas_afectadas": personas_afectadas,
                        "causa": np.random.choice(
                            ["Humana", "Natural", "Desconocida"],
                            p=[0.7, 0.2, 0.1]
                        ),
                        "anio": year,
                        "mes": month,
                        "temporada": self._get_temporada(month)
                    }
                    all_incendios.append(incendio)
        
        df = pd.DataFrame(all_incendios)
        return df
    
    def _get_region(self, comuna: str) -> str:
        """Mapea comuna a región (simplificado)"""
        regiones_map = {
            "Valparaíso": "V", "Viña del Mar": "V", "Quilpué": "V",
            "Villa Alemana": "V", "La Ligua": "V", "Petorca": "V",
            "Talca": "VII", "Linares": "VII", "Parral": "VII",
            "Chillán": "XVI", "Los Ángeles": "VIII",
            "Temuco": "IX", "Villarrica": "IX", "Pucón": "IX",
            "Valdivia": "XIV", "Osorno": "X"
        }
        return regiones_map.get(comuna, "Desconocida")
    
    def _get_temporada(self, month: int) -> str:
        """Define temporada de incendios"""
        if month in [12, 1, 2, 3]:
            return "Alta"
        elif month in [4, 5, 10, 11]:
            return "Media"
        else:
            return "Baja"
    
    def _normalize_conaf_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normaliza y limpia datos de CONAF"""
        # Asegurar que fecha_inicio es datetime
        if 'fecha_inicio' in df.columns:
            df['fecha_inicio'] = pd.to_datetime(df['fecha_inicio'], errors='coerce')
        
        # Eliminar filas con fechas inválidas
        df = df.dropna(subset=['fecha_inicio'])
        
        # Crear columnas adicionales útiles
        if 'fecha_inicio' in df.columns:
            df['anio'] = df['fecha_inicio'].dt.year
            df['mes'] = df['fecha_inicio'].dt.month
            df['dia_semana'] = df['fecha_inicio'].dt.dayofweek
            df['dia_anio'] = df['fecha_inicio'].dt.dayofyear
        
        return df
    
    def load_climate_data(self, file_path: Optional[str] = None,
                         try_download: bool = True) -> pd.DataFrame:
        """
        Carga datos climáticos (CR2 u otra fuente)
        
        Estrategia:
        1. Si se proporciona file_path y existe, lo carga
        2. Si no, busca archivos en data/raw/
        3. Si no encuentra, intenta descargar datos reales (CR2 u Open-Meteo)
        4. Solo como último recurso genera datos sintéticos
        """
        # 1. Intentar cargar desde archivo específico
        if file_path and Path(file_path).exists():
            logger.info(f"Cargando datos climáticos desde {file_path}")
            df = pd.read_csv(file_path, encoding='utf-8')
        else:
            # 2. Buscar archivos en data/raw/
            raw_files = list(self.raw_dir.glob("cr2*.csv"))
            raw_files.extend(list(self.raw_dir.glob("clima*.csv")))
            raw_files.extend(list(self.raw_dir.glob("weather*.csv")))
            
            if raw_files:
                latest_file = max(raw_files, key=lambda p: p.stat().st_mtime)
                logger.info(f"Encontrado archivo climático: {latest_file}")
                df = pd.read_csv(latest_file, encoding='utf-8')
            elif try_download and self.has_downloader:
                # 3. Intentar descargar datos reales
                logger.info("No se encontraron archivos locales. Intentando descargar datos climáticos reales...")
                df = self.downloader.download_cr2_climate_data()
                
                if df is None or len(df) == 0:
                    logger.warning("No se pudieron descargar datos de CR2. Intentando Open-Meteo...")
                    # Intentar Open-Meteo como alternativa
                    df = self.downloader.download_openmeteo_data(
                        lat=-33.4489,  # Santiago
                        lon=-70.6693,
                        start_date='2000-01-01'
                    )
                
                if df is None or len(df) == 0:
                    logger.error("No se pudieron obtener datos climáticos reales.")
                    logger.error("Por favor, descarga datos reales manualmente o verifica la conexión.")
                    df = self._generate_synthetic_climate_data()
                    logger.warning("⚠️ ADVERTENCIA: Usando datos climáticos sintéticos. Los resultados no serán reales.")
                else:
                    logger.info(f"✅ Datos climáticos reales descargados exitosamente: {len(df)} registros")
            else:
                # 4. Último recurso: datos sintéticos
                logger.error("No se encontraron datos y no se intentó descargar.")
                logger.error("Por favor, proporciona datos reales o habilita try_download=True")
                raise FileNotFoundError(
                    "No se encontraron datos climáticos. "
                    "Por favor, descarga datos reales o proporciona file_path."
                )
        
        return df
    
    def _generate_synthetic_climate_data(self, 
                                        start_date: str = "2002-01-01",
                                        end_date: str = "2024-12-31") -> pd.DataFrame:
        """Genera datos climáticos sintéticos"""
        fechas = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generar datos climáticos con estacionalidad y tendencia
        n_days = len(fechas)
        
        # Temperatura máxima (con estacionalidad)
        temp_max = 25 + 8 * np.sin(2 * np.pi * np.arange(n_days) / 365.25 + np.pi/2)
        temp_max += np.random.normal(0, 3, n_days)
        # Aumento por sequía (tendencia)
        temp_max += np.linspace(0, 2, n_days)
        
        # Temperatura mínima
        temp_min = temp_max - np.random.uniform(8, 15, n_days)
        
        # Humedad relativa (inversa a temperatura)
        humedad = 60 - 20 * np.sin(2 * np.pi * np.arange(n_days) / 365.25 + np.pi/2)
        humedad += np.random.normal(0, 10, n_days)
        humedad = np.clip(humedad, 20, 95)
        
        # Velocidad del viento
        viento = 15 + 10 * np.random.gamma(1.5, 2, n_days)
        
        # Precipitación (muy estacional, mayor en invierno)
        precip_base = 5 - 4 * np.sin(2 * np.pi * np.arange(n_days) / 365.25 - np.pi/2)
        precipitacion = np.maximum(0, precip_base + np.random.exponential(2, n_days))
        # Reducir precipitación en años de sequía
        precipitacion *= (1 - np.linspace(0, 0.3, n_days))
        
        # Índice de sequía (acumulado de déficit de precipitación)
        sequia = np.cumsum(np.maximum(0, 10 - precipitacion)) / 100
        
        # Déficit hídrico
        deficit = np.maximum(0, temp_max - 20) * (1 - humedad/100) * (1 - precipitacion/50)
        
        df_clima = pd.DataFrame({
            'fecha': fechas,
            'temperatura_maxima': temp_max,
            'temperatura_minima': temp_min,
            'humedad_relativa': humedad,
            'velocidad_viento': viento,
            'precipitacion': precipitacion,
            'indice_sequia': sequia,
            'deficit_hidrico': deficit,
            'anio': fechas.year,
            'mes': fechas.month,
            'dia_anio': fechas.dayofyear
        })
        
        return df_clima
    
    def create_panel_data(self, 
                         incendios_df: pd.DataFrame,
                         clima_df: pd.DataFrame,
                         comunas: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Crea panel espacio-temporal: comuna x fecha
        
        Combina datos de incendios, clima y otras features
        """
        logger.info("Creando panel espacio-temporal...")
        
        # Obtener lista de comunas únicas
        if comunas is None:
            comunas = sorted(incendios_df['comuna'].unique().tolist())
        
        # Crear rango de fechas
        fecha_min = min(incendios_df['fecha_inicio'].min(), clima_df['fecha'].min())
        fecha_max = max(incendios_df['fecha_inicio'].max(), clima_df['fecha'].max())
        fechas = pd.date_range(start=fecha_min, end=fecha_max, freq='D')
        
        # Crear panel completo
        panel_list = []
        for comuna in comunas:
            for fecha in fechas:
                panel_list.append({
                    'comuna': comuna,
                    'fecha': fecha,
                    'anio': fecha.year,
                    'mes': fecha.month,
                    'dia_anio': fecha.dayofyear,
                    'dia_semana': fecha.dayofweek
                })
        
        panel_df = pd.DataFrame(panel_list)
        
        # Merge con datos de incendios (agregar variable objetivo)
        incendios_agg = incendios_df.groupby(['comuna', 'fecha_inicio']).agg({
            'area_quemada_ha': 'sum',
            'personas_afectadas': 'sum',
            'duracion_dias': 'max'
        }).reset_index()
        incendios_agg = incendios_agg.rename(columns={'fecha_inicio': 'fecha'})
        incendios_agg['incendio_ocurrencia'] = 1
        
        panel_df = panel_df.merge(
            incendios_agg[['comuna', 'fecha', 'incendio_ocurrencia', 
                          'area_quemada_ha', 'personas_afectadas', 'duracion_dias']],
            on=['comuna', 'fecha'],
            how='left'
        )
        
        # Llenar valores faltantes
        panel_df['incendio_ocurrencia'] = panel_df['incendio_ocurrencia'].fillna(0).astype(int)
        panel_df['area_quemada_ha'] = panel_df['area_quemada_ha'].fillna(0)
        panel_df['personas_afectadas'] = panel_df['personas_afectadas'].fillna(0)
        panel_df['duracion_dias'] = panel_df['duracion_dias'].fillna(0)
        
        # Merge con datos climáticos (promedio por fecha, podría ser por región/comuna)
        panel_df = panel_df.merge(
            clima_df[['fecha', 'temperatura_maxima', 'temperatura_minima',
                     'humedad_relativa', 'velocidad_viento', 'precipitacion',
                     'indice_sequia', 'deficit_hidrico']],
            on='fecha',
            how='left'
        )
        
        # Agregar features adicionales
        panel_df = self._add_additional_features(panel_df, incendios_df)
        
        logger.info(f"Panel creado: {len(panel_df)} observaciones ({len(comunas)} comunas x {len(fechas)} fechas)")
        
        return panel_df
    
    def _add_additional_features(self, panel_df: pd.DataFrame, 
                                incendios_df: pd.DataFrame) -> pd.DataFrame:
        """Agrega features adicionales al panel"""
        
        # Features temporales
        panel_df['temporada_alta'] = panel_df['mes'].isin([12, 1, 2, 3]).astype(int)
        panel_df['fin_semana'] = (panel_df['dia_semana'] >= 5).astype(int)
        panel_df['mes_sin'] = np.sin(2 * np.pi * panel_df['mes'] / 12)
        panel_df['mes_cos'] = np.cos(2 * np.pi * panel_df['mes'] / 12)
        
        # Features históricas por comuna (lags temporales)
        for comuna in panel_df['comuna'].unique():
            mask = panel_df['comuna'] == comuna
            comuna_data = panel_df.loc[mask].sort_values('fecha')
            
            # Incendios en últimos 7 días
            panel_df.loc[mask, 'incendios_7d'] = (
                comuna_data['incendio_ocurrencia'].rolling(window=7, min_periods=1).sum().shift(1).fillna(0)
            )
            
            # Incendios en últimos 30 días
            panel_df.loc[mask, 'incendios_30d'] = (
                comuna_data['incendio_ocurrencia'].rolling(window=30, min_periods=1).sum().shift(1).fillna(0)
            )
            
            # Área quemada acumulada últimos 365 días
            panel_df.loc[mask, 'area_quemada_365d'] = (
                comuna_data['area_quemada_ha'].rolling(window=365, min_periods=1).sum().shift(1).fillna(0)
            )
        
        # Features de riesgo base por comuna (basado en histórico)
        # Contar incendios por comuna
        incendios_por_comuna = incendios_df.groupby('comuna').size().reset_index(name='total_incendios_historico')
        
        # Área promedio por comuna
        area_promedio = incendios_df.groupby('comuna')['area_quemada_ha'].mean().reset_index()
        area_promedio.columns = ['comuna', 'area_promedio_historico']
        
        # Combinar
        riesgo_base = incendios_por_comuna.merge(area_promedio, on='comuna', how='left')
        riesgo_base['area_promedio_historico'] = riesgo_base['area_promedio_historico'].fillna(0)
        
        # Normalizar riesgo base
        if riesgo_base['total_incendios_historico'].max() > 0:
            riesgo_base['riesgo_base_comuna'] = (
                riesgo_base['total_incendios_historico'] / riesgo_base['total_incendios_historico'].max()
            )
        else:
            riesgo_base['riesgo_base_comuna'] = 0
        
        panel_df = panel_df.merge(riesgo_base[['comuna', 'riesgo_base_comuna']], 
                                  on='comuna', how='left')
        panel_df['riesgo_base_comuna'] = panel_df['riesgo_base_comuna'].fillna(0)
        
        return panel_df
    
    def save_processed_data(self, panel_df: pd.DataFrame, filename: str = "panel_incendios.parquet"):
        """Guarda panel procesado en formato parquet"""
        output_path = self.processed_dir / filename
        panel_df.to_parquet(output_path, compression='snappy', index=False)
        logger.info(f"Datos procesados guardados en {output_path}")
        return output_path
    
    def load_processed_data(self, filename: str = "panel_incendios.parquet") -> pd.DataFrame:
        """Carga panel procesado"""
        input_path = self.processed_dir / filename
        if not input_path.exists():
            raise FileNotFoundError(f"No se encontró {input_path}")
        
        df = pd.read_parquet(input_path)
        logger.info(f"Datos cargados desde {input_path}")
        return df

