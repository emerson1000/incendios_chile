"""
Módulo para descargar datos reales de incendios forestales de Chile
"""
import pandas as pd
import numpy as np
import requests
from pathlib import Path
from typing import Optional, Dict, List
import logging
from datetime import datetime, timedelta
import time
import json
from bs4 import BeautifulSoup

from config import RAW_DATA_DIR, DATA_SOURCES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataDownloader:
    """Clase para descargar datos reales de fuentes públicas"""
    
    def __init__(self, raw_dir: Path = RAW_DATA_DIR):
        self.raw_dir = raw_dir
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def download_conaf_data(self, save_path: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Intenta descargar datos históricos de incendios de CONAF
        
        Nota: CONAF puede tener diferentes formatos y URLs
        Esta función intenta varias estrategias comunes
        """
        logger.info("Intentando descargar datos de CONAF...")
        
        if save_path is None:
            save_path = self.raw_dir / "conaf_incendios_historico.csv"
        
        # Estrategia 1: Intentar descargar desde URL directa (si existe)
        # CONAF usualmente publica datos en su sitio web
        conaf_url = DATA_SOURCES.get("conaf_incendios")
        
        try:
            # Intentar encontrar enlaces de descarga en la página
            response = self.session.get(conaf_url, timeout=30)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Buscar enlaces a archivos CSV o Excel
                links = soup.find_all('a', href=True)
                for link in links:
                    href = link.get('href', '')
                    if any(ext in href.lower() for ext in ['.csv', '.xlsx', '.xls']):
                        if 'incendio' in href.lower() or 'fuego' in href.lower():
                            logger.info(f"Encontrado enlace de descarga: {href}")
                            # Intentar descargar
                            try:
                                file_url = href if href.startswith('http') else conaf_url + href
                                df = self._download_file(file_url)
                                if df is not None:
                                    df.to_csv(save_path, index=False, encoding='utf-8-sig')
                                    logger.info(f"✅ Datos CONAF descargados y guardados en {save_path}")
                                    return df
                            except Exception as e:
                                logger.warning(f"No se pudo descargar desde {href}: {e}")
                                continue
        except Exception as e:
            logger.warning(f"Error al acceder a CONAF: {e}")
        
        # Estrategia 2: NASA FIRMS (datos satelitales de fuego activo)
        logger.info("Intentando descargar datos de NASA FIRMS como alternativa...")
        try:
            firms_data = self.download_nasa_firms_data(
                country='Chile',
                start_date='2020-01-01',
                end_date=datetime.now().strftime('%Y-%m-%d')
            )
            if firms_data is not None and len(firms_data) > 0:
                # Convertir a formato similar a CONAF
                df = self._convert_firms_to_conaf_format(firms_data)
                df.to_csv(save_path, index=False, encoding='utf-8-sig')
                logger.info(f"✅ Datos de fuego activo (NASA FIRMS) descargados y guardados")
                return df
        except Exception as e:
            logger.warning(f"Error al descargar NASA FIRMS: {e}")
        
        logger.warning("No se pudieron descargar datos reales de CONAF o alternativas")
        return None
    
    def download_cr2_climate_data(self, save_path: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Intenta descargar datos climáticos de CR2 (Centro de Ciencia del Clima y la Resiliencia)
        
        CR2 tiene datos climáticos públicos para Chile
        """
        logger.info("Intentando descargar datos climáticos de CR2...")
        
        if save_path is None:
            save_path = self.raw_dir / "cr2_clima_historico.csv"
        
        cr2_url = DATA_SOURCES.get("cr2_clima")
        
        try:
            # CR2 generalmente tiene una página de descarga de datos
            response = self.session.get(cr2_url, timeout=30)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Buscar enlaces a datos climáticos
                links = soup.find_all('a', href=True)
                for link in links:
                    href = link.get('href', '')
                    text = link.get_text().lower()
                    
                    # Buscar datos diarios o mensuales
                    if any(term in text for term in ['temperatura', 'precipitacion', 'datos', 'clima']):
                        if any(ext in href.lower() for ext in ['.csv', '.xlsx', '.xls', '.txt']):
                            try:
                                file_url = href if href.startswith('http') else cr2_url + href
                                df = self._download_file(file_url)
                                if df is not None:
                                    df.to_csv(save_path, index=False, encoding='utf-8-sig')
                                    logger.info(f"✅ Datos CR2 descargados y guardados en {save_path}")
                                    return df
                            except Exception as e:
                                logger.warning(f"No se pudo descargar desde {href}: {e}")
                                continue
        except Exception as e:
            logger.warning(f"Error al acceder a CR2: {e}")
        
        # Estrategia alternativa: Open-Meteo API (datos climáticos históricos)
        logger.info("Intentando descargar datos climáticos de Open-Meteo como alternativa...")
        try:
            weather_data = self.download_openmeteo_data()
            if weather_data is not None and len(weather_data) > 0:
                weather_data.to_csv(save_path, index=False, encoding='utf-8-sig')
                logger.info(f"✅ Datos climáticos (Open-Meteo) descargados")
                return weather_data
        except Exception as e:
            logger.warning(f"Error al descargar Open-Meteo: {e}")
        
        logger.warning("No se pudieron descargar datos climáticos reales")
        return None
    
    def download_nasa_firms_data(self, 
                                 country: str = 'Chile',
                                 start_date: str = '2020-01-01',
                                 end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Descarga datos de fuego activo de NASA FIRMS (Fire Information for Resource Management System)
        
        FIRMS proporciona datos satelitales de fuego activo casi en tiempo real
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"Descargando datos de NASA FIRMS para {country} ({start_date} a {end_date})...")
        
        # NASA FIRMS API
        firms_api_url = "https://firms.modaps.eosdis.nasa.gov/api/country/csv"
        
        # Parámetros para Chile
        params = {
            'country': country.lower().replace(' ', '%20'),
            'source': 'MODIS_NRT',  # o 'VIIRS_NRT'
            'date': start_date  # Formato YYYY-MM-DD
        }
        
        try:
            all_data = []
            current_date = datetime.strptime(start_date, '%Y-%m-%d')
            end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
            
                # Descargar día por día (para evitar límites de API)
            # Limitar a últimos 30 días para no sobrecargar
            start_date_obj = max(
                current_date, 
                end_date_obj - timedelta(days=30)
            )
            
            while start_date_obj <= end_date_obj:
                date_str = start_date_obj.strftime('%Y-%m-%d')
                params['date'] = date_str
                
                try:
                    response = self.session.get(firms_api_url, params=params, timeout=60)
                    
                    if response.status_code == 200:
                        # Leer CSV directamente desde la respuesta
                        from io import StringIO
                        df_day = pd.read_csv(StringIO(response.text))
                        if len(df_day) > 0:
                            all_data.append(df_day)
                            logger.info(f"  Datos descargados para {date_str}: {len(df_day)} detecciones")
                    
                    # Esperar un poco para no sobrecargar la API
                    time.sleep(1)
                    
                except Exception as e:
                    logger.warning(f"Error al descargar datos para {date_str}: {e}")
                
                start_date_obj += timedelta(days=1)
            
            if all_data:
                df = pd.concat(all_data, ignore_index=True)
                logger.info(f"✅ Total de detecciones de fuego descargadas: {len(df)}")
                return df
            else:
                logger.warning("No se encontraron datos de fuego activo")
                return None
                
        except Exception as e:
            logger.error(f"Error al descargar datos de NASA FIRMS: {e}")
            return None
    
    def download_openmeteo_data(self,
                               lat: float = -33.4489,  # Santiago
                               lon: float = -70.6693,
                               start_date: str = '2000-01-01',
                               end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Descarga datos climáticos históricos de Open-Meteo API
        
        Open-Meteo proporciona datos climáticos históricos gratuitos
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"Descargando datos climáticos de Open-Meteo ({start_date} a {end_date})...")
        
        # Open-Meteo Historical Weather API
        url = "https://archive-api.open-meteo.com/v1/archive"
        
        params = {
            'latitude': lat,
            'longitude': lon,
            'start_date': start_date,
            'end_date': end_date,
            'daily': 'temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max,relative_humidity_2m_max',
            'timezone': 'America/Santiago'
        }
        
        try:
            response = self.session.get(url, params=params, timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                
                # Convertir a DataFrame
                daily_data = data.get('daily', {})
                
                if daily_data:
                    df = pd.DataFrame({
                        'fecha': pd.to_datetime(daily_data.get('time', [])),
                        'temperatura_maxima': daily_data.get('temperature_2m_max', []),
                        'temperatura_minima': daily_data.get('temperature_2m_min', []),
                        'precipitacion': daily_data.get('precipitation_sum', []),
                        'velocidad_viento': daily_data.get('wind_speed_10m_max', []),
                        'humedad_relativa': daily_data.get('relative_humidity_2m_max', [])
                    })
                    
                    logger.info(f"✅ Datos climáticos descargados: {len(df)} días")
                    return df
                else:
                    logger.warning("No se encontraron datos climáticos")
                    return None
            else:
                logger.warning(f"Error al acceder a Open-Meteo: código {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error al descargar datos de Open-Meteo: {e}")
            return None
    
    def _download_file(self, url: str) -> Optional[pd.DataFrame]:
        """Descarga un archivo desde URL y lo convierte a DataFrame"""
        try:
            response = self.session.get(url, timeout=60, stream=True)
            
            if response.status_code == 200:
                # Intentar leer como CSV primero
                try:
                    df = pd.read_csv(url)
                    return df
                except:
                    pass
                
                # Intentar leer como Excel
                try:
                    df = pd.read_excel(url)
                    return df
                except:
                    pass
                
                logger.warning(f"No se pudo leer el archivo desde {url}")
                return None
            else:
                logger.warning(f"Error HTTP {response.status_code} al descargar {url}")
                return None
                
        except Exception as e:
            logger.error(f"Error al descargar archivo: {e}")
            return None
    
    def _convert_firms_to_conaf_format(self, firms_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convierte datos de NASA FIRMS a formato similar a CONAF
        """
        if len(firms_df) == 0:
            return pd.DataFrame()
        
        # Mapear columnas de FIRMS a formato CONAF
        result = pd.DataFrame()
        
        # Fecha
        if 'acq_date' in firms_df.columns:
            result['fecha_inicio'] = pd.to_datetime(firms_df['acq_date'])
        elif 'date' in firms_df.columns:
            result['fecha_inicio'] = pd.to_datetime(firms_df['date'])
        else:
            return pd.DataFrame()
        
        # Coordenadas (para mapear a comunas después)
        if 'latitude' in firms_df.columns:
            result['lat'] = firms_df['latitude']
        if 'longitude' in firms_df.columns:
            result['lon'] = firms_df['longitude']
        
        # Confianza de detección
        if 'confidence' in firms_df.columns:
            result['confianza'] = firms_df['confidence']
        
        # Área (aproximada, FIRMS no tiene área directamente)
        # Usar un valor base pequeño para cada detección
        result['area_quemada_ha'] = 0.1  # Estimación conservadora en hectáreas
        
        # Comuna (será necesario mapear coordenadas a comunas después)
        result['comuna'] = 'Por mapear'
        
        # Otros campos
        result['region'] = 'Por mapear'
        result['causa'] = 'Desconocida'
        result['duracion_dias'] = 1
        result['personas_afectadas'] = 0
        
        return result
    
    def download_all_data(self) -> Dict[str, Optional[pd.DataFrame]]:
        """
        Intenta descargar todos los datos disponibles
        
        Returns:
            Dict con los DataFrames descargados
        """
        results = {}
        
        logger.info("=" * 60)
        logger.info("Iniciando descarga de datos reales...")
        logger.info("=" * 60)
        
        # 1. Datos de incendios (CONAF)
        logger.info("\n[1/2] Descargando datos de incendios...")
        conaf_data = self.download_conaf_data()
        results['incendios'] = conaf_data
        
        # 2. Datos climáticos (CR2)
        logger.info("\n[2/2] Descargando datos climáticos...")
        clima_data = self.download_cr2_climate_data()
        results['clima'] = clima_data
        
        logger.info("\n" + "=" * 60)
        logger.info("Descarga de datos completada")
        logger.info("=" * 60)
        
        return results

