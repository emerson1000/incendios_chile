"""
Procesador avanzado de datos CONAF
Lee múltiples archivos de CONAF, los limpia, unifica y crea un dataset consolidado
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from config import RAW_DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CONAFDataProcessor:
    """Procesador avanzado para múltiples archivos CONAF"""
    
    def __init__(self, raw_dir: Path = RAW_DATA_DIR):
        self.raw_dir = raw_dir
        self.datasets = {}
        self.consolidated_df = None
        
    def load_all_conaf_files(self) -> Dict[str, pd.DataFrame]:
        """
        Carga todos los archivos CONAF disponibles
        """
        logger.info("=" * 60)
        logger.info("Cargando archivos CONAF...")
        logger.info("=" * 60)
        
        files = list(self.raw_dir.glob("*.xls*"))
        files = [f for f in files if 'conaf' not in f.name.lower() and 
                any(keyword in f.name.lower() for keyword in ['ocurrencia', 'daño', 'hectárea', 'resumen', 'magnitud', 'horario'])]
        
        datasets = {}
        
        for file_path in files:
            try:
                logger.info(f"\nProcesando: {file_path.name}")
                
                # Detectar tipo de archivo
                if file_path.suffix == '.xlsx':
                    # Intentar leer todas las hojas
                    excel_file = pd.ExcelFile(file_path)
                    for sheet_name in excel_file.sheet_names:
                        df = pd.read_excel(file_path, sheet_name=sheet_name)
                        key = f"{file_path.stem}_{sheet_name}"
                        datasets[key] = df
                        logger.info(f"  Hoja '{sheet_name}': {len(df)} filas, {len(df.columns)} columnas")
                else:
                    # Archivo .xls
                    try:
                        # Intentar leer primera hoja
                        df = pd.read_excel(file_path, engine='xlrd')
                        datasets[file_path.stem] = df
                        logger.info(f"  {len(df)} filas, {len(df.columns)} columnas")
                    except Exception as e:
                        logger.warning(f"  Error al leer {file_path.name}: {e}")
                        
            except Exception as e:
                logger.error(f"Error procesando {file_path.name}: {e}")
        
        self.datasets = datasets
        logger.info(f"\n✅ Total archivos cargados: {len(datasets)}")
        
        return datasets
    
    def explore_structure(self) -> pd.DataFrame:
        """
        Explora la estructura de todos los datasets
        """
        logger.info("\n" + "=" * 60)
        logger.info("EXPLORACIÓN DE ESTRUCTURA")
        logger.info("=" * 60)
        
        structure = []
        
        for key, df in self.datasets.items():
            structure.append({
                'archivo': key,
                'filas': len(df),
                'columnas': len(df.columns),
                'nombres_columnas': ', '.join(df.columns.astype(str).tolist()[:5]),
                'tipos': str(df.dtypes.value_counts().to_dict()),
                'nulos_total': df.isnull().sum().sum(),
                'muestra_filas': str(df.head(2).values.tolist()[:2])
            })
        
        structure_df = pd.DataFrame(structure)
        
        for _, row in structure_df.iterrows():
            logger.info(f"\n📊 {row['archivo']}")
            logger.info(f"   Filas: {row['filas']}, Columnas: {row['columnas']}")
            logger.info(f"   Columnas: {row['nombres_columnas']}...")
        
        return structure_df
    
    def process_historical_occurrence(self) -> pd.DataFrame:
        """
        Procesa archivo de Ocurrencia y Daño Histórico Nacional
        """
        key = [k for k in self.datasets.keys() if 'ocurrencia' in k.lower() and 'daño' in k.lower() and 'histórico' in k.lower()]
        
        if not key:
            logger.warning("No se encontró archivo de Ocurrencia y Daño Histórico")
            return pd.DataFrame()
        
        df = self.datasets[key[0]].copy()
        
        logger.info(f"\n🔍 Procesando Ocurrencia y Daño Histórico...")
        logger.info(f"   Estructura inicial: {df.shape}")
        
        # Limpiar nombres de columnas
        df.columns = df.columns.str.strip().str.lower()
        
        # Buscar columna de año
        year_cols = [col for col in df.columns if any(term in col.lower() for term in ['año', 'anio', 'year', 'temporada'])]
        
        # Buscar columnas de incendios
        fire_cols = [col for col in df.columns if any(term in col.lower() for term in ['incendio', 'ocurrencia', 'número', 'numero'])]
        
        # Buscar columnas de área
        area_cols = [col for col in df.columns if any(term in col.lower() for term in ['hectárea', 'hectarea', 'área', 'area', 'ha'])]
        
        logger.info(f"   Columnas de año: {year_cols}")
        logger.info(f"   Columnas de incendios: {fire_cols}")
        logger.info(f"   Columnas de área: {area_cols}")
        
        # Si el formato es años como columnas, transponer
        if len(year_cols) == 1 and len(fire_cols) > 1:
            # Intentar pivot
            if year_cols[0] in df.columns:
                df_melted = df.melt(id_vars=[year_cols[0]], 
                                  value_vars=fire_cols + area_cols,
                                  var_name='variable',
                                  value_name='valor')
                return df_melted
        
        return df
    
    def process_comuna_summary(self) -> pd.DataFrame:
        """
        Procesa Resumen de Ocurrencia y Daño por Comuna
        """
        key = [k for k in self.datasets.keys() if 'comuna' in k.lower() and 'resumen' in k.lower()]
        
        if not key:
            logger.warning("No se encontró archivo de Resumen por Comuna")
            return pd.DataFrame()
        
        df = self.datasets[key[0]].copy()
        
        logger.info(f"\n[INFO] Procesando Resumen por Comuna...")
        logger.info(f"   Estructura inicial: {df.shape}")
        logger.info(f"   Primeras 5 filas:\n{df.head()}")
        logger.info(f"   Columnas: {df.columns.tolist()[:15]}")
        
        # Los archivos CONAF suelen tener headers en filas iniciales
        # Buscar la fila que tiene los nombres de columnas reales
        for i in range(min(10, len(df))):
            row = df.iloc[i]
            # Buscar si esta fila tiene "COMUNA" o nombre de comuna
            row_str = ' '.join([str(x).upper() for x in row.values if pd.notna(x)])
            if 'COMUNA' in row_str or any(len(str(x)) > 15 for x in row.values if pd.notna(x) and isinstance(x, str)):
                # Esta podría ser la fila de headers
                df.columns = df.iloc[i].astype(str)
                df = df.iloc[i+1:].reset_index(drop=True)
                break
        
        # Limpiar nombres de columnas
        df.columns = df.columns.str.strip().str.title()
        
        # Buscar columna de comuna
        comuna_col = None
        for col in df.columns:
            if 'comuna' in str(col).lower():
                comuna_col = col
                break
        
        # Si no se encuentra, buscar en la primera columna
        if comuna_col is None:
            # La primera columna suele ser comuna en archivos CONAF
            first_col = df.columns[0]
            if df[first_col].dtype == 'object' and df[first_col].nunique() > 10:
                comuna_col = first_col
        
        if comuna_col:
            df = df.rename(columns={comuna_col: 'comuna'})
            df['comuna'] = df['comuna'].astype(str).str.strip().str.title()
            # Eliminar filas donde comuna no es válida
            df = df[df['comuna'].notna()]
            df = df[df['comuna'] != 'Nan']
            df = df[~df['comuna'].str.contains('CORPORACION|NACIONAL|FORESTAL|RESUMEN', case=False, na=False)]
        else:
            logger.warning("No se pudo identificar columna de comuna")
        
        # Convertir columnas numéricas
        for col in df.columns:
            if col != 'comuna' and col in df.columns:
                try:
                    if isinstance(df[col], pd.Series):
                        df[col] = pd.to_numeric(df[col], errors='ignore')
                except:
                    pass
        
        logger.info(f"   Estructura final: {df.shape}")
        logger.info(f"   Comunas encontradas: {df['comuna'].nunique() if 'comuna' in df.columns else 0}")
        
        return df
    
    def process_monthly_data(self) -> pd.DataFrame:
        """
        Procesa datos mensuales de ocurrencia y daño
        """
        monthly_files = [k for k in self.datasets.keys() if 'mes' in k.lower()]
        
        all_monthly = []
        
        for key in monthly_files:
            df = self.datasets[key].copy()
            
            logger.info(f"\n🔍 Procesando datos mensuales: {key}")
            logger.info(f"   Estructura: {df.shape}")
            
            # Limpiar columnas
            df.columns = df.columns.str.strip().str.lower()
            
            # Buscar columnas de mes
            month_col = [col for col in df.columns if 'mes' in col.lower()][0] if any('mes' in col.lower() for col in df.columns) else None
            
            # Buscar columnas de año
            year_col = [col for col in df.columns if any(term in col.lower() for term in ['año', 'anio', 'year'])][0] if any(term in col.lower() for term in ['año', 'anio', 'year'] for col in df.columns) else None
            
            df['tipo_archivo'] = key
            
            all_monthly.append(df)
        
        if all_monthly:
            combined = pd.concat(all_monthly, ignore_index=True)
            return combined
        
        return pd.DataFrame()
    
    def process_hourly_data(self) -> pd.DataFrame:
        """
        Procesa datos por rango horario
        """
        key = [k for k in self.datasets.keys() if 'horario' in k.lower() or 'rango' in k.lower()]
        
        if not key:
            return pd.DataFrame()
        
        df = self.datasets[key[0]].copy()
        
        logger.info(f"\n🔍 Procesando datos por rango horario...")
        logger.info(f"   Estructura: {df.shape}")
        
        # Limpiar columnas
        df.columns = df.columns.str.strip().str.lower()
        
        return df
    
    def consolidate_all_data(self) -> pd.DataFrame:
        """
        Consolida todos los datasets en uno solo
        """
        logger.info("\n" + "=" * 60)
        logger.info("CONSOLIDANDO DATOS")
        logger.info("=" * 60)
        
        # 1. Procesar resumen por comuna (base principal)
        comuna_df = self.process_comuna_summary()
        
        # 2. Procesar datos mensuales
        monthly_df = self.process_monthly_data()
        
        # 3. Procesar datos históricos
        historical_df = self.process_historical_occurrence()
        
        # 4. Procesar datos horarios
        hourly_df = self.process_hourly_data()
        
        logger.info(f"\n📊 Datasets procesados:")
        logger.info(f"   Comuna: {len(comuna_df)} registros")
        logger.info(f"   Mensual: {len(monthly_df)} registros")
        logger.info(f"   Histórico: {len(historical_df)} registros")
        logger.info(f"   Horario: {len(hourly_df)} registros")
        
        # Crear dataset consolidado
        # Usar comuna como base si existe
        if len(comuna_df) > 0:
            consolidated = comuna_df.copy()
            
            # Agregar metadata de otros datasets
            if len(monthly_df) > 0 and 'comuna' in monthly_df.columns:
                # Agregar estadísticas mensuales
                try:
                    monthly_stats = monthly_df.groupby('comuna').agg({
                        col: ['sum', 'mean', 'max'] 
                        for col in monthly_df.select_dtypes(include=[np.number]).columns[:5]  # Limitar a 5 columnas
                    }).reset_index()
                    
                    # Flatten column names
                    monthly_stats.columns = ['_'.join(col).strip() if col[1] else col[0] 
                                           for col in monthly_stats.columns.values]
                    
                    if 'comuna' in consolidated.columns:
                        consolidated = consolidated.merge(
                            monthly_stats,
                            on='comuna',
                            how='left',
                            suffixes=('', '_mensual')
                        )
                except Exception as e:
                    logger.warning(f"No se pudieron agregar estadísticas mensuales: {e}")
        else:
            # Si no hay datos de comuna, usar mensual como base
            consolidated = monthly_df.copy() if len(monthly_df) > 0 else pd.DataFrame()
        
        # Agregar metadatos
        consolidated['fecha_procesamiento'] = datetime.now()
        consolidated['fuente'] = 'CONAF'
        
        self.consolidated_df = consolidated
        
        logger.info(f"\n✅ Dataset consolidado creado: {len(consolidated)} registros, {len(consolidated.columns)} columnas")
        
        return consolidated
    
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea features avanzadas para ML
        """
        logger.info("\n" + "=" * 60)
        logger.info("CREANDO FEATURES AVANZADAS")
        logger.info("=" * 60)
        
        df = df.copy()
        
        # 1. Features temporales
        if 'mes' in df.columns:
            df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
            df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)
            df['temporada'] = df['mes'].apply(lambda x: 'Alta' if x in [12,1,2,3] else 'Media' if x in [4,5,10,11] else 'Baja')
        
        # 2. Features de intensidad
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        fire_cols = [col for col in numeric_cols if any(term in col.lower() for term in ['incendio', 'ocurrencia', 'número'])]
        area_cols = [col for col in numeric_cols if any(term in col.lower() for term in ['hectárea', 'hectarea', 'área', 'area'])]
        
        if fire_cols and area_cols:
            for fire_col in fire_cols[:1]:  # Tomar primera columna de incendios
                for area_col in area_cols[:1]:  # Tomar primera columna de área
                    df[f'intensidad_incendios_{fire_col}'] = df[fire_col] / (df[area_col] + 1)
                    df[f'area_promedio_por_incendio'] = df[area_col] / (df[fire_col] + 1)
        
        # 3. Features estadísticas por comuna
        if 'comuna' in df.columns:
            for col in numeric_cols[:5]:  # Primeras 5 columnas numéricas
                if col != 'comuna':
                    # Media por comuna
                    df[f'{col}_media_comuna'] = df.groupby('comuna')[col].transform('mean')
                    # Desviación estándar
                    df[f'{col}_std_comuna'] = df.groupby('comuna')[col].transform('std').fillna(0)
                    # Rango
                    df[f'{col}_rango_comuna'] = df.groupby('comuna')[col].transform('max') - df.groupby('comuna')[col].transform('min')
        
        # 4. Features de tendencia
        if 'año' in df.columns or 'anio' in df.columns:
            year_col = 'año' if 'año' in df.columns else 'anio'
            if fire_cols:
                for fire_col in fire_cols[:1]:
                    # Tendencia lineal
                    df_sorted = df.sort_values(year_col)
                    df[f'tendencia_{fire_col}'] = df_sorted.groupby('comuna' if 'comuna' in df.columns else df_sorted.index)[fire_col].transform(
                        lambda x: np.polyfit(range(len(x)), x.fillna(0), 1)[0] if len(x) > 1 else 0
                    )
        
        # 5. Features de riesgo categórico
        if fire_cols:
            fire_col = fire_cols[0]
            df['riesgo_categoria'] = pd.qcut(
                df[fire_col].fillna(0),
                q=3,
                labels=['Bajo', 'Medio', 'Alto'],
                duplicates='drop'
            )
        
        logger.info(f"✅ Features creadas. Total columnas: {len(df.columns)}")
        
        return df
    
    def clean_and_validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpia y valida el dataset consolidado
        """
        logger.info("\n" + "=" * 60)
        logger.info("LIMPIEZA Y VALIDACION")
        logger.info("=" * 60)
        
        initial_rows = len(df)
        
        # 1. Eliminar columnas duplicadas por nombre
        df = df.loc[:, ~df.columns.duplicated(keep='first')]
        
        # 2. Renombrar columnas inválidas
        df.columns = [str(col) if not pd.isna(col) and str(col) != 'nan' else f'col_{i}' 
                     for i, col in enumerate(df.columns)]
        
        # 3. Eliminar filas completamente vacías
        df = df.dropna(how='all')
        
        # 4. Limpiar nombres de comunas
        if 'comuna' in df.columns:
            df['comuna'] = df['comuna'].astype(str).str.strip().str.title()
            df = df[df['comuna'] != 'Nan']
            df = df[df['comuna'] != '']
            df = df[~df['comuna'].str.contains('CORPORACION|NACIONAL|FORESTAL|RESUMEN|PERIODO|REGION', 
                                             case=False, na=False)]
        
        # 5. Normalizar valores numéricos
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            # Reemplazar infinitos
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            # Reemplazar negativos por 0 (para áreas y conteos)
            if any(term in col.lower() for term in ['hectárea', 'hectarea', 'área', 'area', 'incendio', 'ocurrencia']):
                df[col] = df[col].clip(lower=0)
        
        # 6. Eliminar duplicados
        df = df.drop_duplicates()
        
        # 7. Imputar valores faltantes
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                if col.startswith('tendencia') or col.startswith('std'):
                    df[col] = df[col].fillna(0)
                else:
                    median_val = df[col].median()
                    if pd.notna(median_val):
                        df[col] = df[col].fillna(median_val)
                    else:
                        df[col] = df[col].fillna(0)
        
        final_rows = len(df)
        
        logger.info(f"   Filas iniciales: {initial_rows}")
        logger.info(f"   Filas finales: {final_rows}")
        logger.info(f"   Filas eliminadas: {initial_rows - final_rows}")
        logger.info(f"   Columnas: {len(df.columns)}")
        
        return df
    
    def save_consolidated_dataset(self, filename: str = "conaf_consolidado.parquet") -> Path:
        """
        Guarda el dataset consolidado
        """
        if self.consolidated_df is None:
            raise ValueError("Primero debe ejecutar consolidate_all_data()")
        
        output_path = self.raw_dir.parent / "processed" / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.consolidated_df.to_parquet(output_path, compression='snappy', index=False)
        logger.info(f"\n✅ Dataset consolidado guardado en: {output_path}")
        
        # También guardar como CSV para fácil inspección
        csv_path = output_path.with_suffix('.csv')
        self.consolidated_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        logger.info(f"✅ También guardado como CSV: {csv_path}")
        
        return output_path
    
    def get_statistics(self) -> Dict:
        """
        Genera estadísticas del dataset consolidado
        """
        if self.consolidated_df is None:
            return {}
        
        df = self.consolidated_df
        
        stats = {
            'total_registros': len(df),
            'total_columnas': len(df.columns),
            'columnas_numericas': len(df.select_dtypes(include=[np.number]).columns),
            'comunas_unicas': df['comuna'].nunique() if 'comuna' in df.columns else 0,
            'valores_faltantes': df.isnull().sum().sum(),
            'porcentaje_completitud': (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        }
        
        # Estadísticas por columnas numéricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats['estadisticas_numericas'] = df[numeric_cols].describe().to_dict()
        
        return stats
