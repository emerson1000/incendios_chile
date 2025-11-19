"""
Procesador inteligente de datos CONAF
Lee y procesa archivos CONAF con estructura compleja, detectando automáticamente
donde empiezan los datos reales y extrayendo toda la información relevante
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SmartCONAFProcessor:
    """Procesador inteligente para archivos CONAF"""
    
    def __init__(self, raw_dir: Path = RAW_DATA_DIR):
        self.raw_dir = raw_dir
        self.processed_dir = PROCESSED_DATA_DIR
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.all_data = []
        
    def process_all_files(self) -> pd.DataFrame:
        """
        Procesa todos los archivos CONAF inteligentemente
        """
        logger.info("=" * 80)
        logger.info("PROCESADOR INTELIGENTE DE DATOS CONAF")
        logger.info("=" * 80)
        
        files = list(self.raw_dir.glob("*.xls*"))
        files = [f for f in files if any(keyword in f.name.lower() for keyword in 
                                         ['ocurrencia', 'daño', 'hectárea', 'resumen', 'magnitud', 'horario'])]
        
        logger.info(f"\nArchivos encontrados: {len(files)}")
        
        for file_path in files:
            try:
                logger.info(f"\nProcesando: {file_path.name}")
                self._process_file_smart(file_path)
            except Exception as e:
                logger.error(f"Error procesando {file_path.name}: {e}")
                import traceback
                traceback.print_exc()
        
        if len(self.all_data) == 0:
            logger.warning("No se extrajeron datos de los archivos")
            return pd.DataFrame()
        
        # Consolidar todos los datos
        logger.info(f"\nConsolidando {len(self.all_data)} datasets extraídos...")
        
        # Limpiar DataFrames antes de concatenar
        cleaned_data = []
        for df in self.all_data:
            if len(df) > 0:
                # Eliminar columnas duplicadas
                df = df.loc[:, ~df.columns.duplicated(keep='first')]
                # Resetear índice
                df = df.reset_index(drop=True)
                cleaned_data.append(df)
        
        if len(cleaned_data) == 0:
            logger.warning("No hay datos válidos para consolidar")
            return pd.DataFrame()
        
        # Concatenar con manejo de columnas diferentes
        try:
            consolidated = pd.concat(cleaned_data, ignore_index=True, sort=False)
        except Exception as e:
            logger.warning(f"Error al concatenar: {e}. Intentando método alternativo...")
            # Intentar unir por columnas comunes
            common_cols = set(cleaned_data[0].columns)
            for df in cleaned_data[1:]:
                common_cols = common_cols.intersection(set(df.columns))
            
            if len(common_cols) > 0:
                logger.info(f"Usando {len(common_cols)} columnas comunes para consolidar")
                cleaned_data = [df[list(common_cols)] for df in cleaned_data]
                consolidated = pd.concat(cleaned_data, ignore_index=True, sort=False)
            else:
                logger.error("No hay columnas comunes para consolidar")
                return pd.DataFrame()
        
        logger.info(f"\n[OK] Dataset consolidado: {len(consolidated)} registros, {len(consolidated.columns)} columnas")
        
        # Limpiar y enriquecer
        consolidated = self._clean_and_enrich(consolidated)
        
        # Limpiar tipos de datos antes de guardar
        for col in consolidated.columns:
            if consolidated[col].dtype == 'object':
                # Intentar convertir a numérico si es posible
                numeric_vals = pd.to_numeric(consolidated[col], errors='coerce')
                if numeric_vals.notna().sum() > len(consolidated) * 0.5:
                    # Mayoría son numéricos
                    consolidated[col] = numeric_vals
                else:
                    # Mantener como string, limpiar valores
                    consolidated[col] = consolidated[col].astype(str).replace('nan', '')
        
        # Guardar
        output_path = self.processed_dir / "conaf_unified_dataset.parquet"
        try:
            consolidated.to_parquet(output_path, compression='snappy', index=False)
        except Exception as e:
            logger.warning(f"No se pudo guardar como parquet: {e}. Guardando solo CSV.")
            output_path = None
        
        csv_path = self.processed_dir / "conaf_unified_dataset.csv"
        consolidated.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        logger.info(f"\n[OK] Dataset guardado en: {csv_path}")
        if output_path:
            logger.info(f"     Parquet también disponible en: {output_path}")
        
        return consolidated
    
    def _process_file_smart(self, file_path: Path):
        """Procesa un archivo CONAF inteligentemente"""
        
        # Leer archivo Excel
        if file_path.suffix == '.xlsx':
            excel_file = pd.ExcelFile(file_path)
            for sheet_name in excel_file.sheet_names:
                df_raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
                self._extract_data_from_sheet(df_raw, file_path.name, sheet_name)
        else:
            # Archivo .xls
            df_raw = pd.read_excel(file_path, header=None, engine='xlrd')
            self._extract_data_from_sheet(df_raw, file_path.name, None)
    
    def _extract_data_from_sheet(self, df: pd.DataFrame, filename: str, sheet_name: Optional[str]):
        """Extrae datos de una hoja buscando inteligentemente dónde empiezan los datos"""
        
        # Buscar fila de headers
        header_row = None
        for i in range(min(20, len(df))):
            row = df.iloc[i]
            row_str = ' '.join([str(x).upper() for x in row.values if pd.notna(x)])
            
            # Buscar indicadores de headers
            if any(keyword in row_str for keyword in ['COMUNA', 'REGION', 'AÑO', 'ANO', 'MES', 'TEMPORADA', 
                                                     'INCENDIO', 'HECTAREA', 'AREA', 'OCURRENCIA', 'DANIO', 'DAÑO']):
                header_row = i
                break
        
        if header_row is None:
            # Si no se encuentra header, buscar dónde hay datos numéricos
            for i in range(min(20, len(df))):
                row = df.iloc[i]
                numeric_count = sum(1 for x in row.values if pd.notna(x) and isinstance(x, (int, float)))
                if numeric_count > 3:  # Si hay varios números, probablemente son datos
                    header_row = i - 1 if i > 0 else 0
                    break
        
        if header_row is None:
            logger.warning(f"  No se encontró header en {filename}")
            return
        
        # Extraer datos (empezar después del header)
        start_row = header_row + 1
        
        if start_row >= len(df):
            logger.warning(f"  No hay datos después del header en {filename}")
            return
        
        # Usar header_row como nombres de columnas
        try:
            df_clean = df.iloc[start_row:].copy()
            df_clean.columns = df.iloc[header_row].astype(str).str.strip()
        except:
            # Si falla, usar primeras columnas como datos
            df_clean = df.iloc[start_row:].copy()
            df_clean.columns = [f'col_{i}' for i in range(len(df_clean.columns))]
        
        # Identificar columnas relevantes
        df_clean = self._identify_columns(df_clean)
        
        # Agregar metadata
        df_clean['archivo_origen'] = filename
        df_clean['hoja'] = sheet_name if sheet_name else 'default'
        
        # Normalizar valores
        df_clean = self._normalize_values(df_clean)
        
        # Filtrar filas válidas (que tengan al menos un dato numérico relevante)
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            df_clean = df_clean[df_clean[numeric_cols].notna().any(axis=1)]
        
        if len(df_clean) > 0:
            self.all_data.append(df_clean)
            logger.info(f"  Extraídos {len(df_clean)} registros de {filename}")
    
    def _identify_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identifica y normaliza nombres de columnas"""
        
        column_mapping = {}
        
        for col in df.columns:
            col_str = str(col).lower()
            
            # Mapear a nombres estándar
            if any(term in col_str for term in ['comuna', 'comun']):
                column_mapping[col] = 'comuna'
            elif any(term in col_str for term in ['region', 'regi']):
                column_mapping[col] = 'region'
            elif any(term in col_str for term in ['año', 'ano', 'year', 'temporada']):
                column_mapping[col] = 'anio'
            elif any(term in col_str for term in ['mes', 'month']):
                column_mapping[col] = 'mes'
            elif any(term in col_str for term in ['incendio', 'ocurrencia', 'numero', 'número', 'cantidad']):
                if 'incendio' not in [v for v in column_mapping.values()]:
                    column_mapping[col] = 'num_incendios'
            elif any(term in col_str for term in ['hectarea', 'hectárea', 'ha', 'área', 'area', 'superficie']):
                if 'area_quemada' not in [v for v in column_mapping.values()]:
                    column_mapping[col] = 'area_quemada_ha'
            elif any(term in col_str for term in ['danio', 'daño', 'damage']):
                if 'danio' not in [v for v in column_mapping.values()]:
                    column_mapping[col] = 'danio_total'
            elif any(term in col_str for term in ['horario', 'hora', 'rango']):
                column_mapping[col] = 'rango_horario'
        
        # Aplicar mapeo
        df_renamed = df.rename(columns=column_mapping)
        
        return df_renamed
    
    def _normalize_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normaliza valores del DataFrame"""
        
        # Normalizar comuna
        if 'comuna' in df.columns:
            df['comuna'] = df['comuna'].astype(str).str.strip().str.title()
            # Eliminar filas inválidas
            invalid = df['comuna'].str.contains('CORPORACION|NACIONAL|FORESTAL|RESUMEN|PERIODO|REGION|TOTAL|NOTA|NAN',
                                               case=False, na=True)
            df = df[~invalid]
        
        # Normalizar región
        if 'region' in df.columns:
            df['region'] = df['region'].astype(str).str.strip().str.upper()
        
        # Convertir columnas numéricas
        numeric_cols = ['num_incendios', 'area_quemada_ha', 'danio_total', 'anio', 'mes']
        for col in numeric_cols:
            if col in df.columns:
                try:
                    if isinstance(df[col], pd.Series):
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    pass
        
        return df
    
    def _clean_and_enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpia y enriquece el dataset consolidado"""
        
        logger.info("\nLimpiando y enriqueciendo datos...")
        
        # 1. Limpiar nombres de columnas
        df.columns = [str(col) if pd.notna(col) and str(col).lower() != 'nan' else f'col_{i}' 
                     for i, col in enumerate(df.columns)]
        
        # 2. Eliminar columnas completamente vacías
        df = df.dropna(how='all', axis=1)
        
        # 3. Eliminar filas completamente vacías
        df = df.dropna(how='all')
        
        # 4. Eliminar duplicados
        df = df.drop_duplicates()
        
        # 5. Eliminar columnas con nombres inválidos
        invalid_cols = [col for col in df.columns if str(col).lower() == 'nan' or pd.isna(col)]
        if invalid_cols:
            df = df.drop(columns=invalid_cols)
        
        # 6. Crear features adicionales
        if 'mes' in df.columns:
            df['temporada'] = df['mes'].apply(
                lambda x: 'Alta' if x in [12, 1, 2, 3] 
                         else 'Media' if x in [4, 5, 10, 11] 
                         else 'Baja' if pd.notna(x) else None
            )
            
            df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
            df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)
        
        # 7. Estadísticas por comuna
        if 'comuna' in df.columns:
            if 'num_incendios' in df.columns:
                df['incendios_total_comuna'] = df.groupby('comuna')['num_incendios'].transform('sum')
            if 'area_quemada_ha' in df.columns:
                df['area_total_comuna'] = df.groupby('comuna')['area_quemada_ha'].transform('sum')
        
        # 8. Calcular intensidad
        if 'num_incendios' in df.columns and 'area_quemada_ha' in df.columns:
            df['intensidad_incendio'] = df['area_quemada_ha'] / (df['num_incendios'] + 1)
        
        # 9. Agregar fecha de procesamiento
        df['fecha_procesamiento'] = datetime.now()
        df['fuente'] = 'CONAF'
        
        logger.info(f"  Registros finales: {len(df)}")
        logger.info(f"  Columnas finales: {len(df.columns)}")
        
        # Estadísticas rápidas
        if 'num_incendios' in df.columns:
            try:
                num_inc = pd.to_numeric(df['num_incendios'], errors='coerce')
                total = num_inc.sum()
                if pd.notna(total):
                    logger.info(f"  Total incendios: {total:,.0f}")
            except:
                pass
        if 'area_quemada_ha' in df.columns:
            try:
                area = pd.to_numeric(df['area_quemada_ha'], errors='coerce')
                total = area.sum()
                if pd.notna(total):
                    logger.info(f"  Total area quemada: {total:,.0f} ha")
            except:
                pass
        if 'comuna' in df.columns:
            logger.info(f"  Comunas unicas: {df['comuna'].nunique()}")
        
        return df

