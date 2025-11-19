"""
Script maestro para procesar datos CONAF y crear análisis avanzados
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Agregar src al path
sys.path.append(str(Path(__file__).parent))

from src.data.conaf_processor import CONAFDataProcessor
from src.data.etl import FireDataETL
from src.models.prediction import FireRiskPredictor
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    print("=" * 80)
    print("PROCESAMIENTO AVANZADO DE DATOS CONAF - CHILE")
    print("=" * 80)
    print()
    
    # 1. Cargar todos los archivos CONAF
    print("[PASO 1] Cargando archivos CONAF...")
    processor = CONAFDataProcessor()
    datasets = processor.load_all_conaf_files()
    
    if len(datasets) == 0:
        print("[ERROR] No se encontraron archivos CONAF")
        return
    
    # 2. Explorar estructura
    print("\n[PASO 2] Explorando estructura de datos...")
    structure = processor.explore_structure()
    print(f"\nArchivos cargados: {len(structure)}")
    
    # 3. Consolidar datos
    print("\n[PASO 3] Consolidando datos...")
    consolidated = processor.consolidate_all_data()
    
    if len(consolidated) == 0:
        print("[ERROR] No se pudo consolidar datos")
        return
    
    print(f"\n[OK] Dataset consolidado: {len(consolidated)} registros, {len(consolidated.columns)} columnas")
    
    # 4. Crear features avanzadas
    print("\n[PASO 4] Creando features avanzadas...")
    consolidated = processor.create_advanced_features(consolidated)
    
    # 5. Limpiar y validar
    print("\n[PASO 5] Limpiando y validando datos...")
    consolidated = processor.clean_and_validate(consolidated)
    
    processor.consolidated_df = consolidated
    
    # 6. Guardar dataset consolidado
    print("\n[PASO 6] Guardando dataset consolidado...")
    output_path = processor.save_consolidated_dataset()
    
    # 7. Estadísticas
    print("\n[PASO 7] Generando estadísticas...")
    stats = processor.get_statistics()
    
    print("\n" + "=" * 80)
    print("ESTADISTICAS DEL DATASET CONSOLIDADO")
    print("=" * 80)
    print(f"Total registros: {stats.get('total_registros', 0):,}")
    print(f"Total columnas: {stats.get('total_columnas', 0)}")
    print(f"Comunas unicas: {stats.get('comunas_unicas', 0)}")
    print(f"Completitud: {stats.get('porcentaje_completitud', 0):.2f}%")
    
    # 8. Análisis exploratorio avanzado
    print("\n[PASO 8] Analisis exploratorio avanzado...")
    perform_advanced_analysis(consolidated)
    
    # 9. Preparar para ML
    print("\n[PASO 9] Preparando datos para Machine Learning...")
    ml_ready_df = prepare_for_ml(consolidated)
    
    # 10. Entrenar modelo mejorado
    print("\n[PASO 10] Entrenando modelo avanzado...")
    train_advanced_model(ml_ready_df)
    
    print("\n" + "=" * 80)
    print("[OK] PROCESAMIENTO COMPLETADO EXITOSAMENTE")
    print("=" * 80)
    print(f"\nDataset consolidado guardado en: {output_path}")
    print("\nProximos pasos:")
    print("  1. Revisa el dataset consolidado en data/processed/")
    print("  2. Ejecuta: streamlit run dashboard.py")
    print("  3. Revisa los analisis en el dashboard")


def perform_advanced_analysis(df: pd.DataFrame):
    """Realiza análisis avanzados"""
    print("\n  Analisis de distribucion...")
    
    # Top 10 comunas con más incendios
    if 'comuna' in df.columns:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        fire_cols = [col for col in numeric_cols if any(term in col.lower() for term in ['incendio', 'ocurrencia'])]
        
        if fire_cols:
            fire_col = fire_cols[0]
            top_comunas = df.groupby('comuna')[fire_col].sum().sort_values(ascending=False).head(10)
            
            print(f"\n  Top 10 comunas con mas incendios (variable: {fire_col}):")
            for comuna, valor in top_comunas.items():
                print(f"    {comuna}: {valor:,.0f}")
    
    # Análisis temporal
    fire_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                 if any(term in col.lower() for term in ['incendio', 'ocurrencia'])]
    
    if ('año' in df.columns or 'anio' in df.columns) and fire_cols:
        year_col = 'año' if 'año' in df.columns else 'anio'
        fire_col = fire_cols[0]
        temporal = df.groupby(year_col)[fire_col].sum()
        print(f"\n  Incendios por ano:")
        print(f"    Promedio: {temporal.mean():.2f}")
        print(f"    Maximo: {temporal.max():.2f} ({temporal.idxmax()})")
        print(f"    Minimo: {temporal.min():.2f} ({temporal.idxmin()})")
    
    # Análisis mensual
    if 'mes' in df.columns and fire_cols:
        fire_col = fire_cols[0]
        mensual = df.groupby('mes')[fire_col].sum()
        print(f"\n  Patron mensual:")
        print(f"    Mes con mas incendios: {mensual.idxmax()} ({mensual.max():.2f})")
        print(f"    Mes con menos incendios: {mensual.idxmin()} ({mensual.min():.2f})")
    
    print("  [OK] Analisis completado")


def prepare_for_ml(df: pd.DataFrame) -> pd.DataFrame:
    """Prepara el dataset para Machine Learning"""
    ml_df = df.copy()
    
    # Seleccionar columnas relevantes
    numeric_cols = ml_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Eliminar columnas con muchos nulos
    ml_df = ml_df.dropna(thresh=len(ml_df) * 0.5, axis=1)
    
    # Codificar variables categóricas
    if 'comuna' in ml_df.columns:
        # One-hot encoding para comunas (si no son demasiadas)
        if ml_df['comuna'].nunique() < 50:
            ml_df = pd.get_dummies(ml_df, columns=['comuna'], prefix='comuna')
        else:
            # Usar label encoding si hay muchas comunas
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            ml_df['comuna_encoded'] = le.fit_transform(ml_df['comuna'].astype(str))
    
    if 'temporada' in ml_df.columns:
        ml_df = pd.get_dummies(ml_df, columns=['temporada'], prefix='temporada')
    
    return ml_df


def train_advanced_model(df: pd.DataFrame):
    """Entrena un modelo avanzado con los datos consolidados"""
    try:
        # Buscar variable objetivo
        fire_cols = [col for col in df.columns if any(term in col.lower() for term in ['incendio', 'ocurrencia'])]
        
        if not fire_cols:
            print("  [WARNING] No se encontro variable objetivo clara")
            return
        
        fire_col = fire_cols[0]
        
        # Preparar features
        feature_cols = [col for col in df.columns 
                       if col != fire_col and 
                       df[col].dtype in [np.int64, np.float64, np.bool_] and
                       df[col].notna().sum() > len(df) * 0.8]
        
        if len(feature_cols) < 3:
            print("  [WARNING] No hay suficientes features para entrenar")
            return
        
        # Preparar datos
        X = df[feature_cols].fillna(0)
        y = df[fire_col].fillna(0)
        
        # Crear variable binaria para clasificación (si hay incendio o no)
        y_binary = (y > 0).astype(int)
        
        print(f"  Features: {len(feature_cols)}")
        print(f"  Muestras: {len(X)}")
        print(f"  Tasa positiva: {y_binary.mean():.4%}")
        
        # Entrenar modelo
        predictor = FireRiskPredictor(model_type='xgboost', task='classification')
        
        # Dividir datos
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
        )
        
        # Ajustar modelo para usar la misma estructura
        X_train_df = pd.DataFrame(X_train, columns=feature_cols)
        X_test_df = pd.DataFrame(X_test, columns=feature_cols)
        
        # Entrenar
        metrics = predictor.train(
            pd.concat([X_train_df, X_test_df]),
            pd.concat([y_train, y_test]),
            validation_size=0.2,
            temporal_split=False
        )
        
        print(f"\n  Metricas del modelo:")
        print(f"    Accuracy: {metrics.get('accuracy', 0):.4f}")
        print(f"    F1-Score: {metrics.get('f1', 0):.4f}")
        print(f"    ROC-AUC: {metrics.get('roc_auc', 0):.4f}")
        
        # Guardar modelo
        model_path = predictor.save_model()
        print(f"\n  [OK] Modelo guardado: {model_path}")
        
    except Exception as e:
        print(f"  [ERROR] Error al entrenar modelo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
