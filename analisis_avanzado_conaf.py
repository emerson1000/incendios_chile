"""
Análisis avanzado y predicciones usando datos reales de CONAF
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent))

from src.data.conaf_smart_processor import SmartCONAFProcessor
from src.data.etl import FireDataETL
from src.models.prediction import FireRiskPredictor
from src.optimization.resource_allocation import ResourceAllocationOptimizer

print("=" * 80)
print("ANALISIS AVANZADO CON DATOS REALES CONAF")
print("=" * 80)
print()

# 1. Cargar datos consolidados
print("[1] Cargando datos consolidados CONAF...")
conaf_df = pd.read_csv("data/processed/conaf_unified_dataset.csv")
print(f"    Registros cargados: {len(conaf_df):,}")
print(f"    Columnas: {len(conaf_df.columns)}")
print(f"    Columnas disponibles: {conaf_df.columns.tolist()}")
print()

# 2. Análisis exploratorio avanzado
print("[2] Analisis exploratorio avanzado...")
print()

# Estadísticas descriptivas
if 'num_incendios' in conaf_df.columns:
    num_inc = pd.to_numeric(conaf_df['num_incendios'], errors='coerce')
    print(f"    Total incendios registrados: {num_inc.sum():,.0f}")
    print(f"    Promedio por registro: {num_inc.mean():.2f}")
    print(f"    Mediana: {num_inc.median():.2f}")
    print(f"    Maximo: {num_inc.max():.0f}")
    print()

# Top comunas con más incendios
if 'comuna' in conaf_df.columns and 'num_incendios' in conaf_df.columns:
    top_comunas = (conaf_df.groupby('comuna')['num_incendios']
                  .apply(lambda x: pd.to_numeric(x, errors='coerce').sum())
                  .sort_values(ascending=False)
                  .head(15))
    
    print(f"    Top 15 comunas con mas incendios:")
    for comuna, valor in top_comunas.items():
        if pd.notna(valor) and valor > 0:
            print(f"      {comuna}: {valor:,.0f}")
    print()

# Análisis temporal
if 'anio' in conaf_df.columns and 'num_incendios' in conaf_df.columns:
    temporal = (conaf_df.groupby('anio')['num_incendios']
               .apply(lambda x: pd.to_numeric(x, errors='coerce').sum())
               .sort_index())
    
    print(f"    Analisis temporal (por ano):")
    print(f"      Periodo: {temporal.index.min():.0f} - {temporal.index.max():.0f}")
    print(f"      Promedio anual: {temporal.mean():.2f}")
    print(f"      Ano con mas incendios: {temporal.idxmax():.0f} ({temporal.max():.0f})")
    print(f"      Ano con menos incendios: {temporal.idxmin():.0f} ({temporal.min():.0f})")
    print()

# Análisis mensual
if 'mes' in conaf_df.columns and 'num_incendios' in conaf_df.columns:
    mensual = (conaf_df.groupby('mes')['num_incendios']
              .apply(lambda x: pd.to_numeric(x, errors='coerce').sum())
              .sort_index())
    
    print(f"    Patron mensual:")
    meses_nombres = {1: 'Ene', 2: 'Feb', 3: 'Mar', 4: 'Abr', 5: 'May', 6: 'Jun',
                    7: 'Jul', 8: 'Ago', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dic'}
    for mes, valor in mensual.items():
        if pd.notna(valor):
            mes_nombre = meses_nombres.get(int(mes), f'Mes {int(mes)}')
            print(f"      {mes_nombre}: {valor:,.0f}")
    print()

# 3. Crear dataset para ML
print("[3] Preparando datos para Machine Learning...")

# Limpiar y preparar datos
ml_df = conaf_df.copy()

# Convertir num_incendios a numérico
if 'num_incendios' in ml_df.columns:
    ml_df['num_incendios'] = pd.to_numeric(ml_df['num_incendios'], errors='coerce').fillna(0)

# Crear variable objetivo (binaria: hay incendio o no)
ml_df['target_incendio'] = (ml_df['num_incendios'] > 0).astype(int)

# Features numéricas
numeric_features = []
for col in ml_df.columns:
    if col not in ['target_incendio', 'comuna', 'region', 'archivo_origen', 'hoja', 
                   'fecha_procesamiento', 'fuente', 'GERENCIA MANEJO DEL FUEGO']:
        if ml_df[col].dtype in [np.int64, np.float64]:
            numeric_features.append(col)
        else:
            # Intentar convertir
            numeric_vals = pd.to_numeric(ml_df[col], errors='coerce')
            if numeric_vals.notna().sum() > len(ml_df) * 0.3:
                ml_df[col] = numeric_vals
                numeric_features.append(col)

print(f"    Features numericas identificadas: {len(numeric_features)}")
print(f"    Features: {numeric_features[:10]}...")
print()

# Codificar comunas
if 'comuna' in ml_df.columns:
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    ml_df['comuna_encoded'] = le.fit_transform(ml_df['comuna'].astype(str))
    numeric_features.append('comuna_encoded')

# 4. Entrenar modelo avanzado
print("[4] Entrenando modelo avanzado con datos reales...")

# Preparar X e y
X = ml_df[numeric_features].fillna(0)
y = ml_df['target_incendio']

print(f"    Muestras: {len(X):,}")
print(f"    Features: {len(X.columns)}")
print(f"    Tasa positiva (con incendios): {y.mean():.4%}")
print()

# Entrenar modelo
if len(X) > 100 and y.sum() > 10:  # Tener suficientes datos
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    predictor = FireRiskPredictor(model_type='xgboost', task='classification')
    
    # Entrenar
    metrics = predictor.train(
        pd.concat([X_train, X_test]),
        pd.concat([y_train, y_test]),
        validation_size=0.2,
        temporal_split=False
    )
    
    print(f"    [OK] Modelo entrenado exitosamente!")
    print(f"    Accuracy: {metrics.get('accuracy', 0):.4f}")
    print(f"    Precision: {metrics.get('precision', 0):.4f}")
    print(f"    Recall: {metrics.get('recall', 0):.4f}")
    print(f"    F1-Score: {metrics.get('f1', 0):.4f}")
    if metrics.get('roc_auc'):
        print(f"    ROC-AUC: {metrics.get('roc_auc', 0):.4f}")
    print()
    
    # Feature importance
    if predictor.feature_importance is not None:
        print(f"    Top 10 features mas importantes:")
        top_features = predictor.feature_importance.head(10)
        for _, row in top_features.iterrows():
            print(f"      {row['feature']}: {row['importance']:.4f}")
    print()
    
    # Guardar modelo
    model_path = predictor.save_model()
    print(f"    Modelo guardado: {model_path}")
    print()
else:
    print(f"    [WARNING] No hay suficientes datos para entrenar modelo")
    print(f"             Muestras: {len(X)}, Incendios: {y.sum()}")
    print()

# 5. Generar predicciones por comuna
print("[5] Generando predicciones de riesgo por comuna...")

if 'predictor' in locals() and predictor.model is not None:
    # Agregar predicciones al dataset
    risk_scores = predictor.predict(X, return_proba=True)
    ml_df['riesgo_predicho'] = risk_scores
    
    # Top 10 comunas con mayor riesgo predicho
    if 'comuna' in ml_df.columns:
        riesgo_comuna = ml_df.groupby('comuna')['riesgo_predicho'].mean().sort_values(ascending=False).head(10)
        
        print(f"    Top 10 comunas con mayor riesgo predicho:")
        for comuna, riesgo in riesgo_comuna.items():
            categoria = 'Alto' if riesgo > 0.6 else 'Medio' if riesgo > 0.3 else 'Bajo'
            print(f"      {comuna}: {riesgo:.4f} ({categoria})")
        print()

# 6. Preparar para optimización
print("[6] Preparando datos para optimizacion de recursos...")

if 'comuna' in ml_df.columns and 'riesgo_predicho' in ml_df.columns:
    # Crear mapa de riesgo
    risk_map = ml_df.groupby('comuna').agg({
        'riesgo_predicho': 'mean',
        'num_incendios': lambda x: pd.to_numeric(x, errors='coerce').sum()
    }).reset_index()
    risk_map.columns = ['comuna', 'riesgo_probabilidad', 'num_incendios_total']
    
    # Agregar severidad esperada
    risk_map['severidad_esperada'] = risk_map['riesgo_probabilidad'] * risk_map['num_incendios_total'] * 100
    
    print(f"    Mapa de riesgo creado: {len(risk_map)} comunas")
    print(f"    Riesgo promedio: {risk_map['riesgo_probabilidad'].mean():.4f}")
    print(f"    Riesgo maximo: {risk_map['riesgo_probabilidad'].max():.4f}")
    print()
    
    # Optimizar recursos
    print(f"    Optimizando asignacion de recursos...")
    optimizer = ResourceAllocationOptimizer(max_brigades=50, max_bases=20)
    
    try:
        optimizer.prepare_data(risk_map)
        solution = optimizer.optimize(objective='minimize_damage')
        
        print(f"    [OK] Optimizacion completada!")
        print(f"    Bases activas: {solution['total_bases_activas']}")
        print(f"    Total brigadas: {solution['total_brigades']}")
        print(f"    Tiempo respuesta promedio: {solution['tiempo_respuesta_promedio']:.2f} min")
        print()
        
        # Guardar asignación
        allocation_map = optimizer.get_allocation_map()
        allocation_map.to_csv("results/allocation_conaf_real.csv", index=False, encoding='utf-8-sig')
        print(f"    Asignacion guardada en: results/allocation_conaf_real.csv")
        print()
        
    except Exception as e:
        print(f"    [WARNING] Error en optimizacion: {e}")
        print()

# 7. Guardar resultados
print("[7] Guardando resultados...")
ml_df.to_csv("data/processed/conaf_ml_ready.csv", index=False, encoding='utf-8-sig')
risk_map.to_csv("results/risk_map_conaf_real.csv", index=False, encoding='utf-8-sig')
print(f"    Dataset ML guardado: data/processed/conaf_ml_ready.csv")
print(f"    Mapa de riesgo guardado: results/risk_map_conaf_real.csv")
print()

print("=" * 80)
print("[OK] ANALISIS COMPLETADO EXITOSAMENTE!")
print("=" * 80)
print()
print("Resumen:")
print(f"  - Registros procesados: {len(conaf_df):,}")
print(f"  - Comunas analizadas: {conaf_df['comuna'].nunique() if 'comuna' in conaf_df.columns else 0}")
if 'num_incendios' in conaf_df.columns:
    total_inc = pd.to_numeric(conaf_df['num_incendios'], errors='coerce').sum()
    print(f"  - Total incendios: {total_inc:,.0f}")
print(f"  - Modelos entrenados: {1 if 'predictor' in locals() and predictor.model is not None else 0}")
print(f"  - Optimizaciones realizadas: {1 if 'solution' in locals() else 0}")
print()
print("Archivos generados:")
print("  - data/processed/conaf_ml_ready.csv")
print("  - results/risk_map_conaf_real.csv")
if 'solution' in locals():
    print("  - results/allocation_conaf_real.csv")
print()

