"""
Ejemplo de uso del sistema de predicción y optimización de incendios forestales
"""
import pandas as pd
from datetime import datetime

from src.data.etl import FireDataETL
from src.models.prediction import FireRiskPredictor
from src.optimization.resource_allocation import ResourceAllocationOptimizer

def main():
    print("=" * 60)
    print("Ejemplo de Uso: Sistema de Incendios Forestales")
    print("=" * 60)
    
    # 1. ETL
    print("\n[1/4] Cargando y procesando datos...")
    etl = FireDataETL()
    
    # Cargar datos (intenta descargar datos reales primero)
    print("  Intentando cargar/descargar datos reales...")
    incendios_df = etl.load_conaf_data(try_download=True)
    clima_df = etl.load_climate_data(try_download=True)
    
    print(f"  - Incendios cargados: {len(incendios_df)}")
    print(f"  - Registros climáticos: {len(clima_df)}")
    
    # Crear panel espacio-temporal
    panel_df = etl.create_panel_data(incendios_df, clima_df)
    print(f"  - Panel creado: {len(panel_df):,} observaciones")
    print(f"  - Comunas: {panel_df['comuna'].nunique()}")
    
    # Guardar datos procesados
    etl.save_processed_data(panel_df)
    print("  [OK] Datos procesados y guardados")
    
    # 2. Entrenar modelo
    print("\n[2/4] Entrenando modelo de predicción...")
    predictor = FireRiskPredictor(model_type='xgboost', task='classification')
    
    # Preparar features
    X, y = predictor.prepare_features(panel_df)
    print(f"  - Features: {len(X.columns)}")
    print(f"  - Muestras: {len(X):,}")
    print(f"  - Tasa positiva: {y.mean():.4%}")
    
    # Entrenar
    metrics = predictor.train(X, y, validation_size=0.2, temporal_split=True)
    print(f"  [OK] Modelo entrenado:")
    print(f"     - Accuracy: {metrics.get('accuracy', 0):.4f}")
    print(f"     - F1-Score: {metrics.get('f1', 0):.4f}")
    
    # Guardar modelo
    model_path = predictor.save_model()
    print(f"  [OK] Modelo guardado: {model_path}")
    
    # 3. Generar predicciones
    print("\n[3/4] Generando mapa de riesgo...")
    
    # Usar fecha más reciente disponible
    fecha_pred = panel_df['fecha'].max()
    print(f"  - Fecha de predicción: {fecha_pred.date()}")
    
    risk_map = predictor.predict_risk_map(panel_df, fecha=fecha_pred)
    print(f"  - Comunas evaluadas: {len(risk_map)}")
    print(f"  - Riesgo promedio: {risk_map['riesgo_probabilidad'].mean():.4f}")
    print(f"  - Riesgo máximo: {risk_map['riesgo_probabilidad'].max():.4f}")
    
    # Mostrar top 5 comunas con mayor riesgo
    print("\n  Top 5 comunas con mayor riesgo:")
    top_5 = risk_map.nlargest(5, 'riesgo_probabilidad')[['comuna', 'riesgo_probabilidad', 'riesgo_categoria']]
    for idx, row in top_5.iterrows():
        print(f"    - {row['comuna']}: {row['riesgo_probabilidad']:.4f} ({row['riesgo_categoria']})")
    
    # Guardar mapa de riesgo
    from pathlib import Path
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    risk_map_path = results_dir / f"risk_map_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    risk_map.to_csv(risk_map_path, index=False)
    print(f"  [OK] Mapa de riesgo guardado: {risk_map_path}")
    
    # 4. Optimizar asignación de recursos
    print("\n[4/4] Optimizando asignación de recursos...")
    
    # Agregar severidad esperada (si no existe)
    if 'severidad_esperada' not in risk_map.columns:
        risk_map['severidad_esperada'] = risk_map['riesgo_probabilidad'] * 1000  # Hectáreas esperadas
    
    # Crear optimizador
    optimizer = ResourceAllocationOptimizer(
        max_brigades=50,
        max_bases=15
    )
    
    # Preparar datos
    optimizer.prepare_data(risk_map)
    print(f"  - Bases posibles: {len(optimizer.base_locations)}")
    
    # Optimizar
    solution = optimizer.optimize(objective='minimize_damage')
    print(f"  [OK] Optimización completada:")
    print(f"     - Estado: {solution['status']}")
    print(f"     - Bases activas: {solution['total_bases_activas']}")
    print(f"     - Total brigadas: {solution['total_brigades']}")
    print(f"     - Tiempo respuesta promedio: {solution['tiempo_respuesta_promedio']:.2f} min")
    print(f"     - Tiempo respuesta ponderado: {solution['tiempo_respuesta_ponderado']:.2f} min")
    
    # Obtener mapa de asignación
    allocation_map = optimizer.get_allocation_map()
    print(f"\n  Distribución de brigadas por base:")
    brigadas_por_base = pd.DataFrame(
        list(solution['brigadas_por_base'].items()),
        columns=['Base', 'Brigadas']
    ).sort_values('Brigadas', ascending=False)
    
    for _, row in brigadas_por_base.iterrows():
        print(f"    - {row['Base']}: {row['Brigadas']} brigadas")
    
    # Guardar asignación
    allocation_path = results_dir / f"allocation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    allocation_map.to_csv(allocation_path, index=False)
    print(f"\n  [OK] Asignación guardada: {allocation_path}")
    
    print("\n" + "=" * 60)
    print("[OK] Pipeline completado exitosamente!")
    print("=" * 60)
    print("\nPróximos pasos:")
    print("  1. Ejecuta 'streamlit run dashboard.py' para ver el dashboard interactivo")
    print("  2. Revisa los notebooks en 'notebooks/' para análisis más detallados")
    print("  3. Modifica 'config.py' para ajustar parámetros del modelo")

if __name__ == "__main__":
    main()

