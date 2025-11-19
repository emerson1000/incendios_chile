"""
Script principal para ejecutar el pipeline completo de predicción y optimización
"""
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import logging
from datetime import datetime

from src.data.etl import FireDataETL
from src.models.prediction import FireRiskPredictor
from src.optimization.resource_allocation import ResourceAllocationOptimizer
from config import MODEL_CONFIG, OPTIMIZATION_CONFIG

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Sistema de Predicción y Optimización de Recursos para Incendios Forestales'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['etl', 'train', 'predict', 'optimize', 'full'],
        default='full',
        help='Modo de ejecución'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['xgboost', 'lightgbm', 'random_forest'],
        default='xgboost',
        help='Tipo de modelo'
    )
    parser.add_argument(
        '--max-brigades',
        type=int,
        default=OPTIMIZATION_CONFIG['max_brigades'],
        help='Número máximo de brigadas'
    )
    parser.add_argument(
        '--max-bases',
        type=int,
        default=OPTIMIZATION_CONFIG['max_bases'],
        help='Número máximo de bases'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Sistema de Predicción y Optimización de Incendios Forestales")
    logger.info("=" * 60)
    
    # Inicializar ETL
    etl = FireDataETL()
    
    # 1. ETL
    if args.mode in ['etl', 'full']:
        logger.info("\n[1/4] Ejecutando ETL...")
        
        # Cargar datos (intenta descargar datos reales primero)
        logger.info("Intentando cargar/descargar datos reales...")
        incendios_df = etl.load_conaf_data(try_download=True)
        clima_df = etl.load_climate_data(try_download=True)
        
        # Crear panel
        panel_df = etl.create_panel_data(incendios_df, clima_df)
        
        # Guardar
        etl.save_processed_data(panel_df)
        
        logger.info(f"✅ Panel creado: {len(panel_df):,} observaciones")
    
    # 2. Entrenar modelo
    if args.mode in ['train', 'full']:
        logger.info("\n[2/4] Entrenando modelo...")
        
        # Cargar panel
        panel_df = etl.load_processed_data()
        
        # Crear predictor
        predictor = FireRiskPredictor(model_type=args.model_type, task='classification')
        
        # Preparar datos
        X, y = predictor.prepare_features(panel_df)
        
        # Entrenar
        metrics = predictor.train(X, y)
        
        # Guardar modelo
        predictor.save_model()
        
        logger.info(f"✅ Modelo entrenado. Accuracy: {metrics.get('accuracy', 0):.4f}")
    
    # 3. Predicción
    if args.mode in ['predict', 'full']:
        logger.info("\n[3/4] Generando predicciones...")
        
        # Cargar datos
        panel_df = etl.load_processed_data()
        
        # Cargar modelo
        predictor = FireRiskPredictor(model_type=args.model_type, task='classification')
        
        # Buscar último modelo entrenado
        models_dir = Path("models")
        model_files = list(models_dir.glob(f"fire_risk_model_{args.model_type}_classification_*.pkl"))
        
        if model_files:
            latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
            predictor.load_model(str(latest_model))
            logger.info(f"Modelo cargado: {latest_model}")
        else:
            logger.warning("No se encontró modelo entrenado. Entrenando nuevo modelo...")
            X, y = predictor.prepare_features(panel_df)
            predictor.train(X, y)
        
        # Generar mapa de riesgo para fecha más reciente
        fecha_pred = panel_df['fecha'].max()
        risk_map = predictor.predict_risk_map(panel_df, fecha=fecha_pred)
        
        # Guardar mapa de riesgo
        output_path = Path("results") / f"risk_map_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        output_path.parent.mkdir(exist_ok=True)
        risk_map.to_csv(output_path, index=False)
        
        logger.info(f"✅ Mapa de riesgo generado: {len(risk_map)} comunas")
        logger.info(f"   Riesgo promedio: {risk_map['riesgo_probabilidad'].mean():.4f}")
        logger.info(f"   Guardado en: {output_path}")
    
    # 4. Optimización
    if args.mode in ['optimize', 'full']:
        logger.info("\n[4/4] Optimizando asignación de recursos...")
        
        # Cargar mapa de riesgo
        results_dir = Path("results")
        risk_map_files = list(results_dir.glob("risk_map_*.csv"))
        
        if not risk_map_files:
            logger.error("No se encontró mapa de riesgo. Ejecuta 'predict' primero.")
            return
        
        latest_risk_map = max(risk_map_files, key=lambda p: p.stat().st_mtime)
        risk_map = pd.read_csv(latest_risk_map)
        
        # Crear optimizer
        optimizer = ResourceAllocationOptimizer(
            max_brigades=args.max_brigades,
            max_bases=args.max_bases
        )
        
        # Preparar datos
        optimizer.prepare_data(risk_map)
        
        # Optimizar
        solution = optimizer.optimize()
        
        # Obtener mapa de asignación
        allocation_map = optimizer.get_allocation_map()
        
        # Guardar resultados
        output_path = Path("results") / f"allocation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        allocation_map.to_csv(output_path, index=False)
        
        logger.info(f"✅ Optimización completada:")
        logger.info(f"   Bases activas: {solution['total_bases_activas']}")
        logger.info(f"   Total brigadas: {solution['total_brigades']}")
        logger.info(f"   Tiempo respuesta promedio: {solution['tiempo_respuesta_promedio']:.2f} min")
        logger.info(f"   Guardado en: {output_path}")
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ Pipeline completado exitosamente")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

