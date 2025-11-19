"""
Script para procesar datos CONAF con procesador inteligente
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.data.conaf_smart_processor import SmartCONAFProcessor
from src.models.prediction import FireRiskPredictor
import pandas as pd

def main():
    print("=" * 80)
    print("PROCESAMIENTO INTELIGENTE DE DATOS CONAF")
    print("=" * 80)
    print()
    
    # Procesar todos los archivos
    processor = SmartCONAFProcessor()
    consolidated_df = processor.process_all_files()
    
    if len(consolidated_df) > 0:
        print("\n" + "=" * 80)
        print("RESUMEN FINAL")
        print("=" * 80)
        print(f"Total registros: {len(consolidated_df):,}")
        print(f"Total columnas: {len(consolidated_df.columns)}")
        
        if 'comuna' in consolidated_df.columns:
            print(f"Comunas: {consolidated_df['comuna'].nunique()}")
        
        print("\nPrimeras filas del dataset:")
        print(consolidated_df.head(10))
        
        print("\nColumnas disponibles:")
        print(consolidated_df.columns.tolist())
        
        print("\nEstadisticas basicas:")
        print(consolidated_df.describe())
        
        print("\n[OK] Procesamiento completado!")
        print("\nDataset guardado en: data/processed/conaf_unified_dataset.parquet")
    else:
        print("\n[ERROR] No se pudieron procesar los archivos")

if __name__ == "__main__":
    main()

