"""
Script para preparar datos para Streamlit Cloud
Descarga y procesa datos automáticamente si no están disponibles
"""
import sys
from pathlib import Path
import pandas as pd

def setup_data_for_streamlit():
    """
    Prepara datos para Streamlit Cloud
    Si los datos procesados no existen, intenta procesarlos
    """
    print("Verificando datos para Streamlit Cloud...")
    
    data_file = Path("data/processed/conaf_datos_reales_completo.csv")
    
    if data_file.exists():
        print(f"[OK] Datos encontrados: {data_file}")
        print(f"     Tamaño: {data_file.stat().st_size / 1024 / 1024:.2f} MB")
        return True
    else:
        print(f"[INFO] Datos no encontrados. Intentando procesar...")
        
        # Intentar procesar datos
        try:
            from src.data.conaf_smart_processor import SmartCONAFProcessor
            
            processor = SmartCONAFProcessor()
            consolidated = processor.process_all_files()
            
            if len(consolidated) > 0:
                print(f"[OK] Datos procesados: {len(consolidated)} registros")
                return True
            else:
                print("[WARNING] No se pudieron procesar datos automáticamente")
                return False
                
        except Exception as e:
            print(f"[ERROR] Error al procesar datos: {e}")
            print("\nPor favor ejecuta manualmente:")
            print("  python procesar_conaf_correctamente.py")
            return False

if __name__ == "__main__":
    setup_data_for_streamlit()

