"""
Script para descargar datos reales de incendios forestales y clima
"""
import sys
from pathlib import Path
import logging

# Agregar src al path
sys.path.append(str(Path(__file__).parent))

from src.data.downloaders import DataDownloader
from config import RAW_DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """
    Descarga datos reales de incendios y clima de fuentes públicas
    """
    print("=" * 60)
    print("Descarga de Datos Reales de Incendios Forestales - Chile")
    print("=" * 60)
    print()
    
    downloader = DataDownloader(raw_dir=RAW_DATA_DIR)
    
    # Descargar todos los datos disponibles
    results = downloader.download_all_data()
    
    # Resumen
    print()
    print("=" * 60)
    print("RESUMEN DE DESCARGA")
    print("=" * 60)
    
    if results.get('incendios') is not None and len(results['incendios']) > 0:
        print(f"\n[OK] Datos de incendios: {len(results['incendios'])} registros")
        print(f"     Guardado en: {RAW_DATA_DIR}")
    else:
        print("\n[ERROR] No se pudieron descargar datos de incendios")
        print("        Opciones:")
        print("        1. Descarga manualmente desde https://www.conaf.cl/")
        print("        2. Usa NASA FIRMS para datos satelitales")
        print("        3. Coloca archivos CSV en data/raw/")
    
    if results.get('clima') is not None and len(results['clima']) > 0:
        print(f"\n[OK] Datos climáticos: {len(results['clima'])} registros")
        print(f"     Guardado en: {RAW_DATA_DIR}")
    else:
        print("\n[ERROR] No se pudieron descargar datos climáticos")
        print("        Opciones:")
        print("        1. Descarga manualmente desde http://www.cr2.cl/")
        print("        2. Usa Open-Meteo API (se intenta automáticamente)")
        print("        3. Coloca archivos CSV en data/raw/")
    
    print()
    print("=" * 60)
    print("PRÓXIMOS PASOS")
    print("=" * 60)
    print()
    print("1. Revisa los archivos descargados en data/raw/")
    print("2. Ejecuta el pipeline:")
    print("   python example_usage.py")
    print("   o")
    print("   streamlit run dashboard.py")
    print()
    print("NOTA: Si no se descargaron datos reales:")
    print("- Verifica tu conexión a internet")
    print("- Descarga manualmente desde las fuentes oficiales")
    print("- Coloca los archivos CSV en data/raw/")
    print()


if __name__ == "__main__":
    main()

