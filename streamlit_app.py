"""
Archivo principal para Streamlit Cloud
Este archivo es el punto de entrada que Streamlit Cloud busca automáticamente.
Simplemente redirige a dashboard.py que contiene toda la lógica.
"""
import streamlit as st
import sys
import traceback
from pathlib import Path

# Configuración básica de la página primero
try:
    st.set_page_config(
        page_title="Sistema de Incendios Forestales - CONAF Chile",
        page_icon="🔥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
except Exception:
    # Si ya está configurado, ignorar
    pass

# Intentar importar y ejecutar el dashboard
try:
    # Importar todo del dashboard - esto ejecutará todo el código de dashboard.py
    from dashboard import *
except Exception as e:
    # Si hay un error, mostrar mensaje útil en lugar de página en blanco
    st.error("❌ Error al cargar el dashboard")
    st.exception(e)
    
    with st.expander("🔍 Detalles técnicos del error"):
        st.code(traceback.format_exc())
    
    st.info("""
    **Posibles soluciones:**
    1. Verifica que todos los archivos estén en el repositorio
    2. Verifica que `data/processed/conaf_datos_reales_completo.csv` exista
    3. Revisa los logs de Streamlit Cloud para más detalles
    """)
    
    # Mostrar información de debug
    st.sidebar.header("🔧 Información de Debug")
    st.sidebar.write(f"**Directorio actual:** {Path.cwd()}")
    st.sidebar.write(f"**Archivo actual:** {Path(__file__).absolute()}")
    
    # Verificar archivos importantes
    files_to_check = [
        "dashboard.py",
        "data/processed/conaf_datos_reales_completo.csv",
        "config.py"
    ]
    
    st.sidebar.subheader("Archivos importantes:")
    for file_path in files_to_check:
        path = Path(file_path)
        exists = path.exists()
        st.sidebar.write(f"{'✅' if exists else '❌'} {file_path}")
        if exists:
            st.sidebar.write(f"   Tamaño: {path.stat().st_size:,} bytes")

