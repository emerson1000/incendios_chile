"""
Archivo principal para Streamlit Cloud
Este archivo es el punto de entrada que Streamlit Cloud busca automáticamente.
Redirige a dashboard.py que contiene toda la lógica del dashboard.
"""
import streamlit as st
import sys
import os
from pathlib import Path

# Obtener el directorio del archivo actual
try:
    current_dir = Path(__file__).parent
except NameError:
    # Si __file__ no está definido, usar directorio actual
    current_dir = Path(os.getcwd())

# Agregar el directorio actual al path
sys.path.insert(0, str(current_dir))

# Importar y ejecutar el dashboard
try:
    dashboard_path = current_dir / 'dashboard.py'
    if dashboard_path.exists():
        # Ejecutar el dashboard
        with open(dashboard_path, 'r', encoding='utf-8') as f:
            dashboard_code = f.read()
        exec(dashboard_code, globals())
    else:
        st.error("❌ Error: No se encontró dashboard.py en el repositorio.")
        st.info("💡 Por favor verifica que el archivo dashboard.py existe en la raíz del proyecto.")
except Exception as e:
    st.error(f"❌ Error al cargar el dashboard: {e}")
    import traceback
    with st.expander("🔍 Detalles del error"):
        st.code(traceback.format_exc())

