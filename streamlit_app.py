"""
Archivo principal para Streamlit Cloud
Este archivo es el punto de entrada que Streamlit Cloud busca automáticamente.
Simplemente redirige a dashboard.py que contiene toda la lógica.
"""
# IMPORTANTE: NO llamar st.set_page_config() aquí porque dashboard.py ya lo hace
# st.set_page_config() solo puede llamarse una vez y debe ser lo primero

# Importar todo del dashboard - esto ejecutará todo el código de dashboard.py
# Si hay un error, Python lanzará una excepción y Streamlit la mostrará
from dashboard import *

