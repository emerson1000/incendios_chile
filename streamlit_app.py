"""
Archivo principal para Streamlit Cloud
Este archivo es el punto de entrada que Streamlit Cloud busca automáticamente.
Carga dashboard.py con manejo robusto de errores.
"""
import streamlit as st
import sys
import traceback
from pathlib import Path

# Configuración básica de la página PRIMERO
# Esto debe estar ANTES de cualquier otro código de Streamlit
try:
    st.set_page_config(
        page_title="Sistema de Incendios Forestales - CONAF Chile",
        page_icon="🔥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
except Exception:
    # Si ya está configurado (puede pasar en reloads), ignorar
    pass

# Título básico que SIEMPRE se mostrará, incluso si hay errores
st.title("🔥 Sistema de Predicción y Optimización de Recursos para Incendios Forestales")
st.markdown("**Datos oficiales de CONAF - Chile (1985-2024)**")

# Intentar cargar y ejecutar el dashboard con manejo robusto de errores
try:
    # Verificar que dashboard.py existe
    dashboard_path = Path("dashboard.py")
    if not dashboard_path.exists():
        st.error("❌ **Error: No se encontró dashboard.py**")
        st.info("💡 Verifica que el archivo esté en el repositorio de GitHub")
        st.stop()
    
    # Agregar directorio actual al path
    current_dir = Path(__file__).parent.absolute()
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    
    # Importar el dashboard usando importlib para mejor control
    import importlib.util
    
    spec = importlib.util.spec_from_file_location("dashboard", dashboard_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo cargar dashboard.py desde {dashboard_path}")
    
    dashboard_module = importlib.util.module_from_spec(spec)
    
    # Ejecutar el módulo - esto ejecutará todo el código de dashboard.py
    spec.loader.exec_module(dashboard_module)
    
    # Si llegamos aquí, el dashboard se cargó correctamente
    # Los elementos del dashboard ya están renderizados por el módulo
    
except IndentationError as e:
    st.error("❌ **Error de indentación en el código**")
    st.exception(e)
    if hasattr(e, 'lineno'):
        st.warning(f"Error en línea {e.lineno}: {e.text}")
    with st.expander("🔍 Detalles del error", expanded=True):
        st.code(traceback.format_exc())
    
except SyntaxError as e:
    st.error("❌ **Error de sintaxis**")
    st.exception(e)
    if hasattr(e, 'lineno'):
        st.warning(f"Error en línea {e.lineno}: {e.text}")
    with st.expander("🔍 Detalles del error", expanded=True):
        st.code(traceback.format_exc())
    
except ImportError as e:
    st.error("❌ **Error al importar módulos**")
    st.exception(e)
    st.info("💡 Verifica que todos los archivos y dependencias estén en el repositorio")
    with st.expander("🔍 Detalles del error", expanded=True):
        st.code(traceback.format_exc())
    
except Exception as e:
    # Cualquier otro error durante la ejecución
    st.error("❌ **Error al cargar el dashboard**")
    st.exception(e)
    
    # Información de debug
    with st.expander("🔍 Información de Debug", expanded=True):
        st.write(f"**Tipo de error:** `{type(e).__name__}`")
        st.write(f"**Mensaje:** `{str(e)}`")
        st.write(f"**Directorio actual:** `{Path.cwd()}`")
        st.write(f"**Archivo streamlit_app.py:** `{Path(__file__).absolute()}`")
        st.write(f"**Dashboard existe:** `{Path('dashboard.py').exists()}`")
        
        if Path('dashboard.py').exists():
            st.success(f"✅ Dashboard encontrado: {Path('dashboard.py').stat().st_size:,} bytes")
        else:
            st.error("❌ Dashboard NO encontrado")
        
        st.code(traceback.format_exc())
    
    st.warning("""
    **Si ves este mensaje, hay un error en la ejecución del dashboard.**
    
    Por favor verifica:
    1. Que `dashboard.py` existe y está correcto
    2. Que todas las dependencias están instaladas
    3. Que los datos necesarios están disponibles
    """)
    
    # No usar st.stop() - mejor mostrar información útil
    st.info("💡 Revisa los logs de Streamlit Cloud para más detalles")
