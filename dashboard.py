"""
Dashboard interactivo profesional para predicción y optimización de recursos contra incendios
Usa datos reales de CONAF con filtros interactivos para investigadores
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
try:
    import folium
    from streamlit_folium import folium_static
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False
from pathlib import Path
import sys

# Agregar src al path
sys.path.append(str(Path(__file__).parent))

try:
    from src.models.prediction import FireRiskPredictor
    from src.optimization.resource_allocation import ResourceAllocationOptimizer
except ImportError as e:
    st.error(f"Error al importar módulos: {e}")
    st.stop()

# Configuración de la página
# NOTA: Esta configuración puede ya haberse hecho en streamlit_app.py
# Si ya está configurado, esta llamada será ignorada sin causar error
try:
    st.set_page_config(
        page_title="Sistema de Incendios Forestales - CONAF Chile",
        page_icon="🔥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
except Exception:
    # Si ya está configurado (desde streamlit_app.py), ignorar
    pass

# NOTA: El título ya se muestra en streamlit_app.py para evitar duplicación
# Solo mostramos el mensaje importante y separador
st.info("💡 **IMPORTANTE:** Usa los filtros en la barra lateral (←) para seleccionar años, regiones y comunas específicas. Los datos se actualizarán automáticamente.")

st.markdown("---")

# Cargar datos reales una vez
@st.cache_data
def load_conaf_data():
    """Carga datos reales de CONAF"""
    file_path = Path("data/processed/conaf_datos_reales_completo.csv")
    
    if file_path.exists():
        try:
            df = pd.read_csv(file_path)
            # Convertir anio a int
            df['anio'] = pd.to_numeric(df['anio'], errors='coerce')
            df = df[df['anio'].notna()]
            # Corregir años (algunos están como 84 en lugar de 1984)
            df.loc[df['anio'] < 1900, 'anio'] = df.loc[df['anio'] < 1900, 'anio'] + 1900
            df['anio'] = df['anio'].astype(int)
            # Limpiar comunas
            df['comuna'] = df['comuna'].astype(str).str.strip().str.title()
            # Limpiar regiones - proceso robusto
            # Convertir a string primero para manejar NaN
            df['region'] = df['region'].astype(str)
            # Reemplazar valores NaN de pandas (aparecen como 'nan' en string)
            df['region'] = df['region'].replace(['nan', 'NaN', 'NAN', 'None', 'NONE', ''], 'Sin Región')
            # Limpiar espacios
            df['region'] = df['region'].str.strip()
            # Convertir a mayúsculas para normalizar
            df['region'] = df['region'].str.upper()
            # Reemplazar cualquier variante de 'Sin Región' o valores inválidos
            df.loc[df['region'].isin(['NAN', 'SIN REGIÓN', 'SIN REGION', 'SIN REGIóN', '']), 'region'] = 'Sin Región'
            # Finalmente, usar fillna por si acaso
            df['region'] = df['region'].fillna('Sin Región')
            return df
        except Exception as e:
            st.error(f"Error al cargar datos: {e}")
            return None
    
    # Si no existe, mostrar mensaje útil y sugerir alternativas
    st.error("""
    ❌ **Dataset procesado no encontrado**
    
    El archivo `data/processed/conaf_datos_reales_completo.csv` no está disponible.
    """)
    
    with st.expander("🔧 Soluciones", expanded=True):
        st.markdown("""
        **Opción 1: Incluir dataset en el repositorio (Recomendado)**
        
        1. Edita `.gitignore` y agrega esta línea para permitir el dataset:
           ```
           !data/processed/conaf_datos_reales_completo.csv
           ```
        
        2. Agrega el archivo al repositorio:
           ```bash
           git add data/processed/conaf_datos_reales_completo.csv
           git commit -m "Add processed CONAF dataset"
           git push
           ```
        
        **Opción 2: Procesar datos automáticamente**
        
        El sistema puede intentar procesar datos automáticamente si los archivos raw
        de CONAF están disponibles en `data/raw/`.
        """)
        
        if st.button("🔄 Intentar Procesar Datos Automáticamente", type="secondary"):
            with st.spinner("Buscando archivos CONAF raw para procesar..."):
                try:
                    from src.data.conaf_smart_processor import SmartCONAFProcessor
                    processor = SmartCONAFProcessor()
                    df = processor.process_all_files()
                    if len(df) > 0:
                        st.success(f"✅ Datos procesados automáticamente: {len(df):,} registros")
                        # Guardar para uso futuro
                        output_path = Path("data/processed/conaf_datos_reales_completo.csv")
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        df.to_csv(output_path, index=False)
                        st.info("💾 Dataset guardado en `data/processed/conaf_datos_reales_completo.csv`")
                        st.rerun()
                    else:
                        st.error("❌ No se encontraron archivos CONAF para procesar")
                        st.info("💡 Por favor coloca los archivos Excel/XLS de CONAF en `data/raw/`")
                except FileNotFoundError as e:
                    st.error(f"❌ Archivos raw no encontrados: {e}")
                    st.info("💡 Necesitas los archivos Excel/XLS de CONAF en `data/raw/`")
                except Exception as e:
                    st.error(f"❌ Error al procesar: {e}")
                    import traceback
                    with st.expander("Detalles del error"):
                        st.code(traceback.format_exc())
    
    return None

# Inicializar datos
try:
    if 'conaf_data' not in st.session_state:
        with st.spinner("Cargando datos de CONAF..."):
            st.session_state.conaf_data = load_conaf_data()

    if st.session_state.conaf_data is None or len(st.session_state.conaf_data) == 0:
        # Mostrar mensaje de error claramente ANTES de detenerse
        st.error("❌ **No se encontraron datos de CONAF**")
        st.warning("""
        **El dataset procesado no está disponible.**
        
        Por favor verifica que el archivo `data/processed/conaf_datos_reales_completo.csv` 
        esté en el repositorio de GitHub.
        """)
        
        st.info("""
        **Para solucionar:**
        1. Verifica que el archivo existe en GitHub: `data/processed/conaf_datos_reales_completo.csv`
        2. Si no existe, agrégalo al repositorio:
           ```bash
           git add data/processed/conaf_datos_reales_completo.csv
           git commit -m "Add dataset"
           git push
           ```
        3. Espera 1-2 minutos para que Streamlit Cloud se actualice
        """)
        
        # Mostrar información de debug
        with st.expander("🔍 Información de Debug", expanded=True):
            data_path = Path("data/processed/conaf_datos_reales_completo.csv")
            st.write(f"**Ruta esperada:** `{data_path}`")
            st.write(f"**Ruta absoluta:** `{data_path.absolute()}`")
            st.write(f"**Existe:** `{data_path.exists()}`")
            
            if data_path.exists():
                st.success(f"✅ Archivo encontrado: {data_path.stat().st_size:,} bytes")
            else:
                st.error("❌ Archivo NO encontrado")
            
            # Listar archivos en data/processed
            processed_dir = Path("data/processed")
            st.write(f"\n**Directorio data/processed existe:** `{processed_dir.exists()}`")
            if processed_dir.exists():
                st.write("**Archivos en data/processed:**")
                files = list(processed_dir.iterdir())
                if files:
                    for f in files:
                        st.write(f"  - `{f.name}` ({f.stat().st_size:,} bytes)")
                else:
                    st.write("  (vacío)")
            else:
                st.write("  (directorio no existe)")
        
        # Mostrar que la app está funcionando pero sin datos
        st.sidebar.warning("⚠️ Dashboard sin datos - Ver información arriba")
        
        # NO usar st.stop() aquí - dejar que se muestre el error
        # En su lugar, crear un DataFrame vacío para evitar errores posteriores
        st.session_state.conaf_data = pd.DataFrame(columns=['comuna', 'region', 'anio', 'num_incendios', 'area_quemada_ha'])
        
except Exception as e:
    st.error(f"❌ **Error al inicializar datos**")
    st.exception(e)
    import traceback
    with st.expander("🔍 Detalles técnicos del error", expanded=True):
        st.code(traceback.format_exc())
    
    # Crear DataFrame vacío para evitar más errores
    st.session_state.conaf_data = pd.DataFrame(columns=['comuna', 'region', 'anio', 'num_incendios', 'area_quemada_ha'])
    st.sidebar.error(f"Error: {str(e)[:50]}...")

# Obtener datos base - verificar que existe
if 'conaf_data' not in st.session_state or st.session_state.conaf_data is None:
    # Si no hay datos, crear DataFrame vacío
    st.session_state.conaf_data = pd.DataFrame(columns=['comuna', 'region', 'anio', 'num_incendios', 'area_quemada_ha'])

df_base = st.session_state.conaf_data.copy()

# Si no hay datos, mostrar advertencia pero continuar
if len(df_base) == 0:
    st.warning("⚠️ **No hay datos disponibles.** Por favor verifica la información de debug arriba.")

# Sidebar - Filtros
st.sidebar.header("⚙️ Filtros y Configuración")

# Filtros en sidebar
st.sidebar.subheader("📅 Filtro de Años")
# Manejar caso cuando no hay datos o DataFrame está vacío
if len(df_base) > 0 and 'anio' in df_base.columns:
    anos_disponibles = sorted(df_base['anio'].dropna().unique())
    ano_min = int(anos_disponibles[0]) if len(anos_disponibles) > 0 else 1985
    ano_max = int(anos_disponibles[-1]) if len(anos_disponibles) > 0 else 2023
else:
    # Valores por defecto si no hay datos
    anos_disponibles = []
    ano_min = 1985
    ano_max = 2023

ano_inicio = st.sidebar.number_input(
    "Año Inicio",
    min_value=ano_min,
    max_value=ano_max,
    value=max(2015, ano_min),
    step=1,
    key="ano_inicio"
)

ano_fin = st.sidebar.number_input(
    "Año Fin",
    min_value=ano_min,
    max_value=ano_max,
    value=ano_max,
    step=1,
    key="ano_fin"
)

if ano_inicio > ano_fin:
    st.sidebar.warning("⚠️ El año inicio debe ser menor o igual al año fin")
    ano_fin = ano_inicio

# Filtro de regiones
st.sidebar.subheader("🗺️ Filtro de Regiones")
# Manejar caso cuando no hay datos
if len(df_base) > 0 and 'region' in df_base.columns:
    regiones_disponibles = sorted([r for r in df_base['region'].dropna().unique() if r != 'Sin Región' and pd.notna(r)])
    regiones_disponibles.insert(0, 'Todas las Regiones')
else:
    regiones_disponibles = ['Todas las Regiones']

region_seleccionada = st.sidebar.selectbox(
    "Seleccionar Región",
    regiones_disponibles,
    index=0,
    key="region_select"
)

# Filtro de comunas (depende de región)
st.sidebar.subheader("🏘️ Filtro de Comunas")
# Manejar caso cuando no hay datos
if len(df_base) > 0 and 'comuna' in df_base.columns:
    if region_seleccionada != 'Todas las Regiones':
        df_filtrado_region = df_base[df_base['region'] == region_seleccionada]
        comunas_disponibles = sorted([c for c in df_filtrado_region['comuna'].dropna().unique() if pd.notna(c)])
        comunas_disponibles.insert(0, 'Todas las Comunas de la Región')
    else:
        comunas_disponibles = sorted([c for c in df_base['comuna'].dropna().unique() if pd.notna(c)])
        comunas_disponibles.insert(0, 'Todas las Comunas')
else:
    comunas_disponibles = ['Todas las Comunas']

# Limitar comunas para no sobrecargar
if len(comunas_disponibles) > 200:
    comunas_disponibles = comunas_disponibles[:200]

comuna_seleccionada = st.sidebar.selectbox(
    "Seleccionar Comuna",
    comunas_disponibles,
    index=0,
    key="comuna_select"
)

# Aplicar filtros
try:
    df_filtrado = df_base[
        (df_base['anio'] >= ano_inicio) &
        (df_base['anio'] <= ano_fin)
    ].copy()
    
    if region_seleccionada != 'Todas las Regiones':
        df_filtrado = df_filtrado[df_filtrado['region'] == region_seleccionada]
    
    if comuna_seleccionada != 'Todas las Comunas' and comuna_seleccionada != 'Todas las Comunas de la Región':
        df_filtrado = df_filtrado[df_filtrado['comuna'] == comuna_seleccionada]
except Exception as e:
    st.sidebar.error(f"Error al aplicar filtros: {e}")
    df_filtrado = df_base.copy()

# Mostrar info de filtros
st.sidebar.markdown("---")
st.sidebar.info(f"""
**Datos Filtrados:**
- Registros: {len(df_filtrado):,}
- Años: {ano_inicio}-{ano_fin}
- Región: {region_seleccionada[:20]}
- Comuna: {(comuna_seleccionada[:20] + '...' if len(comuna_seleccionada) > 20 else comuna_seleccionada)}
""")

# Guardar datos filtrados en sesión
st.session_state.datos_filtrados = df_filtrado

# Inicializar sesión para otras variables
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
if 'risk_map' not in st.session_state:
    st.session_state.risk_map = None
if 'optimizer' not in st.session_state:
    st.session_state.optimizer = ResourceAllocationOptimizer()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Datos y Análisis",
    "🤖 Predicción de Riesgo",
    "🎯 Optimización de Recursos",
    "📈 Reportes y Estadísticas"
])

# ===== TAB 1: Datos y Análisis =====
with tab1:
    st.header("📊 Visualización de Datos CONAF")
    
    if len(df_filtrado) == 0:
        st.warning("⚠️ No hay datos para los filtros seleccionados. Por favor ajusta los filtros en la barra lateral.")
    else:
        try:
            # Métricas principales
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_incendios = df_filtrado['num_incendios'].sum()
                st.metric("Total Incendios", f"{total_incendios:,.0f}")
            
            with col2:
                total_area = df_filtrado['area_quemada_ha'].sum()
                st.metric("Área Quemada Total", f"{total_area:,.2f} ha")
            
            with col3:
                comunas_unicas = df_filtrado['comuna'].nunique()
                st.metric("Comunas Afectadas", comunas_unicas)
            
            with col4:
                anos_unicos = df_filtrado['anio'].nunique()
                st.metric("Años Analizados", anos_unicos)
            
            st.markdown("---")
            
            # Gráficos
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Incendios por Año")
                try:
                    incendios_anuales = df_filtrado.groupby('anio')['num_incendios'].sum().reset_index()
                    fig1 = px.bar(
                        incendios_anuales, 
                        x='anio', 
                        y='num_incendios',
                        title=f'Incendios por Año ({ano_inicio}-{ano_fin})',
                        labels={'num_incendios': 'Número de Incendios', 'anio': 'Año'},
                        color='num_incendios',
                        color_continuous_scale='Reds'
                    )
                    st.plotly_chart(fig1, width='stretch')
                except Exception as e:
                    st.error(f"Error al generar gráfico: {e}")
            
            with col2:
                st.subheader("Área Quemada por Año")
                try:
                    area_anual = df_filtrado.groupby('anio')['area_quemada_ha'].sum().reset_index()
                    fig2 = px.line(
                        area_anual,
                        x='anio',
                        y='area_quemada_ha',
                        title=f'Área Quemada por Año ({ano_inicio}-{ano_fin})',
                        labels={'area_quemada_ha': 'Área (ha)', 'anio': 'Año'},
                        markers=True
                    )
                    st.plotly_chart(fig2, width='stretch')
                except Exception as e:
                    st.error(f"Error al generar gráfico: {e}")
            
            # Top comunas con más incendios
            st.subheader("🏘️ Top 15 Comunas con Más Incendios (Período Seleccionado)")
            try:
                top_comunas = (
                    df_filtrado.groupby('comuna')['num_incendios']
                    .sum()
                    .sort_values(ascending=False)
                    .head(15)
                    .reset_index()
                )
                top_comunas.columns = ['Comuna', 'Total Incendios']
                st.dataframe(top_comunas, width='stretch', height=400)
                
                # Gráfico de top comunas
                fig_top = px.bar(
                    top_comunas,
                    x='Total Incendios',
                    y='Comuna',
                    orientation='h',
                    title='Top 15 Comunas con Más Incendios',
                    labels={'Total Incendios': 'Número de Incendios'},
                    color='Total Incendios',
                    color_continuous_scale='Oranges'
                )
                fig_top.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_top, width='stretch')
            except Exception as e:
                st.error(f"Error al generar top comunas: {e}")
            
            # Tabla de datos
            with st.expander("📋 Ver Datos Detallados"):
                try:
                    st.dataframe(
                        df_filtrado[['comuna', 'region', 'anio', 'num_incendios', 'area_quemada_ha', 'temporada']].sort_values('num_incendios', ascending=False),
                        width='stretch',
                        height=400
                    )
                except Exception as e:
                    st.error(f"Error al mostrar datos: {e}")
        
        except Exception as e:
            st.error(f"Error en visualización: {e}")
            import traceback
            st.code(traceback.format_exc())

# ===== TAB 2: Predicción de Riesgo =====
with tab2:
    st.header("🤖 Modelo de Predicción de Riesgo")
    
    if len(df_filtrado) == 0:
        st.warning("⚠️ Por favor selecciona datos válidos usando los filtros en la barra lateral.")
    else:
        st.info(f"💡 Analizando {len(df_filtrado):,} registros de {df_filtrado['comuna'].nunique()} comunas entre {ano_inicio} y {ano_fin}")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Configuración del Modelo")
            
            # Explicación de tipos de modelo
            with st.expander("ℹ️ ¿Qué tipo de modelo elegir?", expanded=False):
                st.markdown("""
                **XGBoost** (Recomendado):
                - ✅ Mejor rendimiento general para datos tabulares
                - ✅ Maneja bien relaciones no lineales
                - ✅ Buen balance entre velocidad y precisión
                - ✅ Ideal para: Predicción de riesgo de incendios
                
                **LightGBM**:
                - ✅ Muy rápido en entrenamiento
                - ✅ Eficiente con datasets grandes
                - ✅ Buen rendimiento, similar a XGBoost
                - ✅ Ideal para: Análisis rápidos o datasets muy grandes
                
                **Random Forest**:
                - ✅ Más interpretable
                - ✅ Menos propenso a overfitting
                - ✅ Más lento que XGBoost/LightGBM
                - ✅ Ideal para: Cuando necesitas entender mejor las decisiones del modelo
                """)
            
            model_type = st.selectbox(
                "Tipo de Modelo",
                ["xgboost", "lightgbm", "random_forest"],
                index=0,
                help="Selecciona el algoritmo de Machine Learning a usar"
            )
            
            # Explicación de tipos de tarea
            with st.expander("ℹ️ ¿Qué tipo de tarea elegir?", expanded=False):
                st.markdown("""
                **Classification (Clasificación)** - Recomendado para la mayoría de casos:
                - 🎯 **Objetivo**: Predecir si HABRÁ o NO HABRÁ incendio (Sí/No)
                - 📊 **Output**: Probabilidad de riesgo (0% a 100%)
                - ✅ **Ideal para**: 
                    - Alertas tempranas ("¿Hay riesgo de incendio hoy?")
                    - Asignación preventiva de recursos
                    - Identificar zonas de alto riesgo
                - 📈 **Métricas**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
                
                **Regression (Regresión)**:
                - 🎯 **Objetivo**: Predecir CUÁNTOS incendios habrá (número exacto)
                - 📊 **Output**: Número estimado de incendios
                - ✅ **Ideal para**:
                    - Planificación de recursos (¿cuántas brigadas necesito?)
                    - Estimación de daño esperado
                    - Presupuestos y logística
                - 📈 **Métricas**: RMSE, MAE, R²
                
                **💡 Recomendación**: Usa **Classification** para la mayoría de casos de uso operacional.
                """)
            
            task_type = st.selectbox(
                "Tipo de Tarea",
                ["classification", "regression"],
                index=0,
                help="Classification: ¿Habrá incendio? | Regression: ¿Cuántos incendios?"
            )
        
        with col2:
            st.subheader("Entrenamiento")
            
            # Información importante sobre cómo funcionan los modelos
            st.info("""
            **📝 ¿Cómo funcionan los modelos?**
            
            - **Se entrenan sobre la marcha**: Al hacer clic en "Entrenar Modelo", se entrena un nuevo modelo con los datos filtrados.
            - **Se guardan en la sesión**: Una vez entrenado, el modelo permanece disponible durante tu sesión de navegador.
            - **No hay modelos pre-entrenados**: Cada usuario debe entrenar su propio modelo, lo que permite ajustarlo a datos específicos.
            - **Se pierden al cerrar**: Si cierras el navegador, necesitarás entrenar el modelo nuevamente.
            """)
            
            # Información sobre el entrenamiento
            st.info(f"""
            **Datos para entrenar:**
            - {len(df_filtrado):,} registros
            - {df_filtrado['comuna'].nunique()} comunas
            - Período: {ano_inicio}-{ano_fin}
            
            El modelo aprenderá patrones históricos de estos datos.
            """)
            
            if st.button("🚀 Entrenar Modelo", type="primary", use_container_width=True):
                with st.spinner("Entrenando modelo con datos reales..."):
                    try:
                        # Preparar datos para ML
                        # IMPORTANTE: Crear panel completo incluyendo comunas sin incendios
                        # para tener ambas clases (0 = sin incendio, 1 = con incendio)
                        
                        # Obtener todas las combinaciones posibles de comuna-año
                        todas_comunas = df_filtrado['comuna'].unique()
                        todos_anios = sorted(df_filtrado['anio'].unique())
                        
                        # Crear panel completo con todas las combinaciones
                        from itertools import product
                        panel_completo = pd.DataFrame(
                            list(product(todas_comunas, todos_anios)),
                            columns=['comuna', 'anio']
                        )
                        
                        # Agregar datos reales
                        panel_agregado = df_filtrado.groupby(['comuna', 'anio']).agg({
                            'num_incendios': 'sum',
                            'area_quemada_ha': 'sum'
                        }).reset_index()
                        
                        # Merge: los que no tienen datos tendrán NaN en num_incendios
                        panel_df = panel_completo.merge(
                            panel_agregado,
                            on=['comuna', 'anio'],
                            how='left'
                        )
                        
                        # Llenar NaN con 0 (comunas sin incendios en ese año)
                        panel_df['num_incendios'] = panel_df['num_incendios'].fillna(0)
                        panel_df['area_quemada_ha'] = panel_df['area_quemada_ha'].fillna(0)
                        
                        # Preparar datos para ML
                        # Agregar features básicas temporales primero
                        panel_df['mes'] = 1  # Feature temporal básica
                        panel_df['dia_anio'] = panel_df['anio'] * 365  # Día del año aproximado
                        
                        # Agregar features históricas básicas por comuna
                        historico_comuna = df_filtrado.groupby('comuna').agg({
                            'num_incendios': ['sum', 'mean', 'max'],
                            'area_quemada_ha': 'sum'
                        }).reset_index()
                        historico_comuna.columns = ['comuna', 'incendios_total_hist', 'incendios_promedio_hist', 'incendios_max_hist', 'area_total_hist']
                        panel_df = panel_df.merge(historico_comuna, on='comuna', how='left')
                        
                        # Crear variable objetivo - usar el nombre que espera prepare_features
                        if task_type == 'classification':
                            # Para clasificación: 1 = hubo incendio, 0 = no hubo
                            panel_df['incendio_ocurrencia'] = (panel_df['num_incendios'] > 0).astype(int)
                            target_col = 'incendio_ocurrencia'
                        else:
                            # Para regresión: queremos predecir el número de incendios
                            # Pero prepare_features espera 'incendio_ocurrencia', así que usamos num_incendios como target
                            panel_df['incendio_ocurrencia'] = panel_df['num_incendios'].copy()
                            target_col = 'incendio_ocurrencia'
                        
                        # Crear predictor
                        predictor = FireRiskPredictor(model_type=model_type, task=task_type)
                        
                        # Preparar features - pasar el nombre de la columna objetivo
                        X, y = predictor.prepare_features(panel_df, target_col=target_col)
                        
                        # Entrenar
                        metrics = predictor.train(X, y, validation_size=0.2, temporal_split=True)
                        
                        # Guardar en sesión
                        st.session_state.predictor = predictor
                        st.session_state.panel_data = panel_df
                        st.session_state.task_type = task_type  # Guardar tipo de tarea para predicción
                        
                        st.success("✅ Modelo entrenado exitosamente con datos reales")
                        
                        # Mostrar métricas
                        st.subheader("📊 Métricas del Modelo")
                        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                        
                        with col_m1:
                            accuracy_val = float(metrics.get('accuracy', 0))
                            st.markdown(f"**Accuracy:**")
                            st.markdown(f"### {accuracy_val:.3f}")
                            st.caption("Porcentaje de predicciones correctas")
                        
                        with col_m2:
                            f1_val = float(metrics.get('f1', 0))
                            st.markdown(f"**F1-Score:**")
                            st.markdown(f"### {f1_val:.3f}")
                            st.caption("Balance entre precisión y recall")
                        
                        with col_m3:
                            precision_val = float(metrics.get('precision', 0))
                            st.markdown(f"**Precision:**")
                            st.markdown(f"### {precision_val:.3f}")
                            st.caption("Verdaderos positivos / (Verdaderos + Falsos positivos)")
                        
                        with col_m4:
                            recall_val = float(metrics.get('recall', 0))
                            st.markdown(f"**Recall:**")
                            st.markdown(f"### {recall_val:.3f}")
                            st.caption("Verdaderos positivos / (Verdaderos positivos + Falsos negativos)")
                        
                        # ROC-AUC en una fila separada si existe
                        if metrics.get('roc_auc') is not None:
                            roc_auc_val = float(metrics.get('roc_auc', 0))
                            col_roc1, col_roc2 = st.columns([1, 3])
                            with col_roc1:
                                st.markdown(f"**ROC-AUC:**")
                                st.markdown(f"### {roc_auc_val:.3f}")
                                st.caption("Área bajo la curva ROC (0.5 = aleatorio, 1.0 = perfecto)")
                        
                        # Feature importance
                        if predictor.feature_importance is not None and len(predictor.feature_importance) > 0:
                            st.subheader("🔍 Importancia de Features (Top 10)")
                            top_features = predictor.feature_importance.head(10)
                            fig_importance = px.bar(
                                top_features,
                                x='importance',
                                y='feature',
                                orientation='h',
                                title='Top 10 Features Más Importantes',
                                color='importance',
                                color_continuous_scale='Blues'
                            )
                            fig_importance.update_layout(yaxis={'categoryorder': 'total ascending'})
                            st.plotly_chart(fig_importance, width='stretch')
                    
                    except Exception as e:
                        st.error(f"Error al entrenar modelo: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
        
        # Predicción de riesgo
        if st.session_state.predictor is not None:
            st.success("✅ Modelo entrenado y listo para hacer predicciones")
            st.markdown("---")
            st.subheader("🗺️ Mapa de Riesgo")
            
            if st.button("🔮 Generar Mapa de Riesgo", type="primary"):
                with st.spinner("Generando mapa de riesgo..."):
                    try:
                        ultimo_anio = df_filtrado['anio'].max()
                        comunas_unicas = df_filtrado['comuna'].unique()
                        
                        pred_df = pd.DataFrame({
                            'comuna': comunas_unicas,
                            'anio': ultimo_anio + 1,
                            'mes': 1,
                            'dia_anio': (ultimo_anio + 1) * 365
                        })
                        
                        historico_comuna = df_filtrado.groupby('comuna').agg({
                            'num_incendios': ['sum', 'mean', 'max'],
                            'area_quemada_ha': 'sum'
                        }).reset_index()
                        
                        historico_comuna.columns = ['comuna', 'incendios_total', 'incendios_promedio', 'incendios_max', 'area_total']
                        pred_df = pred_df.merge(historico_comuna, on='comuna', how='left')
                        
                        # Agregar columna dummy 'incendio_ocurrencia' que prepare_features espera (no se usará para predicción)
                        # Usamos 0 como valor dummy ya que es solo para satisfacer el formato esperado
                        pred_df['incendio_ocurrencia'] = 0
                        
                        # Obtener task_type de session_state si existe
                        task_type_pred = st.session_state.get('task_type', 'classification')
                        
                        # Preparar features para predicción - pasar target_col aunque no se use
                        X_pred, _ = st.session_state.predictor.prepare_features(pred_df, target_col='incendio_ocurrencia')
                        
                        if task_type_pred == 'classification':
                            riesgos = st.session_state.predictor.predict(X_pred, return_proba=True)
                        else:
                            predicciones = st.session_state.predictor.predict(X_pred)
                            riesgos = (predicciones - predicciones.min()) / (predicciones.max() - predicciones.min() + 1e-10)
                        
                        risk_map = pd.DataFrame({
                            'comuna': comunas_unicas,
                            'riesgo_probabilidad': riesgos,
                            'incendios_historico': historico_comuna['incendios_total'].values,
                            'area_historica': historico_comuna['area_total'].values
                        })
                        
                        st.session_state.risk_map = risk_map
                        st.success("✅ Mapa de riesgo generado")
                        
                    except Exception as e:
                        st.error(f"Error al generar mapa de riesgo: {str(e)}")
            
            # Mostrar mapa de riesgo
            if st.session_state.risk_map is not None:
                risk_map = st.session_state.risk_map
                
                st.subheader("📋 Riesgo por Comuna")
                try:
                    risk_map_sorted = risk_map.sort_values('riesgo_probabilidad', ascending=False)
                    risk_map_sorted['riesgo_categoria'] = pd.cut(
                        risk_map_sorted['riesgo_probabilidad'],
                        bins=[0, 0.3, 0.6, 1.0],
                        labels=['Bajo', 'Medio', 'Alto']
                    )
                    
                    st.dataframe(
                        risk_map_sorted[['comuna', 'riesgo_probabilidad', 'riesgo_categoria', 'incendios_historico', 'area_historica']].head(20),
                        width='stretch'
                    )
                    
                    fig_risk = px.bar(
                        risk_map_sorted.head(20),
                        x='riesgo_probabilidad',
                        y='comuna',
                        orientation='h',
                        title='Top 20 Comunas con Mayor Riesgo',
                        labels={'riesgo_probabilidad': 'Probabilidad de Riesgo'},
                        color='riesgo_probabilidad',
                        color_continuous_scale='Reds'
                    )
                    fig_risk.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig_risk, width='stretch')
                except Exception as e:
                    st.error(f"Error al mostrar mapa de riesgo: {e}")
        else:
            st.warning("""
            ⚠️ **No hay modelo entrenado**
            
            Para hacer predicciones:
            1. Ve a la sección "Entrenamiento" arriba
            2. Selecciona el tipo de modelo (XGBoost, LightGBM o Random Forest) y tarea (Clasificación o Regresión)
            3. Haz clic en "🚀 Entrenar Modelo"
            4. Una vez entrenado, podrás generar mapas de riesgo aquí
            
            **💡 Nota**: El modelo se entrena con los datos que filtres en la barra lateral (años, región, comuna). 
            El modelo se guarda en tu sesión de navegador y se pierde al cerrar la pestaña.
            """)

# ===== TAB 3: Optimización de Recursos =====
with tab3:
    st.header("🎯 Optimización de Asignación de Recursos")
    
    if st.session_state.risk_map is None:
        st.warning("⚠️ Por favor genera un mapa de riesgo primero en la pestaña 'Predicción de Riesgo'")
    else:
        st.info(f"🎯 Optimizando recursos para {region_seleccionada} ({comuna_seleccionada[:30]})")
        
        st.subheader("Configuración de Optimización")
        
        # Explicación sobre optimización
        with st.expander("ℹ️ ¿Qué hace la optimización de recursos?", expanded=False):
            st.markdown("""
            **🎯 Objetivo de la Optimización:**
            
            Dado un número limitado de brigadas y bases de operaciones, el sistema
            calcula la **mejor ubicación** para minimizar el daño esperado o el tiempo
            de respuesta.
            
            **📊 Cómo funciona:**
            1. Usa el mapa de riesgo generado previamente
            2. Considera la distancia entre bases y zonas de riesgo
            3. Optimiza matemáticamente la asignación
            4. Genera recomendaciones de ubicación óptima
            
            **💡 Casos de uso:**
            - Planificación estratégica antes de la temporada de incendios
            - Reubicación de recursos durante emergencias
            - Evaluación de nuevas ubicaciones de bases
            - Optimización de presupuesto y recursos
            """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            max_brigades = st.number_input(
                "Máximo de Brigadas Disponibles",
                min_value=1,
                max_value=500,
                value=50,
                step=5,
                help="Número total de brigadas que puedes desplegar"
            )
        
        with col2:
            max_bases = st.number_input(
                "Máximo de Bases Posibles",
                min_value=1,
                max_value=50,
                value=10,
                step=1,
                help="Número máximo de bases de operaciones a activar"
            )
        
        with col3:
            # Explicación de objetivos
            with st.expander("ℹ️ ¿Qué objetivo elegir?"):
                st.markdown("""
                **Minimize Damage (Minimizar Daño)** - Recomendado:
                - 🎯 Minimiza el área quemada esperada
                - ✅ Prioriza zonas de alto riesgo
                - ✅ Ideal para: Planificación preventiva
                - 📊 Considera: Probabilidad × Severidad esperada
                
                **Minimize Response Time (Minimizar Tiempo de Respuesta)**:
                - 🎯 Minimiza el tiempo promedio de llegada
                - ✅ Prioriza cobertura geográfica
                - ✅ Ideal para: Respuesta rápida a emergencias
                - 📊 Considera: Distancia × Riesgo
                """)
            
            objective = st.selectbox(
                "Objetivo de Optimización",
                ["minimize_damage", "minimize_response_time"],
                index=0,
                help="Elige qué minimizar: daño esperado o tiempo de respuesta"
            )
        
        if st.button("⚙️ Optimizar Asignación", type="primary", use_container_width=True):
            with st.spinner("Optimizando asignación de recursos..."):
                try:
                    optimizer = ResourceAllocationOptimizer(
                        max_brigades=max_brigades,
                        max_bases=max_bases
                    )
                    
                    optimizer.prepare_data(st.session_state.risk_map)
                    solution = optimizer.optimize(objective=objective)
                    
                    st.session_state.optimizer = optimizer
                    st.session_state.optimizer.solution = solution
                    
                    st.success("✅ Optimización completada")
                    
                    st.subheader("📊 Resultados de la Optimización")
                    
                    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
                    
                    with col_r1:
                        st.metric("Bases Activas", solution['total_bases_activas'])
                    
                    with col_r2:
                        st.metric("Total Brigadas", solution['total_brigades'])
                    
                    with col_r3:
                        st.metric("Tiempo Respuesta Promedio", f"{solution['tiempo_respuesta_promedio']:.1f} min")
                    
                    with col_r4:
                        st.metric("Tiempo Respuesta Ponderado", f"{solution['tiempo_respuesta_ponderado']:.1f} min")
                    
                    st.subheader("🏠 Distribución de Brigadas por Base")
                    try:
                        brigadas_df = pd.DataFrame(
                            list(solution['brigadas_por_base'].items()),
                            columns=['Base', 'Brigadas']
                        ).sort_values('Brigadas', ascending=False)
                        
                        st.dataframe(brigadas_df, width='stretch')
                        
                        fig_brigadas = px.bar(
                            brigadas_df,
                            x='Base',
                            y='Brigadas',
                            title='Brigadas por Base',
                            color='Brigadas',
                            color_continuous_scale='Greens'
                        )
                        st.plotly_chart(fig_brigadas, width='stretch')
                    except Exception as e:
                        st.error(f"Error al mostrar distribución: {e}")
                    
                    st.subheader("🗺️ Mapa de Asignación")
                    try:
                        allocation_map = optimizer.get_allocation_map()
                        st.dataframe(allocation_map.head(30), width='stretch', height=400)
                    except Exception as e:
                        st.error(f"Error al generar mapa: {e}")
                        
                except Exception as e:
                    st.error(f"Error en optimización: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

# ===== TAB 4: Reportes y Estadísticas =====
with tab4:
    st.header("📈 Reportes y Estadísticas Avanzadas")
    
    if len(df_filtrado) == 0:
        st.warning("⚠️ Por favor selecciona datos válidos usando los filtros")
    else:
        try:
            st.subheader("📊 Análisis Temporal")
            
            col1, col2 = st.columns(2)
            
            with col1:
                incendios_anuales = df_filtrado.groupby('anio')['num_incendios'].sum().reset_index()
                z = np.polyfit(incendios_anuales['anio'], incendios_anuales['num_incendios'], 1)
                p = np.poly1d(z)
                incendios_anuales['tendencia'] = p(incendios_anuales['anio'])
                
                fig_tendencia = go.Figure()
                fig_tendencia.add_trace(go.Scatter(
                    x=incendios_anuales['anio'],
                    y=incendios_anuales['num_incendios'],
                    mode='lines+markers',
                    name='Incendios',
                    line=dict(color='red', width=2)
                ))
                fig_tendencia.add_trace(go.Scatter(
                    x=incendios_anuales['anio'],
                    y=incendios_anuales['tendencia'],
                    mode='lines',
                    name='Tendencia',
                    line=dict(color='blue', dash='dash', width=2)
                ))
                fig_tendencia.update_layout(
                    title='Tendencia de Incendios por Año',
                    xaxis_title='Año',
                    yaxis_title='Número de Incendios'
                )
                st.plotly_chart(fig_tendencia, width='stretch')
            
            with col2:
                if region_seleccionada == 'Todas las Regiones':
                    # Filtrar 'Sin Región' y valores inválidos antes de agrupar
                    # Normalizar a mayúsculas para comparación
                    df_region_clean = df_filtrado[
                        (df_filtrado['region'].notna()) & 
                        (df_filtrado['region'].astype(str).str.upper() != 'SIN REGIÓN') &
                        (df_filtrado['region'].astype(str).str.upper() != 'SIN REGION') &
                        (df_filtrado['region'].astype(str).str.upper() != 'SIN REGIóN') &
                        (df_filtrado['region'] != 'Sin Región') &
                        (df_filtrado['region'].astype(str) != 'nan') &
                        (df_filtrado['region'].astype(str) != 'NAN')
                    ].copy()
                    
                    if len(df_region_clean) > 0:
                        incendios_region = df_region_clean.groupby('region')['num_incendios'].sum().reset_index()
                        incendios_region = incendios_region.sort_values('num_incendios', ascending=False).head(10)
                        
                        if len(incendios_region) > 0:
                            fig_region = px.bar(
                                incendios_region,
                                x='num_incendios',
                                y='region',
                                orientation='h',
                                title='Top 10 Regiones con Más Incendios',
                                labels={'num_incendios': 'Número de Incendios', 'region': 'Región'},
                                color='num_incendios',
                                color_continuous_scale='Reds'
                            )
                            fig_region.update_layout(yaxis={'categoryorder': 'total ascending'})
                            st.plotly_chart(fig_region, width='stretch')
                        else:
                            st.info("No hay datos de regiones válidas para mostrar")
                    else:
                        st.warning("⚠️ Todos los registros tienen región 'Sin Región' o NaN. Verifica los datos.")
                else:
                    comunas_region = df_filtrado.groupby('comuna')['num_incendios'].sum().reset_index()
                    comunas_region = comunas_region.sort_values('num_incendios', ascending=False).head(10)
                    
                    fig_comuna = px.bar(
                        comunas_region,
                        x='num_incendios',
                        y='comuna',
                        orientation='h',
                        title=f'Top 10 Comunas con Más Incendios ({region_seleccionada})',
                        labels={'num_incendios': 'Número de Incendios'},
                        color='num_incendios',
                        color_continuous_scale='Oranges'
                    )
                    fig_comuna.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig_comuna, width='stretch')
            
            st.subheader("📋 Estadísticas Descriptivas")
            try:
                stats_df = df_filtrado.groupby('comuna').agg({
                    'num_incendios': ['sum', 'mean', 'std', 'min', 'max'],
                    'area_quemada_ha': ['sum', 'mean']
                }).reset_index()
                
                stats_df.columns = ['Comuna', 'Total Incendios', 'Promedio', 'Desv. Est.', 'Mínimo', 'Máximo', 
                               'Área Total (ha)', 'Área Promedio (ha)']
                
                st.dataframe(
                    stats_df.sort_values('Total Incendios', ascending=False).head(20),
                    width='stretch',
                    height=400
                )
            except Exception as e:
                st.error(f"Error al generar estadísticas: {e}")
            
            st.subheader("💾 Exportar Datos")
            try:
                csv = df_filtrado.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="📥 Descargar Datos Filtrados (CSV)",
                    data=csv,
                    file_name=f"incendios_conaf_{ano_inicio}_{ano_fin}_{region_seleccionada[:10]}.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Error al exportar: {e}")
        
        except Exception as e:
            st.error(f"Error en reportes: {e}")
            import traceback
            st.code(traceback.format_exc())

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    <p><strong>Sistema de Predicción y Optimización de Recursos para Incendios Forestales - Chile</strong></p>
    <p>Datos oficiales de CONAF (1985-2024) | Desarrollado con Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)
