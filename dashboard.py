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
import re
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

# Mapeo completo de comunas a regiones (basado en división administrativa oficial de Chile)
MAPEO_COMUNA_REGION = {
    # XV de Arica y Parinacota
    'Arica': 'XV', 'Camarones': 'XV', 'Putre': 'XV', 'General Lagos': 'XV',
    # I de Tarapacá
    'Alto Hospicio': 'I', 'Iquique': 'I', 'Huara': 'I', 'Camiña': 'I', 'Colchane': 'I', 
    'Pica': 'I', 'Pozo Almonte': 'I',
    # II de Antofagasta
    'Tocopilla': 'II', 'María Elena': 'II', 'Calama': 'II', 'Ollagüe': 'II', 
    'San Pedro de Atacama': 'II', 'Antofagasta': 'II', 'Mejillones': 'II', 
    'Sierra Gorda': 'II', 'Taltal': 'II',
    # III de Atacama
    'Chañaral': 'III', 'Diego de Almagro': 'III', 'Copiapó': 'III', 'Caldera': 'III', 
    'Tierra Amarilla': 'III', 'Vallenar': 'III', 'Freirina': 'III', 'Huasco': 'III', 
    'Alto del Carmen': 'III',
    # IV de Coquimbo
    'La Serena': 'IV', 'La Higuera': 'IV', 'Coquimbo': 'IV', 'Andacollo': 'IV', 
    'Vicuña': 'IV', 'Paihuano': 'IV', 'Ovalle': 'IV', 'Río Hurtado': 'IV', 
    'Monte Patria': 'IV', 'Combarbalá': 'IV', 'Punitaqui': 'IV', 'Illapel': 'IV', 
    'Salamanca': 'IV', 'Los Vilos': 'IV', 'Canela': 'IV',
    # V de Valparaíso
    'La Ligua': 'V', 'Petorca': 'V', 'Cabildo': 'V', 'Zapallar': 'V', 'Papudo': 'V', 
    'Los Andes': 'V', 'San Esteban': 'V', 'Calle Larga': 'V', 'Rinconada': 'V', 
    'San Felipe': 'V', 'Putaendo': 'V', 'Santa María': 'V', 'Panquehue': 'V', 
    'Llaillay': 'V', 'Catemu': 'V', 'Quillota': 'V', 'La Cruz': 'V', 'Calera': 'V', 
    'Nogales': 'V', 'Hijuelas': 'V', 'Limache': 'V', 'Olmué': 'V', 'Valparaíso': 'V', 
    'Viña del Mar': 'V', 'Quintero': 'V', 'Puchuncaví': 'V', 'Quilpué': 'V', 
    'Villa Alemana': 'V', 'Casablanca': 'V', 'Concón': 'V', 'Juan Fernández': 'V', 
    'San Antonio': 'V', 'Cartagena': 'V', 'El Tabo': 'V', 'El Quisco': 'V', 
    'Algarrobo': 'V', 'Santo Domingo': 'V', 'Isla de Pascua': 'V',
    # VI del Libertador General Bernardo O'Higgins
    'Rancagua': 'VI', 'Graneros': 'VI', 'Mostazal': 'VI', 'Codegua': 'VI', 
    'Machalí': 'VI', 'Olivar': 'VI', 'Requinoa': 'VI', 'Rengo': 'VI', 'Malloa': 'VI', 
    'Quinta de Tilcoco': 'VI', 'San Vicente': 'VI', 'Pichidegua': 'VI', 'Peumo': 'VI', 
    'Coltauco': 'VI', 'Coinco': 'VI', 'Doñihue': 'VI', 'Las Cabras': 'VI', 
    'San Fernando': 'VI', 'Chimbarongo': 'VI', 'Placilla': 'VI', 'Nancagua': 'VI', 
    'Chépica': 'VI', 'Santa Cruz': 'VI', 'Lolol': 'VI', 'Pumanque': 'VI', 
    'Palmilla': 'VI', 'Peralillo': 'VI', 'Pichilemu': 'VI', 'Navidad': 'VI', 
    'Litueche': 'VI', 'La Estrella': 'VI', 'Marchihue': 'VI', 'Paredones': 'VI',
    # VII del Maule
    'Curicó': 'VII', 'Teno': 'VII', 'Romeral': 'VII', 'Molina': 'VII', 
    'Sagrada Familia': 'VII', 'Hualañé': 'VII', 'Licantén': 'VII', 'Vichuquén': 'VII', 
    'Rauco': 'VII', 'Talca': 'VII', 'Pelarco': 'VII', 'Río Claro': 'VII', 
    'San Clemente': 'VII', 'Maule': 'VII', 'San Rafael': 'VII', 'Empedrado': 'VII', 
    'Pencahue': 'VII', 'Constitución': 'VII', 'Curepto': 'VII', 'Linares': 'VII', 
    'Yerbas Buenas': 'VII', 'Colbún': 'VII', 'Longaví': 'VII', 'Parral': 'VII', 
    'Retiro': 'VII', 'Villa Alegre': 'VII', 'San Javier': 'VII', 'Cauquenes': 'VII', 
    'Pelluhue': 'VII', 'Chanco': 'VII',
    # VIII del Biobío
    'Chillán': 'VIII', 'San Carlos': 'VIII', 'Ñiquén': 'VIII', 'San Fabián': 'VIII', 
    'Coihueco': 'VIII', 'Pinto': 'VIII', 'San Ignacio': 'VIII', 'El Carmen': 'VIII', 
    'Yungay': 'VIII', 'Pemuco': 'VIII', 'Bulnes': 'VIII', 'Quillón': 'VIII', 
    'Ránquil': 'VIII', 'Portezuelo': 'VIII', 'Coelemu': 'VIII', 'Treguaco': 'VIII', 
    'Cobquecura': 'VIII', 'Quirihue': 'VIII', 'Ninhue': 'VIII', 'San Nicolás': 'VIII', 
    'Chillán Viejo': 'VIII', 'Alto Biobío': 'VIII', 'Los Angeles': 'VIII', 
    'Los Ángeles': 'VIII', 'Cabrero': 'VIII', 'Tucapel': 'VIII', 'Antuco': 'VIII', 
    'Quilleco': 'VIII', 'Santa Bárbara': 'VIII', 'Quilaco': 'VIII', 'Mulchén': 'VIII', 
    'Negrete': 'VIII', 'Nacimiento': 'VIII', 'Laja': 'VIII', 'San Rosendo': 'VIII', 
    'Yumbel': 'VIII', 'Concepción': 'VIII', 'Talcahuano': 'VIII', 'Penco': 'VIII', 
    'Tomé': 'VIII', 'Florida': 'VIII', 'Hualpén': 'VIII', 'Hualqui': 'VIII', 
    'Santa Juana': 'VIII', 'Lota': 'VIII', 'Coronel': 'VIII', 'San Pedro de la Paz': 'VIII', 
    'Chiguayante': 'VIII', 'Lebu': 'VIII', 'Arauco': 'VIII', 'Curanilahue': 'VIII', 
    'Los Alamos': 'VIII', 'Los Álamos': 'VIII', 'Cañete': 'VIII', 'Contulmo': 'VIII', 
    'Tirua': 'VIII', 'Tirúa': 'VIII',
    # IX de la Araucanía
    'Angol': 'IX', 'Renaico': 'IX', 'Collipulli': 'IX', 'Lonquimay': 'IX', 
    'Curacautín': 'IX', 'Ercilla': 'IX', 'Victoria': 'IX', 'Traiguén': 'IX', 
    'Lumaco': 'IX', 'Purén': 'IX', 'Los Sauces': 'IX', 'Temuco': 'IX', 
    'Lautaro': 'IX', 'Perquenco': 'IX', 'Vilcún': 'IX', 'Cholchol': 'IX', 
    'Cunco': 'IX', 'Melipeuco': 'IX', 'Curarrehue': 'IX', 'Pucón': 'IX', 
    'Villarrica': 'IX', 'Freire': 'IX', 'Pitrufquén': 'IX', 'Gorbea': 'IX', 
    'Loncoche': 'IX', 'Toltén': 'IX', 'Teodoro Schmidt': 'IX', 'Saavedra': 'IX', 
    'Carahue': 'IX', 'Nueva Imperial': 'IX', 'Galvarino': 'IX', 'Padre las Casas': 'IX',
    # XIV de los Ríos
    'Valdivia': 'XIV', 'Mariquina': 'XIV', 'Lanco': 'XIV', 'Máfil': 'XIV', 
    'Corral': 'XIV', 'Panguipulli': 'XIV', 'Paillaco': 'XIV', 'La Unión': 'XIV', 
    'Futrono': 'XIV', 'Río Bueno': 'XIV', 'Lago Ranco': 'XIV',
    # X de los Lagos
    'Osorno': 'X', 'San Pablo': 'X', 'Puyehue': 'X', 'Puerto Octay': 'X', 
    'Purranque': 'X', 'Río Negro': 'X', 'San Juan de la Costa': 'X', 
    'Puerto Montt': 'X', 'Puerto Varas': 'X', 'Cochamó': 'X', 'Calbuco': 'X', 
    'Maullín': 'X', 'Los Muermos': 'X', 'Fresia': 'X', 'Llanquihue': 'X', 
    'Frutillar': 'X', 'Castro': 'X', 'Ancud': 'X', 'Quemchi': 'X', 'Dalcahue': 'X', 
    'Curaco de Vélez': 'X', 'Quinchao': 'X', 'Puqueldón': 'X', 'Chonchi': 'X', 
    'Queilén': 'X', 'Quellón': 'X', 'Chaitén': 'X', 'Hualaihué': 'X', 
    'Futaleufú': 'X', 'Palena': 'X',
    # XI Aysén del General Carlos Ibáñez del Campo
    'Coyhaique': 'XI', 'Lago Verde': 'XI', 'Aysén': 'XI', 'Cisnes': 'XI', 
    'Guaitecas': 'XI', 'Chile Chico': 'XI', 'Río Ibánez': 'XI', 'Cochrane': 'XI', 
    "O'Higgins": 'XI', 'Tortel': 'XI',
    # XII de Magallanes y Antártica Chilena
    'Natales': 'XII', 'Torres del Paine': 'XII', 'Punta Arenas': 'XII', 
    'Río Verde': 'XII', 'Laguna Blanca': 'XII', 'San Gregorio': 'XII', 
    'Porvenir': 'XII', 'Primavera': 'XII', 'Timaukel': 'XII', 'Cabo de Hornos': 'XII', 
    'Antártica': 'XII',
    # Metropolitana de Santiago (RM)
    'Santiago': 'RM', 'Independencia': 'RM', 'Conchalí': 'RM', 'Huechuraba': 'RM', 
    'Recoleta': 'RM', 'Providencia': 'RM', 'Vitacura': 'RM', 'Lo Barnechea': 'RM', 
    'Las Condes': 'RM', 'Ñuñoa': 'RM', 'La Reina': 'RM', 'Macul': 'RM', 
    'Peñalolén': 'RM', 'La Florida': 'RM', 'San Joaquín': 'RM', 'La Granja': 'RM', 
    'La Pintana': 'RM', 'San Ramón': 'RM', 'San Miguel': 'RM', 'La Cisterna': 'RM', 
    'El Bosque': 'RM', 'Pedro Aguirre Cerda': 'RM', 'Lo Espejo': 'RM', 
    'Estación Central': 'RM', 'Cerrillos': 'RM', 'Maipú': 'RM', 'Quinta Normal': 'RM', 
    'Lo Prado': 'RM', 'Pudahuel': 'RM', 'Cerro Navia': 'RM', 'Renca': 'RM', 
    'Quilicura': 'RM', 'Colina': 'RM', 'Lampa': 'RM', 'Tiltil': 'RM', 
    'Puente Alto': 'RM', 'San José de Maipo': 'RM', 'Pirque': 'RM', 
    'San Bernardo': 'RM', 'Buin': 'RM', 'Paine': 'RM', 'Calera de Tango': 'RM', 
    'Melipilla': 'RM', 'María Pinto': 'RM', 'Curacaví': 'RM', 'Alhué': 'RM', 
    'San Pedro': 'RM', 'Talagante': 'RM', 'Peñaflor': 'RM', 'Isla de Maipo': 'RM', 
    'El Monte': 'RM', 'Padre Hurtado': 'RM'
}

# Función para obtener región de una comuna
def obtener_region_por_comuna(comuna_str):
    """Obtiene la región de una comuna usando el mapeo oficial"""
    if pd.isna(comuna_str) or comuna_str == '':
        return None
    
    comuna_normalizada = str(comuna_str).strip().title()
    
    # Buscar coincidencia exacta
    if comuna_normalizada in MAPEO_COMUNA_REGION:
        return MAPEO_COMUNA_REGION[comuna_normalizada]
    
    # Buscar coincidencia sin acentos y con variaciones comunes
    comuna_sin_acentos = comuna_normalizada.replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u').replace('ñ', 'n')
    for comuna_key, region in MAPEO_COMUNA_REGION.items():
        comuna_key_sin_acentos = comuna_key.replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u').replace('ñ', 'n')
        if comuna_sin_acentos.lower() == comuna_key_sin_acentos.lower():
            return region
    
    return None

# Función para normalizar regiones a formato estándar
def normalizar_region(region_str):
    """Normaliza nombres de regiones a formato estándar (I, II, III, ..., XVI, RM)"""
    if pd.isna(region_str) or region_str == '':
        return 'Sin Región'
    
    region_str = str(region_str).strip().upper()
    
    # Mapeo de números a romanos
    numero_a_romano = {
        '1': 'I', '2': 'II', '3': 'III', '4': 'IV', '5': 'V',
        '6': 'VI', '7': 'VII', '8': 'VIII', '9': 'IX', '10': 'X',
        '11': 'XI', '12': 'XII', '13': 'XIII', '14': 'XIV', '15': 'XV', '16': 'XVI'
    }
    
    # Mapeo de variantes comunes
    variantes = {
        'RM': 'RM', 'REGION METROPOLITANA': 'RM', 'METROPOLITANA': 'RM',
        'METROPOLITANA DE SANTIAGO': 'RM', 'SANTIAGO': 'RM',
        'I': 'I', 'PRIMERA': 'I', 'TARAPACA': 'I', 'TARAPACÁ': 'I',
        'II': 'II', 'SEGUNDA': 'II', 'ANTOFAGASTA': 'II',
        'III': 'III', 'TERCERA': 'III', 'ATACAMA': 'III',
        'IV': 'IV', 'CUARTA': 'IV', 'COQUIMBO': 'IV',
        'V': 'V', 'QUINTA': 'V', 'VALPARAISO': 'V', 'VALPARAÍSO': 'V',
        'VI': 'VI', 'SEXTA': 'VI', "O'HIGGINS": 'VI', 'OHIGGINS': 'VI',
        'VII': 'VII', 'SEPTIMA': 'VII', 'SEPTIMA': 'VII', 'MAULE': 'VII',
        'VIII': 'VIII', 'OCTAVA': 'VIII', 'BIOBIO': 'VIII', 'BÍOBÍO': 'VIII', 'BIO BIO': 'VIII',
        'IX': 'IX', 'NOVENA': 'IX', 'ARAUCANIA': 'IX', 'ARAUCANÍA': 'IX',
        'X': 'X', 'DECIMA': 'X', 'DÉCIMA': 'X', 'LOS LAGOS': 'X',
        'XI': 'XI', 'DECIMA PRIMERA': 'XI', 'DÉCIMA PRIMERA': 'XI', 'AYSEN': 'XI', 'AYSÉN': 'XI',
        'XII': 'XII', 'DECIMA SEGUNDA': 'XII', 'DÉCIMA SEGUNDA': 'XII', 'MAGALLANES': 'XII',
        'XIV': 'XIV', 'DECIMA CUARTA': 'XIV', 'DÉCIMA CUARTA': 'XIV', 'LOS RIOS': 'XIV', 'LOS RÍOS': 'XIV',
        'XV': 'XV', 'DECIMA QUINTA': 'XV', 'DÉCIMA QUINTA': 'XV', 'ARICA Y PARINACOTA': 'XV',
        'XVI': 'XVI', 'DECIMA SEXTA': 'XVI', 'DÉCIMA SEXTA': 'XVI', 'ÑUBLE': 'XVI'
    }
    
    # Buscar variantes exactas primero
    if region_str in variantes:
        return variantes[region_str]
    
    # Buscar si contiene "REGION" o "REGIÓN" seguido de número
    if 'REGION' in region_str or 'REGIÓN' in region_str:
        # Extraer número o romano
        numeros = re.findall(r'\d+', region_str)
        if numeros:
            num = numeros[0]
            if num in numero_a_romano:
                return numero_a_romano[num]
        
        # Buscar números romanos
        for romano in ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI']:
            if romano in region_str:
                return romano
        
        # Buscar RM
        if 'METROPOLITANA' in region_str or 'SANTIAGO' in region_str:
            return 'RM'
    
    # Si es solo un número, convertir a romano
    if region_str.isdigit():
        if region_str in numero_a_romano:
            return numero_a_romano[region_str]
    
    # Si ya es un romano válido, devolverlo
    if region_str in ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI', 'RM']:
        return region_str
    
    # Si contiene valores inválidos
    if region_str in ['NAN', 'NONE', 'SIN REGIÓN', 'SIN REGION', 'SIN REGIóN', '']:
        return 'Sin Región'
    
    # Si no se puede normalizar, devolver el original (puede ayudar a debug)
    return region_str

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
            
            # Normalizar regiones usando función robusta
            df['region'] = df['region'].apply(normalizar_region)
            
            # Completar regiones faltantes usando mapeo de comunas
            # Identificar filas sin región o con 'Sin Región'
            sin_region_mask = (df['region'].isna()) | (df['region'] == 'Sin Región') | (df['region'] == '')
            
            if sin_region_mask.any():
                # Para cada comuna sin región, buscar en el mapeo
                for idx in df[sin_region_mask].index:
                    comuna = df.loc[idx, 'comuna']
                    region_encontrada = obtener_region_por_comuna(comuna)
                    if region_encontrada:
                        # Normalizar la región encontrada
                        df.loc[idx, 'region'] = normalizar_region(region_encontrada)
            
            # Validar y limpiar datos numéricos
            # Asegurar que num_incendios y area_quemada_ha sean numéricos
            if 'num_incendios' in df.columns:
                df['num_incendios'] = pd.to_numeric(df['num_incendios'], errors='coerce').fillna(0).astype(int)
            
            if 'area_quemada_ha' in df.columns:
                df['area_quemada_ha'] = pd.to_numeric(df['area_quemada_ha'], errors='coerce').fillna(0)
                # Asegurar que no haya valores negativos
                df['area_quemada_ha'] = df['area_quemada_ha'].clip(lower=0)
            
            # Validar consistencia: si hay incendios pero área es 0, puede ser válido (incendios muy pequeños)
            # pero también puede ser un error. Ajustar casos donde num_incendios > 0 y area_quemada_ha == 0
            # Asignar un mínimo razonable (0.01 ha = 100 m²) para incendios muy pequeños
            inconsistencias = (df['num_incendios'] > 0) & (df['area_quemada_ha'] == 0)
            if inconsistencias.any():
                # Para incendios registrados pero sin área, asignar un mínimo razonable
                # Esto representa incendios muy pequeños (< 1 ha) que fueron controlados rápidamente
                df.loc[inconsistencias, 'area_quemada_ha'] = 0.01  # 0.01 ha = 100 m² (mínimo razonable)
            
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
    # Obtener todas las regiones únicas, excluyendo 'Sin Región' y valores inválidos
    regiones_unicas = df_base['region'].dropna().unique()
    regiones_unicas = [
        r for r in regiones_unicas 
        if pd.notna(r) 
        and str(r).strip() != '' 
        and str(r) != 'Sin Región'
    ]
    
    # Función para ordenar regiones de forma inteligente
    def ordenar_region(region):
        region_str = str(region)
        # Mapeo de números romanos
        romanos = {'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'VI': 6, 
                  'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10, 'XI': 11, 'XII': 12,
                  'XIII': 13, 'XIV': 14, 'XV': 15, 'XVI': 16, 'RM': 17}
        
        if region_str in romanos:
            return (0, romanos[region_str])
        
        # Si no es una región conocida, poner al final
        return (1, region_str)
    
    # Ordenar regiones
    regiones_disponibles = sorted(regiones_unicas, key=ordenar_region)
else:
    regiones_disponibles = []

# Permitir selección múltiple de regiones
regiones_seleccionadas = st.sidebar.multiselect(
    "Seleccionar Región(es)",
    regiones_disponibles,
    default=[],
    key="region_select"
)

# Filtro de comunas (depende de región)
st.sidebar.subheader("🏘️ Filtro de Comunas")
# Manejar caso cuando no hay datos
if len(df_base) > 0 and 'comuna' in df_base.columns:
    # Si hay regiones seleccionadas, filtrar comunas por esas regiones
    if len(regiones_seleccionadas) > 0:
        # Normalizar las regiones seleccionadas para comparar con el dataframe
        regiones_seleccionadas_normalizadas = [normalizar_region(r) for r in regiones_seleccionadas]
        
        # Filtrar comunas que pertenecen a las regiones seleccionadas
        df_filtrado_region = df_base[df_base['region'].isin(regiones_seleccionadas_normalizadas)]
        
        # Obtener comunas únicas del dataframe filtrado
        comunas_unicas = df_filtrado_region['comuna'].dropna().unique()
        comunas_disponibles = sorted([
            c for c in comunas_unicas 
            if pd.notna(c) 
            and str(c).strip() != ''
            and str(c).strip().upper() not in ['NAN', 'NONE', 'CORPORACION', 'NACIONAL', 'FORESTAL']
        ])
    else:
        # Si no hay regiones seleccionadas, mostrar todas las comunas
        comunas_unicas = df_base['comuna'].dropna().unique()
        comunas_disponibles = sorted([
            c for c in comunas_unicas 
            if pd.notna(c) 
            and str(c).strip() != ''
            and str(c).strip().upper() not in ['NAN', 'NONE', 'CORPORACION', 'NACIONAL', 'FORESTAL']
        ])
else:
    comunas_disponibles = []

# Permitir selección múltiple de comunas
comunas_seleccionadas = st.sidebar.multiselect(
    "Seleccionar Comuna(s)",
    comunas_disponibles,
    default=[],
    key="comuna_select"
)

# Aplicar filtros
try:
    df_filtrado = df_base[
        (df_base['anio'] >= ano_inicio) &
        (df_base['anio'] <= ano_fin)
    ].copy()
    
    # Filtrar por regiones seleccionadas (si hay alguna seleccionada)
    if len(regiones_seleccionadas) > 0:
        # Normalizar las regiones seleccionadas para comparar con el dataframe
        regiones_seleccionadas_normalizadas = [normalizar_region(r) for r in regiones_seleccionadas]
        df_filtrado = df_filtrado[df_filtrado['region'].isin(regiones_seleccionadas_normalizadas)]
    
    # Filtrar por comunas seleccionadas (si hay alguna seleccionada)
    if len(comunas_seleccionadas) > 0:
        df_filtrado = df_filtrado[df_filtrado['comuna'].isin(comunas_seleccionadas)]
except Exception as e:
    st.sidebar.error(f"Error al aplicar filtros: {e}")
    df_filtrado = df_base.copy()

# Mostrar info de filtros
st.sidebar.markdown("---")
region_info = ", ".join(regiones_seleccionadas[:2]) if len(regiones_seleccionadas) > 0 else "Todas"
if len(regiones_seleccionadas) > 2:
    region_info += f" (+{len(regiones_seleccionadas)-2} más)"
comuna_info = ", ".join(comunas_seleccionadas[:2]) if len(comunas_seleccionadas) > 0 else "Todas"
if len(comunas_seleccionadas) > 2:
    comuna_info += f" (+{len(comunas_seleccionadas)-2} más)"

st.sidebar.info(f"""
**Datos Filtrados:**
- Registros: {len(df_filtrado):,}
- Años: {ano_inicio}-{ano_fin}
- Región(es): {region_info}
- Comuna(s): {comuna_info}
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
            # Información sobre calidad de datos
            if 'num_incendios' in df_filtrado.columns and 'area_quemada_ha' in df_filtrado.columns:
                # Contar casos con incendios pero área muy pequeña o cero
                incendios_pequenos = ((df_filtrado['num_incendios'] > 0) & (df_filtrado['area_quemada_ha'] < 0.1)).sum()
                if incendios_pequenos > 0:
                    with st.expander("ℹ️ Nota sobre calidad de datos", expanded=False):
                        st.info(f"""
                        **Incendios con área muy pequeña (< 0.1 ha):** {incendios_pequenos:,} registros
                        
                        Estos casos representan:
                        - ✅ **Incendios muy pequeños** controlados rápidamente (< 1,000 m²)
                        - ✅ **Incendios que no alcanzaron 1 hectárea** (redondeados a 0.01 ha)
                        - ⚠️ **Posibles errores en los datos originales** (incendios registrados sin área medida)
                        
                        Para mantener la consistencia, se ha asignado un mínimo de 0.01 ha (100 m²) a estos casos.
                        """)
            
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
                        # Agregar features temporales más robustas
                        panel_df['mes'] = 1  # Feature temporal básica
                        panel_df['dia_anio'] = panel_df['anio'] * 365  # Día del año aproximado
                        
                        # Feature de año (normalizado para ayudar al modelo)
                        anos_unicos = sorted(panel_df['anio'].unique())
                        ano_min = min(anos_unicos)
                        ano_max = max(anos_unicos)
                        panel_df['anio_normalizado'] = (panel_df['anio'] - ano_min) / (ano_max - ano_min + 1e-10)
                        
                        # Features cíclicas temporales (para capturar patrones estacionales)
                        panel_df['mes_sin'] = np.sin(2 * np.pi * panel_df['mes'] / 12)
                        panel_df['mes_cos'] = np.cos(2 * np.pi * panel_df['mes'] / 12)
                        
                        # Agregar features históricas básicas por comuna (promedios históricos)
                        historico_comuna = df_filtrado.groupby('comuna').agg({
                            'num_incendios': ['sum', 'mean', 'max', 'std'],
                            'area_quemada_ha': ['sum', 'mean']
                        }).reset_index()
                        historico_comuna.columns = ['comuna', 'incendios_total_hist', 'incendios_promedio_hist', 
                                                    'incendios_max_hist', 'incendios_std_hist', 
                                                    'area_total_hist', 'area_promedio_hist']
                        panel_df = panel_df.merge(historico_comuna, on='comuna', how='left')
                        
                        # Agregar features históricas temporales (incendios en años anteriores)
                        # Para cada comuna-año, calcular incendios en años anteriores
                        panel_df = panel_df.sort_values(['comuna', 'anio'])
                        panel_df['incendios_anio_anterior'] = panel_df.groupby('comuna')['num_incendios'].shift(1).fillna(0)
                        panel_df['incendios_2_anios_antes'] = panel_df.groupby('comuna')['num_incendios'].shift(2).fillna(0)
                        panel_df['area_anio_anterior'] = panel_df.groupby('comuna')['area_quemada_ha'].shift(1).fillna(0)
                        
                        # Promedio móvil de últimos 3 años
                        panel_df['incendios_promedio_3_anios'] = (
                            panel_df.groupby('comuna')['num_incendios']
                            .transform(lambda x: x.rolling(window=3, min_periods=1).mean().shift(1))
                            .fillna(0)  # Si no hay datos anteriores, usar 0
                        )
                        
                        # Llenar NaN en features históricas con 0
                        features_historicas = ['incendios_total_hist', 'incendios_promedio_hist', 
                                              'incendios_max_hist', 'incendios_std_hist', 
                                              'area_total_hist', 'area_promedio_hist']
                        for feat in features_historicas:
                            if feat in panel_df.columns:
                                panel_df[feat] = panel_df[feat].fillna(0)
                        
                        # Llenar NaN en features temporales con valores razonables
                        panel_df['incendios_std_hist'] = panel_df['incendios_std_hist'].fillna(0)
                        panel_df['incendios_promedio_3_anios'] = panel_df['incendios_promedio_3_anios'].fillna(0)
                        
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
                        
                        # Diagnóstico antes de entrenar
                        if task_type == 'regression':
                            with st.expander("🔍 Diagnóstico de Datos para Regresión", expanded=True):
                                st.markdown(f"""
                                **Distribución del target (num_incendios):**
                                - Total de muestras: {len(y):,}
                                - Mínimo: {y.min():.0f}
                                - Máximo: {y.max():.0f}
                                - Media: {y.mean():.3f}
                                - Mediana: {y.median():.3f}
                                - Desviación estándar: {y.std():.3f}
                                - Muestras con 0 incendios: {(y == 0).sum():,} ({(y == 0).mean()*100:.1f}%)
                                - Muestras con >0 incendios: {(y > 0).sum():,} ({(y > 0).mean()*100:.1f}%)
                                
                                **Features disponibles:**
                                - Número de features: {len(X.columns)}
                                - Features: {', '.join(X.columns[:10].tolist())}{'...' if len(X.columns) > 10 else ''}
                                
                                **💡 Nota:** Un R² negativo indica que el modelo predice peor que simplemente usar la media del target.
                                Esto puede ocurrir si:
                                - Las features no son suficientemente informativas
                                - Hay muchos valores cero (datos esparcidos)
                                - El modelo necesita más datos o features más relevantes
                                """)
                        
                        # Entrenar
                        metrics = predictor.train(X, y, validation_size=0.2, temporal_split=True)
                        
                        # Guardar en sesión
                        st.session_state.predictor = predictor
                        st.session_state.panel_data = panel_df
                        st.session_state.task_type = task_type  # Guardar tipo de tarea para predicción
                        st.session_state.model_type = model_type  # Guardar tipo de modelo para validación
                        # Limpiar mapa de riesgo anterior cuando se entrena un nuevo modelo
                        st.session_state.risk_map = None
                        
                        st.success(f"✅ Modelo {model_type.upper()} ({task_type}) entrenado exitosamente con datos reales")
                        
                        # Mostrar métricas según el tipo de tarea
                        st.subheader("📊 Métricas del Modelo")
                        
                        if task_type == 'classification':
                            # Métricas de clasificación
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
                        
                        else:  # regression
                            # Métricas de regresión
                            col_m1, col_m2, col_m3 = st.columns(3)
                            
                            with col_m1:
                                rmse_val = float(metrics.get('rmse', 0))
                                st.markdown(f"**RMSE:**")
                                st.markdown(f"### {rmse_val:.3f}")
                                st.caption("Raíz del error cuadrático medio (menor es mejor)")
                            
                            with col_m2:
                                mae_val = float(metrics.get('mae', 0))
                                st.markdown(f"**MAE:**")
                                st.markdown(f"### {mae_val:.3f}")
                                st.caption("Error absoluto medio (menor es mejor)")
                            
                            with col_m3:
                                r2_val = float(metrics.get('r2', 0))
                                st.markdown(f"**R²:**")
                                
                                # Mostrar R² con color según su valor
                                if r2_val < 0:
                                    st.markdown(f"### ⚠️ {r2_val:.3f}")
                                    st.caption("⚠️ **NEGATIVO**: El modelo es peor que predecir la media")
                                elif r2_val < 0.3:
                                    st.markdown(f"### ⚠️ {r2_val:.3f}")
                                    st.caption("⚠️ **BAJO**: El modelo explica poca variabilidad")
                                elif r2_val < 0.7:
                                    st.markdown(f"### {r2_val:.3f}")
                                    st.caption("⚠️ **MODERADO**: El modelo explica variabilidad moderada")
                                else:
                                    st.markdown(f"### ✅ {r2_val:.3f}")
                                    st.caption("✅ **ALTO**: El modelo explica mucha variabilidad")
                            
                            # Información adicional sobre interpretación
                            if r2_val < 0:
                                st.error(f"""
                                **⚠️ R² NEGATIVO ({r2_val:.3f}) - El modelo está funcionando muy mal:**
                                
                                Esto significa que el modelo predice **peor que simplemente usar la media** del target.
                                
                                **Posibles causas:**
                                1. **Features insuficientes**: Las features no capturan patrones relevantes
                                2. **Datos esparcidos**: Muchos valores en cero hacen difícil aprender patrones
                                3. **Overfitting**: El modelo memoriza el entrenamiento pero no generaliza
                                4. **Split temporal problemático**: Datos de validación muy diferentes a entrenamiento
                                5. **Modelo inadecuado**: El algoritmo puede no ser el mejor para estos datos
                                
                                **Soluciones sugeridas:**
                                - ✅ Usa **classification** en lugar de regression (predice ocurrencia, no cantidad)
                                - ✅ Incluye más features relevantes (datos climáticos, geográficos)
                                - ✅ Aumenta el rango de años en los filtros
                                - ✅ Considera transformar el target (log, binning)
                                """)
                            else:
                                st.info(f"""
                                **Interpretación de métricas de regresión:**
                                
                                - **RMSE ({rmse_val:.3f})**: Error promedio en la misma unidad que el target. 
                                  Indica cuántos incendios se predice incorrectamente en promedio.
                                  {f"⚠️ Alto: {rmse_val:.1f} errores en promedio" if rmse_val > 10 else "✅ Razonable"}
                                
                                - **MAE ({mae_val:.3f})**: Error absoluto promedio. Más fácil de interpretar que RMSE.
                                
                                - **R² ({r2_val:.3f})**: Porcentaje de variabilidad explicada por el modelo.
                                  - R² = 1.0: Predicción perfecta
                                  - R² = 0.0: El modelo no es mejor que predecir la media
                                  - R² < 0.0: El modelo es peor que predecir la media ⚠️
                                  - R² > 0.7: Buen ajuste ✅
                                """)
                        
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
            # Mostrar información del modelo actual
            model_type_actual = st.session_state.get('model_type', 'desconocido')
            task_type_actual = st.session_state.get('task_type', 'desconocido')
            st.success(f"✅ Modelo {model_type_actual.upper()} ({task_type_actual}) entrenado y listo para hacer predicciones")
            
            # Mostrar advertencia si el modelo actual no coincide con el seleccionado
            if model_type_actual != model_type:
                st.warning(f"⚠️ **IMPORTANTE:** El modelo actualmente entrenado es **{model_type_actual.upper()}**, pero has seleccionado **{model_type.upper()}** en el selector. "
                          f"Las predicciones usarán el modelo **{model_type_actual.upper()}** que está actualmente entrenado. "
                          f"Para usar **{model_type.upper()}**, haz clic en '🚀 Entrenar Modelo' con el tipo seleccionado.")
            
            # Mostrar información sobre el modelo actual
            with st.expander("ℹ️ Información del Modelo Actual", expanded=False):
                st.markdown(f"""
                **Modelo actualmente entrenado:** {model_type_actual.upper()}  
                **Tipo de tarea:** {task_type_actual}  
                **Modelo seleccionado en el selector:** {model_type.upper()}  
                
                **💡 Nota:** Si quieres cambiar el modelo, selecciona el tipo de modelo que deseas y haz clic en "🚀 Entrenar Modelo". 
                Cada modelo (XGBoost, LightGBM, Random Forest) puede dar resultados diferentes incluso con los mismos datos.
                """)
            
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
                        
                        # Validar que el modelo está entrenado
                        if st.session_state.predictor.model is None:
                            st.error("❌ Error: El modelo no está entrenado. Por favor entrena el modelo primero.")
                        else:
                            # Mostrar información del modelo que se está usando
                            model_type_usado = st.session_state.get('model_type', 'desconocido')
                            task_type_usado = st.session_state.get('task_type', 'desconocido')
                            
                            st.info(f"🔍 Usando modelo **{model_type_usado.upper()}** ({task_type_usado}) para predicción")
                            
                            # Preparar features para predicción - pasar target_col aunque no se use
                            X_pred, _ = st.session_state.predictor.prepare_features(pred_df, target_col='incendio_ocurrencia')
                            
                            # Verificar que las features son correctas
                            if X_pred is None or len(X_pred) == 0:
                                st.error("❌ Error: No se pudieron preparar las features para predicción")
                            else:
                                st.info(f"📊 Prediciendo riesgo para {len(X_pred)} comunas con {len(X_pred.columns)} features")
                                
                                # Mostrar estadísticas de las predicciones
                                if task_type_pred == 'classification':
                                    riesgos = st.session_state.predictor.predict(X_pred, return_proba=True)
                                else:
                                    predicciones = st.session_state.predictor.predict(X_pred)
                                    riesgos = (predicciones - predicciones.min()) / (predicciones.max() - predicciones.min() + 1e-10)
                                
                                # Mostrar estadísticas detalladas de las predicciones
                                st.info(f"📈 Estadísticas de predicción del modelo **{model_type_usado.upper()}**: "
                                       f"Min={riesgos.min():.4f}, "
                                       f"Max={riesgos.max():.4f}, "
                                       f"Mean={riesgos.mean():.4f}, "
                                       f"Std={riesgos.std():.4f}, "
                                       f"Median={np.median(riesgos):.4f}")
                                
                                # Mostrar información de debug para verificar que el modelo es diferente
                                if hasattr(st.session_state.predictor.model, 'n_estimators'):
                                    n_estimators = st.session_state.predictor.model.n_estimators
                                    st.info(f"🔍 Debug: Modelo {model_type_usado} con {n_estimators} estimadores")
                                
                                # Verificar que hay variabilidad en las predicciones
                                if riesgos.std() < 0.001:
                                    st.warning("⚠️ **Advertencia:** Las predicciones tienen muy poca variabilidad (std < 0.001). "
                                              "Esto podría indicar que el modelo está prediciendo valores muy similares para todas las comunas. "
                                              "Esto es normal si las features históricas son muy similares entre comunas o si el modelo tiene un sesgo fuerte.")
                                
                                risk_map = pd.DataFrame({
                                    'comuna': comunas_unicas,
                                    'riesgo_probabilidad': riesgos,
                                    'incendios_historico': historico_comuna['incendios_total'].values,
                                    'area_historica': historico_comuna['area_total'].values
                                })
                                
                                # Guardar también el tipo de modelo usado para esta predicción
                                risk_map['modelo_usado'] = model_type_usado
                                risk_map['task_type'] = task_type_usado
                                
                                st.session_state.risk_map = risk_map
                                st.success(f"✅ Mapa de riesgo generado usando modelo {model_type_usado.upper()}")
                        
                    except Exception as e:
                        st.error(f"Error al generar mapa de riesgo: {str(e)}")
            
            # Mostrar mapa de riesgo
            if st.session_state.risk_map is not None:
                risk_map = st.session_state.risk_map.copy()
                
                # Mostrar información del modelo usado para esta predicción
                modelo_usado_pred = risk_map.get('modelo_usado', st.session_state.get('model_type', 'desconocido')).iloc[0] if 'modelo_usado' in risk_map.columns else st.session_state.get('model_type', 'desconocido')
                task_usado_pred = risk_map.get('task_type', st.session_state.get('task_type', 'desconocido')).iloc[0] if 'task_type' in risk_map.columns else st.session_state.get('task_type', 'desconocido')
                
                st.info(f"📊 Mapa de riesgo generado con modelo **{modelo_usado_pred.upper()}** ({task_usado_pred})")
                
                # Eliminar columnas de metadatos para mostrar
                columnas_mostrar = ['comuna', 'riesgo_probabilidad', 'incendios_historico', 'area_historica']
                
                st.subheader("📋 Riesgo por Comuna")
                try:
                    risk_map_sorted = risk_map.sort_values('riesgo_probabilidad', ascending=False)
                    risk_map_sorted['riesgo_categoria'] = pd.cut(
                        risk_map_sorted['riesgo_probabilidad'],
                        bins=[0, 0.3, 0.6, 1.0],
                        labels=['Bajo', 'Medio', 'Alto']
                    )
                    
                    # Filtrar columnas para mostrar (excluir metadatos)
                    columnas_display = [col for col in ['comuna', 'riesgo_probabilidad', 'riesgo_categoria', 'incendios_historico', 'area_historica'] if col in risk_map_sorted.columns]
                    
                    st.dataframe(
                        risk_map_sorted[columnas_display].head(20),
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
        region_str = ", ".join(regiones_seleccionadas) if len(regiones_seleccionadas) > 0 else "Todas las Regiones"
        comuna_str = ", ".join(comunas_seleccionadas[:2]) if len(comunas_seleccionadas) > 0 else "Todas las Comunas"
        if len(comunas_seleccionadas) > 2:
            comuna_str += f" (+{len(comunas_seleccionadas)-2} más)"
        st.info(f"🎯 Optimizando recursos para {region_str} ({comuna_str})")
        
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
                if len(regiones_seleccionadas) == 0:
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
                    
                    region_title = ", ".join(regiones_seleccionadas[:2]) if len(regiones_seleccionadas) > 0 else "Todas las Regiones"
                    if len(regiones_seleccionadas) > 2:
                        region_title += f" (+{len(regiones_seleccionadas)-2} más)"
                    fig_comuna = px.bar(
                        comunas_region,
                        x='num_incendios',
                        y='comuna',
                        orientation='h',
                        title=f'Top 10 Comunas con Más Incendios ({region_title})',
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
                    file_name=f"incendios_conaf_{ano_inicio}_{ano_fin}_{'_'.join(regiones_seleccionadas[:2]) if len(regiones_seleccionadas) > 0 else 'todas'}.csv",
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
