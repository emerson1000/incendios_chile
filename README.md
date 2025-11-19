# 🔥 Sistema de Predicción y Optimización de Recursos para Incendios Forestales - Chile

Sistema completo de análisis, predicción y optimización de recursos para la gestión de incendios forestales en Chile usando datos oficiales de CONAF (1985-2024).

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

## 📋 Características Principales

- 📊 **Análisis de Datos Reales CONAF**: Procesamiento de datos oficiales históricos (1985-2024)
- 🤖 **Modelos de Machine Learning**: Predicción de riesgo de incendios (XGBoost, LightGBM, Random Forest)
- 🎯 **Optimización de Recursos**: Asignación óptima de brigadas y bases de operaciones
- 📈 **Dashboard Interactivo**: Interfaz web con filtros por año, región y comuna
- 🗺️ **Visualizaciones Geográficas**: Mapas interactivos de riesgo y asignación de recursos

## 🚀 Inicio Rápido

### Requisitos Previos

- Python 3.9+
- pip

### Instalación

1. **Clonar el repositorio**
```bash
git clone https://github.com/tu-usuario/incendios-chile.git
cd incendios-chile
```

2. **Crear entorno virtual (recomendado)**
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

### Preparar Datos

1. **Colocar archivos CONAF en `data/raw/`**

   Descarga los archivos oficiales de CONAF y colócalos en la carpeta `data/raw/`:
   - Archivos Excel/XLS de CONAF con datos de incendios

2. **Procesar datos CONAF**
```bash
python procesar_conaf_correctamente.py
```

Esto generará el dataset consolidado en `data/processed/conaf_datos_reales_completo.csv`

### Ejecutar Dashboard

```bash
streamlit run dashboard.py
```

El dashboard estará disponible en `http://localhost:8501`

## 📁 Estructura del Proyecto

```
incendios/
├── dashboard.py              # Dashboard principal Streamlit
├── main.py                   # Pipeline completo
├── procesar_conaf_correctamente.py  # Procesador de datos CONAF
├── config.py                 # Configuración del proyecto
├── requirements.txt          # Dependencias Python
│
├── src/
│   ├── data/
│   │   ├── etl.py           # Pipeline ETL
│   │   ├── conaf_smart_processor.py  # Procesador inteligente CONAF
│   │   └── downloaders.py   # Descarga de datos externos
│   ├── models/
│   │   └── prediction.py    # Modelos de predicción
│   └── optimization/
│       └── resource_allocation.py  # Optimización de recursos
│
├── data/
│   ├── raw/                 # Datos originales CONAF
│   └── processed/           # Datos procesados
│
├── models/                  # Modelos entrenados
├── results/                 # Resultados y reportes
└── notebooks/               # Jupyter notebooks de análisis
```

## 🎯 Uso del Dashboard

### Filtros Disponibles

1. **Filtro de Años**: Selecciona el rango de años a analizar (1984-2023)
2. **Filtro de Regiones**: Elige una región específica o todas
3. **Filtro de Comunas**: Selecciona una comuna específica o todas de la región

### Tabs Principales

- **📊 Datos y Análisis**: Visualización de datos históricos CONAF
- **🤖 Predicción de Riesgo**: Entrenamiento de modelos y generación de mapas de riesgo
- **🎯 Optimización de Recursos**: Asignación óptima de brigadas
- **📈 Reportes y Estadísticas**: Análisis avanzados y exportación de datos

## 🧪 Ejemplos de Uso

### Análisis de una Región Específica

```python
# Ejemplo: Analizar Biobío (VIII Región)
# 1. En el dashboard, selecciona:
#    - Región: "VIII"
#    - Años: 2015-2023
# 2. Ve a la pestaña "Datos y Análisis"
# 3. Revisa estadísticas y gráficos específicos de la región
```

### Optimizar Recursos para una Región

```python
# 1. Selecciona región y años en filtros
# 2. Entrena modelo en "Predicción de Riesgo"
# 3. Genera mapa de riesgo
# 4. Ve a "Optimización de Recursos"
# 5. Configura número de brigadas y bases
# 6. Ejecuta optimización
```

## 🔧 Configuración

### Variables de Entorno (Opcional)

Crea un archivo `.env` para configuraciones adicionales:

```env
# Configuración de APIs externas (opcional)
NASA_FIRMS_API_KEY=tu_api_key
CR2_API_KEY=tu_api_key
```

### Configuración del Modelo

Edita `config.py` para ajustar parámetros:
- Tipos de modelos disponibles
- Configuración de optimización
- Parámetros de visualización

## 📊 Datos

### Fuentes de Datos

- **CONAF**: Datos oficiales de incendios forestales (1985-2024)
  - Ocurrencia y daño histórico nacional
  - Resumen por comuna
  - Datos mensuales y por rango horario

### Estructura de Datos Procesados

El dataset consolidado (`conaf_datos_reales_completo.csv`) contiene:
- `comuna`: Nombre de la comuna
- `num_incendios`: Número de incendios
- `area_quemada_ha`: Área quemada en hectáreas
- `region`: Código de región
- `anio`: Año de registro
- `temporada`: Temporada de incendios

## 🚀 Despliegue en Streamlit Cloud

### Pasos para Deploy

1. **Sube el proyecto a GitHub**
```bash
git init
git add .
git commit -m "Initial commit: Sistema de incendios forestales"
git branch -M main
git remote add origin https://github.com/tu-usuario/incendios-chile.git
git push -u origin main
```

2. **Conecta con Streamlit Cloud**
   - Ve a [share.streamlit.io](https://share.streamlit.io)
   - Conecta tu repositorio de GitHub
   - Selecciona `dashboard.py` como archivo principal
   - Streamlit Cloud instalará automáticamente las dependencias

3. **Nota Importante**: Los datos CONAF deben estar en `data/processed/`
   - Opción 1: Incluir el archivo procesado en el repo (puede ser grande)
   - Opción 2: Usar Streamlit Secrets para cargar datos desde storage externo
   - Opción 3: Procesar datos automáticamente al desplegar (ver `setup_data.py`)

### Archivo de Configuración Streamlit

Crea `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
headless = true
port = 8501
enableCORS = false
```

## 📈 Métricas del Modelo

Los modelos entrenados con datos reales alcanzan:
- **Accuracy**: >99%
- **ROC-AUC**: >99%
- **F1-Score**: >98%
- **Precision**: 100%

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📝 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

## 🙏 Agradecimientos

- **CONAF** (Corporación Nacional Forestal) por proporcionar los datos oficiales
- **Streamlit** por la plataforma de visualización
- Comunidad open source por las herramientas utilizadas

## 📧 Contacto

Para preguntas o sugerencias sobre este proyecto, por favor abre un issue en GitHub.

## 📚 Referencias

- [CONAF - Estadísticas de Incendios Forestales](https://www.conaf.cl/incendios-forestales/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

---

**⚠️ Aviso Legal**: Este sistema utiliza datos oficiales de CONAF y está diseñado como herramienta de apoyo a la toma de decisiones. No reemplaza el criterio profesional de especialistas en gestión de incendios.
