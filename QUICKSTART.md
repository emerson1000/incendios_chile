# 🚀 Guía Rápida de Inicio

## Instalación Rápida

```bash
# 1. Clonar/descargar el proyecto
cd incendios

# 2. Crear entorno virtual (opcional pero recomendado)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

## Uso Rápido

### Opción 1: Dashboard (Más Fácil) ⭐

```bash
streamlit run dashboard.py
```

Abre tu navegador en `http://localhost:8501` y:
1. Ve a la pestaña "📊 Datos y ETL"
2. Haz clic en "🔄 Generar/Cargar Datos"
3. Ve a "🤖 Predicción de Riesgo"
4. Haz clic en "🚀 Entrenar Modelo"
5. Haz clic en "🔮 Generar Predicción"
6. Ve a "🎯 Optimización de Recursos"
7. Haz clic en "⚙️ Optimizar Asignación"

### Opción 2: Script de Ejemplo

```bash
python example_usage.py
```

Esto ejecuta todo el pipeline:
- Genera datos sintéticos
- Entrena modelo
- Genera predicciones
- Optimiza asignación de recursos

### Opción 3: Script Principal

```bash
# Pipeline completo
python main.py --mode full

# Solo ETL
python main.py --mode etl

# Solo entrenar modelo
python main.py --mode train --model-type xgboost

# Solo predicción
python main.py --mode predict

# Solo optimización
python main.py --mode optimize --max-brigades 50
```

### Opción 4: Programático

```python
from src.data.etl import FireDataETL
from src.models.prediction import FireRiskPredictor
from src.optimization.resource_allocation import ResourceAllocationOptimizer

# Ver ejemplo completo en example_usage.py
```

## Estructura de Datos

### Datos Procesados
Los datos procesados se guardan en:
- `data/processed/panel_incendios.parquet`

### Modelos Entrenados
Los modelos se guardan en:
- `models/fire_risk_model_*.pkl`

### Resultados
Los resultados se guardan en:
- `results/risk_map_*.csv` - Mapas de riesgo
- `results/allocation_*.csv` - Asignación de recursos

## Configuración

Edita `config.py` para ajustar:
- Tipo de modelo (xgboost, lightgbm, random_forest)
- Número máximo de brigadas
- Número máximo de bases
- Features del modelo
- Parámetros de optimización

## Datos Reales

Por defecto, el sistema genera datos sintéticos. Para usar datos reales:

1. Descarga datasets de:
   - CONAF: https://www.conaf.cl/
   - CR2: http://www.cr2.cl/
   - NASA FIRMS: https://firms.modaps.eosdis.nasa.gov/

2. Colócalos en `data/raw/`

3. Modifica `src/data/etl.py` para cargar tus archivos

## Próximos Pasos

1. ✅ Instalar dependencias
2. ✅ Ejecutar dashboard o script de ejemplo
3. 📖 Revisar notebooks en `notebooks/`
4. ⚙️ Ajustar configuración en `config.py`
5. 🔄 Integrar datos reales
6. 🎯 Personalizar modelos y features

## Solución de Problemas

### Error: "ModuleNotFoundError"
```bash
pip install -r requirements.txt
```

### Error: "No module named 'streamlit_folium'"
```bash
pip install folium streamlit-folium
```

### Error: "Solver not found" (optimización)
Instala un solver de optimización:
- Windows/Mac: `pip install pulp` (incluye CBC)
- Linux: `sudo apt-get install coinor-cbc`

### Datos muy lentos
- Usa menos años de datos históricos
- Reduce número de comunas
- Usa `temporal_split=True` para entrenamiento más rápido

## Recursos Adicionales

- 📖 **README.md**: Documentación completa
- 📓 **notebooks/**: Ejemplos detallados
- 💻 **example_usage.py**: Código de ejemplo
- ⚙️ **config.py**: Configuración del proyecto

---

**¿Preguntas?** Abre un issue en el repositorio.

