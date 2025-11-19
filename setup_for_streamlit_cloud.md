# 🚀 Guía para Desplegar en Streamlit Cloud

## Pasos para Subir a GitHub y Streamlit Cloud

### 1. Preparar el Proyecto

```bash
# Asegúrate de estar en el directorio del proyecto
cd c:\Users\emers\OneDrive\Documentos\incendios

# Verifica que los archivos necesarios estén presentes
ls dashboard.py
ls requirements.txt
ls .streamlit/config.toml
```

### 2. Inicializar Git (si no está inicializado)

```bash
# Inicializar repositorio
git init

# Agregar archivos
git add .

# Hacer commit inicial
git commit -m "Initial commit: Sistema de predicción y optimización de recursos para incendios forestales"
```

### 3. Crear Repositorio en GitHub

1. Ve a [github.com](https://github.com) e inicia sesión
2. Click en "New repository" (botón verde)
3. Nombre del repositorio: `incendios-chile` (o el que prefieras)
4. Descripción: "Sistema de predicción y optimización de recursos para incendios forestales en Chile"
5. **NO** inicialices con README, .gitignore o licencia (ya los tenemos)
6. Click en "Create repository"

### 4. Conectar Repositorio Local con GitHub

```bash
# Agregar remote (reemplaza TU_USUARIO con tu usuario de GitHub)
git remote add origin https://github.com/TU_USUARIO/incendios-chile.git

# Renombrar rama a main
git branch -M main

# Subir código
git push -u origin main
```

Si GitHub te pide autenticación, puedes usar:
- GitHub CLI (`gh auth login`)
- Token de acceso personal (PAT)
- GitHub Desktop (GUI más fácil)

### 5. Configurar Streamlit Cloud

1. Ve a [share.streamlit.io](https://share.streamlit.io)
2. Inicia sesión con tu cuenta de GitHub
3. Click en "New app"
4. Selecciona:
   - **Repository**: `TU_USUARIO/incendios-chile`
   - **Branch**: `main`
   - **Main file path**: `dashboard.py`
5. Click en "Deploy!"

### 6. ⚠️ IMPORTANTE: Datos para Streamlit Cloud

Streamlit Cloud necesita acceso a los datos. Tienes 3 opciones:

#### Opción A: Incluir Dataset Procesado en el Repo (Más Simple)

```bash
# Editar .gitignore temporalmente para incluir el dataset
# Comentar estas líneas en .gitignore:
# data/processed/*
# *.csv

# Agregar el dataset
git add data/processed/conaf_datos_reales_completo.csv
git commit -m "Add processed CONAF dataset"
git push
```

**Nota**: Esto hará el repo más grande (el CSV tiene ~1.3 MB), pero es la forma más simple.

#### Opción B: Usar GitHub Releases (Recomendado)

1. Sube el dataset como Release en GitHub
2. Usa `requests` para descargarlo automáticamente en el dashboard

#### Opción C: Storage Externo (Más Complejo)

Usa Google Drive, AWS S3, o similar y configura Streamlit Secrets.

### 7. Configurar Secrets (si usas APIs externas)

Si necesitas API keys:
1. En Streamlit Cloud, ve a "Settings" → "Secrets"
2. Agrega variables como:
```toml
NASA_FIRMS_API_KEY = "tu_key"
CR2_API_KEY = "tu_key"
```

### 8. Esperar el Deploy

Streamlit Cloud automáticamente:
- Instalará todas las dependencias de `requirements.txt`
- Ejecutará `streamlit run dashboard.py`
- Te dará una URL pública: `https://TU-APP-NAME.streamlit.app`

## 📋 Checklist Pre-Deploy

- [ ] `.gitignore` está configurado correctamente
- [ ] `requirements.txt` tiene todas las dependencias
- [ ] `.streamlit/config.toml` existe
- [ ] `dashboard.py` es el archivo principal
- [ ] Los datos CONAF están procesados o configurados para descargarse
- [ ] `README.md` está actualizado
- [ ] Código está funcionando localmente

## 🐛 Troubleshooting

### Error: "ModuleNotFoundError"
- Verifica que todas las dependencias estén en `requirements.txt`
- Algunos paquetes pueden necesitar versión específica para Streamlit Cloud

### Error: "FileNotFoundError: conaf_datos_reales_completo.csv"
- El dataset debe estar en el repositorio o configurado para descargarse
- Verifica la ruta en `dashboard.py`

### Error: "Out of memory"
- Streamlit Cloud tiene límites de memoria
- Considera usar un dataset más pequeño o optimizar el código

## 📝 Comandos Útiles

```bash
# Ver estado de git
git status

# Agregar archivos específicos
git add dashboard.py requirements.txt

# Ver qué se va a subir
git status

# Hacer commit
git commit -m "Descripción del cambio"

# Subir cambios
git push

# Ver commits
git log --oneline

# Crear nueva rama
git checkout -b feature/nueva-funcionalidad
```

## 🔗 Recursos

- [GitHub Docs](https://docs.github.com/)
- [Streamlit Cloud Docs](https://docs.streamlit.io/streamlit-community-cloud)
- [Streamlit Deploy Guide](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app)

