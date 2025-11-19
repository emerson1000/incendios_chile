# 🚀 Guía Rápida: Subir Proyecto a GitHub y Streamlit Cloud

## ✅ Estado Actual

Tu proyecto está listo:
- ✅ Git inicializado
- ✅ Archivos preparados
- ✅ README.md profesional creado
- ✅ .gitignore configurado
- ✅ requirements.txt actualizado
- ✅ Configuración Streamlit Cloud lista

## 📤 PASO 1: Agregar Archivos y Hacer Commit

```powershell
# Agregar todos los archivos
git add .

# Ver qué se va a subir
git status

# Hacer commit inicial
git commit -m "Initial commit: Sistema de predicción y optimización de recursos para incendios forestales Chile"
```

## 🔗 PASO 2: Crear Repositorio en GitHub

1. **Ve a GitHub**: https://github.com/new
2. **Crea nuevo repositorio**:
   - **Repository name**: `incendios-chile` (o el nombre que prefieras)
   - **Description**: "Sistema de predicción y optimización de recursos para incendios forestales en Chile usando datos CONAF"
   - **Visibilidad**: Public (recomendado para Streamlit Cloud gratuito) o Private
   - **NO marques** "Initialize with README" (ya lo tenemos)
   - **NO marques** "Add .gitignore" (ya lo tenemos)
   - **NO marques** "Choose a license" (ya incluimos MIT)
   - Click en **"Create repository"**

## 📤 PASO 3: Conectar y Subir

**⚠️ IMPORTANTE**: Reemplaza `TU_USUARIO` con tu usuario real de GitHub

```powershell
# Agregar remote (HTTPS - más fácil)
git remote add origin https://github.com/TU_USUARIO/incendios-chile.git

# O si prefieres SSH (necesitas configurar SSH keys primero):
# git remote add origin git@github.com:TU_USUARIO/incendios-chile.git

# Renombrar rama a main
git branch -M main

# Subir código a GitHub
git push -u origin main
```

### Si GitHub pide autenticación:

**Opción 1: Personal Access Token (Recomendado)**

1. Ve a: https://github.com/settings/tokens
2. Click en "Generate new token (classic)"
3. Dale un nombre (ej: "incendios-project")
4. Selecciona scopes: ✅ `repo` (todos los permisos)
5. Click en "Generate token"
6. **Copia el token** (solo se muestra una vez)
7. Cuando git pida contraseña:
   - Usuario: tu usuario de GitHub
   - Contraseña: **pega el token** (no tu contraseña normal)

**Opción 2: GitHub CLI**

```powershell
# Instalar GitHub CLI si no lo tienes
# winget install --id GitHub.cli

# Autenticarse
gh auth login

# Luego hacer push normalmente
git push -u origin main
```

## 🌐 PASO 4: Deploy en Streamlit Cloud

1. **Ve a Streamlit Cloud**: https://share.streamlit.io
2. **Inicia sesión** con tu cuenta de GitHub
3. **Click en "New app"**
4. **Configura tu app**:
   - **Repository**: Selecciona `TU_USUARIO/incendios-chile`
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py` (o `dashboard.py` - ambos funcionan)
   - **Python version**: `3.9` (o la que prefieras, 3.9+)
5. **Click en "Deploy!"**
6. ⏳ Espera unos minutos mientras Streamlit Cloud:
   - Clona el repositorio
   - Instala dependencias de `requirements.txt`
   - Ejecuta `streamlit run dashboard.py`

## ⚠️ IMPORTANTE: Datos para Streamlit Cloud

El dashboard necesita el dataset `conaf_datos_reales_completo.csv`. Tienes **2 opciones**:

### Opción A: Incluir Dataset en el Repo (Más Simple)

```powershell
# 1. Temporalmente, edita .gitignore y comenta estas líneas:
#    data/processed/*
#    *.csv

# 2. Agregar el dataset
git add data/processed/conaf_datos_reales_completo.csv

# 3. Commit
git commit -m "Add processed CONAF dataset"

# 4. Push
git push

# 5. Luego descomenta las líneas en .gitignore para futuros cambios
```

**Ventajas**: Datos disponibles inmediatamente  
**Desventajas**: Repo más grande (~1.3 MB adicionales)

### Opción B: Procesar Datos Automáticamente (Recomendado)

El dashboard ya tiene código para procesar datos automáticamente si no existen. Solo necesitas:

1. **Incluir los archivos CONAF originales** en `data/raw/`:
   - Los archivos Excel/XLS de CONAF
   - Estos son más pequeños que el dataset procesado

```powershell
# Agregar archivos raw (temporalmente comentar data/raw/* en .gitignore)
git add data/raw/*.xls data/raw/*.xlsx

# Commit
git commit -m "Add CONAF raw data files"

# Push
git push
```

**Ventajas**: Repo más limpio, datos frescos  
**Desventajas**: Primera carga puede tardar un poco más

## 🔍 Verificación Post-Deploy

Una vez desplegado en Streamlit Cloud, verifica:

1. ✅ El dashboard carga sin errores
2. ✅ Los filtros aparecen en la barra lateral
3. ✅ Los datos se visualizan correctamente
4. ✅ Puedes entrenar modelos
5. ✅ La optimización funciona

## 📋 Comandos Rápidos

```powershell
# Ver estado
git status

# Ver diferencias
git diff

# Ver historial
git log --oneline

# Para futuros cambios:
git add .
git commit -m "Descripción del cambio"
git push

# Ver remotes configurados
git remote -v

# Actualizar desde GitHub
git pull origin main
```

## 🆘 Troubleshooting

### Error: "Repository not found"
- Verifica que el repositorio existe en GitHub
- Verifica que el nombre de usuario sea correcto
- Verifica que tengas permisos de escritura

### Error: "Authentication failed"
- Usa Personal Access Token en lugar de contraseña
- Verifica que el token tenga permisos `repo`

### Error en Streamlit Cloud: "ModuleNotFoundError"
- Verifica que todas las dependencias estén en `requirements.txt`
- Algunos paquetes pueden necesitar versiones específicas

### Error: "FileNotFoundError: conaf_datos_reales_completo.csv"
- El dataset debe estar en el repo O
- Los archivos raw deben estar para procesamiento automático

## 🎯 URLs Después del Deploy

- **Repositorio GitHub**: `https://github.com/TU_USUARIO/incendios-chile`
- **Streamlit App**: `https://TU-APP-NAME.streamlit.app`

## 📝 Notas Importantes

1. **Datos grandes**: El dataset procesado (~1.3 MB) puede incluirse en el repo sin problemas
2. **Secrets**: Si necesitas API keys, configúralas en Streamlit Cloud → Settings → Secrets
3. **Actualizaciones**: Cada vez que hagas `git push`, Streamlit Cloud redeployará automáticamente
4. **Límites**: Streamlit Cloud gratuito tiene límites, pero son generosos para este proyecto

---

**¡Listo para deploy!** 🚀

Si tienes dudas, consulta `setup_for_streamlit_cloud.md` o `GITHUB_SETUP.md` para más detalles.

