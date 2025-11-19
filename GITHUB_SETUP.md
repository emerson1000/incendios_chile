# 📦 Guía Completa: Subir Proyecto a GitHub y Streamlit Cloud

## ✅ Preparación Completa - TODO LISTO

Ya has creado:
- ✅ `README.md` profesional
- ✅ `.gitignore` configurado
- ✅ `requirements.txt` actualizado
- ✅ `.streamlit/config.toml` para Streamlit Cloud
- ✅ Dashboard funcional con datos reales

## 🚀 PASO 1: Inicializar Git (si no está hecho)

```powershell
# Navegar al directorio del proyecto
cd c:\Users\emers\OneDrive\Documentos\incendios

# Inicializar repositorio (si no está inicializado)
git init

# Verificar estado
git status
```

## 📝 PASO 2: Preparar Archivos para Git

### Opción A: Excluir Datos Grandes (Recomendado)

Los archivos grandes (CSV, modelos, resultados) ya están en `.gitignore`. 

**Importante**: Para Streamlit Cloud, necesitarás incluir el dataset procesado. Tienes dos opciones:

#### Sub-opción 1: Incluir dataset procesado temporalmente

```powershell
# Temporalmente, comentar estas líneas en .gitignore:
# data/processed/*
# *.csv

# Agregar el dataset
git add data/processed/conaf_datos_reales_completo.csv
```

#### Sub-opción 2: Descargar datos automáticamente (Recomendado)

El dashboard ya tiene código para procesar datos automáticamente si no existen. Esto funciona mejor en Streamlit Cloud.

## 📦 PASO 3: Agregar Archivos al Repositorio

```powershell
# Agregar todos los archivos relevantes
git add .

# Ver qué se va a subir
git status

# Hacer commit inicial
git commit -m "Initial commit: Sistema de predicción y optimización de recursos para incendios forestales Chile"
```

## 🔗 PASO 4: Crear Repositorio en GitHub

1. **Ve a GitHub**: https://github.com/new
2. **Crea nuevo repositorio**:
   - Repository name: `incendios-chile` (o el nombre que prefieras)
   - Description: "Sistema de predicción y optimización de recursos para incendios forestales en Chile - Datos CONAF"
   - Visibilidad: **Public** (para Streamlit Cloud gratuito) o **Private**
   - **NO** marques "Initialize with README" (ya lo tenemos)
   - Click en **"Create repository"**

## 📤 PASO 5: Conectar y Subir a GitHub

```powershell
# Agregar remote (REEMPLAZA TU_USUARIO con tu usuario de GitHub)
git remote add origin https://github.com/TU_USUARIO/incendios-chile.git

# O si usas SSH:
# git remote add origin git@github.com:TU_USUARIO/incendios-chile.git

# Renombrar rama a main
git branch -M main

# Subir código
git push -u origin main
```

**Si GitHub pide autenticación:**
- Usa un Personal Access Token (PAT)
- O GitHub CLI: `gh auth login`

## 🌐 PASO 6: Deploy en Streamlit Cloud

1. **Ve a Streamlit Cloud**: https://share.streamlit.io
2. **Inicia sesión** con tu cuenta de GitHub
3. **Click en "New app"**
4. **Configura tu app**:
   - **Repository**: `TU_USUARIO/incendios-chile`
   - **Branch**: `main`
   - **Main file path**: `dashboard.py`
   - **Python version**: `3.9` (o la que prefieras)
5. **Click en "Deploy!"**

### ⚠️ IMPORTANTE: Datos en Streamlit Cloud

Si el dataset no está en el repo, Streamlit Cloud intentará procesarlo automáticamente cuando alguien acceda al dashboard. Esto puede tomar unos minutos la primera vez.

**Alternativa: Incluir dataset en el repo**

Si quieres incluir el dataset (1.3 MB aproximadamente):

```powershell
# 1. Temporalmente comentar en .gitignore:
#    data/processed/*
#    *.csv

# 2. Agregar el archivo
git add data/processed/conaf_datos_reales_completo.csv

# 3. Commit
git commit -m "Add processed CONAF dataset"

# 4. Push
git push

# 5. Descomentar .gitignore para futuros cambios
```

## 🔧 Verificación Post-Deploy

Después del deploy, verifica:

1. ✅ El dashboard carga correctamente
2. ✅ Los filtros funcionan
3. ✅ Los datos se visualizan
4. ✅ Los modelos pueden entrenarse
5. ✅ La optimización funciona

## 📋 Checklist Final

Antes de subir a GitHub, verifica:

- [ ] `README.md` está actualizado con tu información
- [ ] `.gitignore` excluye archivos grandes innecesarios
- [ ] `requirements.txt` tiene todas las dependencias
- [ ] `.streamlit/config.toml` existe
- [ ] `dashboard.py` es el archivo principal
- [ ] Código funciona localmente
- [ ] No hay datos sensibles en el código (API keys, etc.)

## 🆘 Solución de Problemas

### Error: "Failed to authenticate"

**Solución**: Usa un Personal Access Token
1. GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Genera nuevo token con permisos `repo`
3. Úsalo como contraseña cuando git lo pida

### Error: "Repository not found"

**Solución**: Verifica que:
- El repositorio existe en GitHub
- El nombre del usuario es correcto
- Tienes permisos en el repositorio

### Error en Streamlit Cloud: "FileNotFoundError"

**Solución**: 
- Verifica que el dataset esté en el repo O
- El dashboard intentará procesarlo automáticamente (puede tardar)

### Error: "ModuleNotFoundError" en Streamlit Cloud

**Solución**: 
- Verifica que todas las dependencias estén en `requirements.txt`
- Algunos paquetes pueden necesitar versiones específicas

## 📝 Comandos Útiles

```powershell
# Ver estado de git
git status

# Ver diferencias
git diff

# Ver historial
git log --oneline

# Crear nueva rama
git checkout -b feature/nueva-funcionalidad

# Volver a main
git checkout main

# Merge rama
git merge feature/nueva-funcionalidad

# Ver remotes
git remote -v

# Actualizar desde GitHub
git pull origin main

# Forzar push (¡cuidado!)
git push -f origin main
```

## 🎯 Próximos Pasos Después del Deploy

1. **Compartir tu app**: `https://TU-APP-NAME.streamlit.app`
2. **Documentar características** en el README
3. **Agregar badges** (opcional)
4. **Configurar CI/CD** (opcional)
5. **Agregar tests** (opcional)

## 📚 Recursos Adicionales

- [GitHub Docs](https://docs.github.com/)
- [Streamlit Cloud Docs](https://docs.streamlit.io/streamlit-community-cloud)
- [Git Handbook](https://guides.github.com/introduction/git-handbook/)

---

**¡Listo para subir!** 🚀

Si tienes dudas durante el proceso, consulta `setup_for_streamlit_cloud.md` para más detalles.

