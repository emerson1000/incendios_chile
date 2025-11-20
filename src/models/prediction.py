"""
Modelo de predicción de riesgo de incendio forestal
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle
import logging
from datetime import datetime

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)
import xgboost as xgb
import lightgbm as lgb
import shap

from config import MODELS_DIR, MODEL_CONFIG, FEATURES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FireRiskPredictor:
    """Clase para entrenar y usar modelos de predicción de riesgo de incendio"""
    
    def __init__(self, model_type: str = "xgboost", task: str = "classification"):
        """
        Args:
            model_type: Tipo de modelo ('xgboost', 'lightgbm', 'random_forest')
            task: Tipo de tarea ('classification' para ocurrencia, 'regression' para severidad)
        """
        self.model_type = model_type
        self.task = task
        self.model = None
        self.feature_names = None
        self.scaler = None
        self.feature_importance = None
        self.config = MODEL_CONFIG
        
        # Configurar modelo
        self._init_model()
    
    def _init_model(self):
        """Inicializa el modelo según el tipo seleccionado"""
        if self.task == "classification":
            if self.model_type == "xgboost":
                self.model = xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    eval_metric='logloss',
                    use_label_encoder=False
                )
            elif self.model_type == "lightgbm":
                self.model = lgb.LGBMClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    verbose=-1
                )
            elif self.model_type == "random_forest":
                self.model = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                )
        else:  # regression
            if self.model_type == "xgboost":
                self.model = xgb.XGBRegressor(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42
                )
            elif self.model_type == "lightgbm":
                self.model = lgb.LGBMRegressor(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    verbose=-1
                )
            elif self.model_type == "random_forest":
                self.model = RandomForestRegressor(
                    n_estimators=200,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                )
    
    def prepare_features(self, df: pd.DataFrame, target_col: str = "incendio_ocurrencia") -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepara features y target para el modelo
        
        Args:
            df: DataFrame con panel de datos
            target_col: Nombre de la columna objetivo
        
        Returns:
            X: DataFrame con features
            y: Serie con target
        """
        logger.info("Preparando features...")
        
        # Si ya tenemos feature_names guardados (modo predicción), usarlos
        # Si no, construir desde cero (modo entrenamiento)
        is_prediction = self.feature_names is not None and len(self.feature_names) > 0
        
        if is_prediction:
            logger.info(f"Modo predicción: usando {len(self.feature_names)} features predefinidas")
            # En modo predicción, debemos asegurar que las features coincidan exactamente
            
            # Crear dummies de comunas SI hay comunas en el entrenamiento
            comuna_features_training = [f for f in self.feature_names if f.startswith('comuna_')]
            
            if 'comuna' in df.columns and len(comuna_features_training) > 0:
                comunas_training = [f.replace('comuna_', '') for f in comuna_features_training]
                
                # Crear dummies para las comunas actuales en df
                comuna_dummies = pd.get_dummies(df['comuna'], prefix='comuna', sparse=True)
                
                # Asegurar que todas las comunas del entrenamiento estén presentes (poner 0 si no están)
                for col_name in comuna_features_training:
                    if col_name not in comuna_dummies.columns:
                        comuna_dummies[col_name] = 0
                
                # Eliminar comunas nuevas que no estaban en entrenamiento
                comuna_dummies = comuna_dummies[comuna_features_training]
                
                # Concatenar con df
                df = pd.concat([df, comuna_dummies], axis=1)
            
            # Asegurar que todas las features del entrenamiento estén presentes
            # Construir X usando exactamente las features del entrenamiento
            X = pd.DataFrame(index=df.index)
            
            for feat_name in self.feature_names:
                if feat_name in df.columns:
                    X[feat_name] = df[feat_name].values
                else:
                    # Si falta una feature, rellenar con 0
                    logger.warning(f"Feature '{feat_name}' no encontrada en datos de predicción, rellenando con 0")
                    X[feat_name] = 0
            
            # Asegurar el orden correcto (importante para algunos modelos)
            X = X[self.feature_names]
            
            # Llenar NaN con 0 (por seguridad)
            X = X.fillna(0)
            
            # Preparar y (puede ser dummy para predicción)
            y = df[target_col].copy() if target_col in df.columns else pd.Series([0] * len(df), index=df.index)
            
            # No eliminar filas en predicción (queremos predecir todas)
            # Solo asegurar que no haya NaN
            y = y.fillna(0)
            
        else:
            # Modo entrenamiento: construir features desde cero
            feature_cols = []
            
            # Features climáticas
            for feat in FEATURES["climatic"]:
                if feat in df.columns:
                    feature_cols.append(feat)
            
            # Features temporales
            temporal_features = [
                'mes', 'dia_semana', 'dia_anio', 'anio',
                'temporada_alta', 'fin_semana', 'mes_sin', 'mes_cos',
                'anio_normalizado'  # Agregar también esta si existe
            ]
            for feat in temporal_features:
                if feat in df.columns:
                    feature_cols.append(feat)
            
            # Features históricas (incluir todas las que agregamos en el dashboard)
            historical_features = [
                'incendios_7d', 'incendios_30d', 'area_quemada_365d', 
                'riesgo_base_comuna', 'incendios_total_hist', 'incendios_promedio_hist',
                'incendios_max_hist', 'incendios_std_hist', 'area_total_hist',
                'area_promedio_hist', 'incendios_anio_anterior', 'incendios_2_anios_antes',
                'area_anio_anterior', 'incendios_promedio_3_anios'
            ]
            for feat in historical_features:
                if feat in df.columns:
                    feature_cols.append(feat)
            
            # Eliminar columnas no numéricas y con muchos nulos
            numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [col for col in numeric_cols if df[col].notna().sum() > len(df) * 0.8]
            
            # Si hay comuna como categorical, crear dummies
            if 'comuna' in df.columns:
                comuna_dummies = pd.get_dummies(df['comuna'], prefix='comuna', sparse=True)
                feature_cols.extend(comuna_dummies.columns.tolist())
                df = pd.concat([df, comuna_dummies], axis=1)
            
            # Preparar X e y
            X = df[feature_cols].copy()
            y = df[target_col].copy()
            
            # Eliminar filas con valores faltantes
            mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[mask]
            y = y[mask]
            
            # Guardar feature_names para predicciones futuras
            self.feature_names = X.columns.tolist()
        
        logger.info(f"Features preparadas: {len(X.columns)} features, {len(X)} muestras")
        
        return X, y
    
    def train(self, X: pd.DataFrame, y: pd.Series, 
              validation_size: float = 0.2, 
              temporal_split: bool = True) -> Dict:
        """
        Entrena el modelo
        
        Args:
            X: Features
            y: Target
            validation_size: Proporción de datos para validación
            temporal_split: Si True, usa split temporal (no aleatorio)
        
        Returns:
            Dict con métricas de entrenamiento
        """
        logger.info(f"Entrenando modelo {self.model_type} para tarea {self.task}...")
        
        # Split train/test
        if temporal_split:
            # Split temporal: últimos datos para test
            split_idx = int(len(X) * (1 - validation_size))
            X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_size, random_state=42
            )
        
        logger.info(f"Train: {len(X_train)} muestras, Val: {len(X_val)} muestras")
        
        # Validaciones para clasificación
        if self.task == "classification":
            unique_classes_train = sorted(y_train.unique())
            unique_classes_val = sorted(y_val.unique())
            train_dist = y_train.value_counts().to_dict()
            val_dist = y_val.value_counts().to_dict()
            
            logger.info(f"Clases en entrenamiento: {unique_classes_train} (distribución: {train_dist})")
            logger.info(f"Clases en validación: {unique_classes_val} (distribución: {val_dist})")
            
            # Verificar que haya al menos 2 clases en entrenamiento
            if len(unique_classes_train) < 2:
                error_msg = (
                    f"❌ Error: No hay suficientes clases para clasificación.\n\n"
                    f"   • Solo se encontró la clase {unique_classes_train[0]}\n"
                    f"   • Distribución en entrenamiento: {train_dist}\n"
                    f"   • Se necesita al menos una muestra de clase 0 (sin incendio) y clase 1 (con incendio)\n\n"
                    f"   Solución: Ajusta los filtros para incluir:\n"
                    f"   • Más años de datos\n"
                    f"   • Más comunas\n"
                    f"   • Esto asegurará que haya comunas sin incendios en algunos años"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
        
        # Entrenar modelo con manejo de errores robusto
        # Cada tipo de modelo tiene diferentes parámetros en fit()
        try:
            if self.model_type == 'xgboost':
                # XGBoost acepta eval_set y verbose
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
            elif self.model_type == 'lightgbm':
                # LightGBM acepta eval_set, pero NO acepta verbose (se configura en constructor)
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)]
                )
            else:
                # Random Forest solo acepta X, y (no eval_set ni verbose)
                self.model.fit(X_train, y_train)
        except ValueError as e:
            error_msg = str(e)
            if "classes" in error_msg.lower() or "class" in error_msg.lower():
                unique_train = sorted(y_train.unique())
                unique_val = sorted(y_val.unique())
                train_dist = y_train.value_counts().to_dict()
                val_dist = y_val.value_counts().to_dict()
                
                enhanced_error = (
                    f"❌ Error al entrenar modelo de clasificación:\n\n"
                    f"   • Clases en entrenamiento: {unique_train} (distribución: {train_dist})\n"
                    f"   • Clases en validación: {unique_val} (distribución: {val_dist})\n"
                    f"   • Error original: {error_msg}\n\n"
                    f"   Problema: Todos los datos tienen la misma clase.\n\n"
                    f"   Solución: Ajusta los filtros para incluir más datos y asegurar que haya\n"
                    f"   tanto comunas con incendios (clase 1) como sin incendios (clase 0)."
                )
                logger.error(enhanced_error)
                raise ValueError(enhanced_error) from e
            raise
        
        # Predecir en validación
        if self.task == "classification":
            y_pred = self.model.predict(X_val)
            y_pred_proba = self.model.predict_proba(X_val)[:, 1] if hasattr(self.model, 'predict_proba') else None
            
            # Métricas
            metrics = {
                'accuracy': accuracy_score(y_val, y_pred),
                'precision': precision_score(y_val, y_pred, zero_division=0),
                'recall': recall_score(y_val, y_pred, zero_division=0),
                'f1': f1_score(y_val, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_val, y_pred_proba) if y_pred_proba is not None else None
            }
            
            logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"Precision: {metrics['precision']:.4f}")
            logger.info(f"Recall: {metrics['recall']:.4f}")
            logger.info(f"F1: {metrics['f1']:.4f}")
            if metrics['roc_auc']:
                logger.info(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        
        else:  # regression
            y_pred = self.model.predict(X_val)
            
            metrics = {
                'rmse': np.sqrt(mean_squared_error(y_val, y_pred)),
                'mae': mean_absolute_error(y_val, y_pred),
                'r2': r2_score(y_val, y_pred)
            }
            
            logger.info(f"RMSE: {metrics['rmse']:.4f}")
            logger.info(f"MAE: {metrics['mae']:.4f}")
            logger.info(f"R²: {metrics['r2']:.4f}")
        
        # Feature importance
        self._compute_feature_importance()
        
        metrics['feature_importance'] = self.feature_importance
        
        return metrics
    
    def _compute_feature_importance(self):
        """Computa importancia de features"""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'get_feature_importance'):
            importances = self.model.get_feature_importance()
        else:
            importances = None
        
        if importances is not None:
            self.feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
    
    def predict(self, X: pd.DataFrame, return_proba: bool = True) -> np.ndarray:
        """
        Predice riesgo de incendio
        
        Args:
            X: Features
            return_proba: Si True y es clasificación, retorna probabilidades
        
        Returns:
            Predicciones o probabilidades
        """
        if self.model is None:
            raise ValueError("Modelo no entrenado. Llama a train() primero.")
        
        if self.task == "classification" and return_proba:
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(X)[:, 1]
            else:
                return self.model.predict(X)
        else:
            return self.model.predict(X)
    
    def predict_risk_map(self, df: pd.DataFrame, 
                        fecha: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """
        Genera mapa de riesgo para una fecha específica
        
        Args:
            df: Panel de datos completo
            fecha: Fecha para predecir (si None, usa última fecha disponible)
        
        Returns:
            DataFrame con predicciones por comuna
        """
        if fecha is None:
            fecha = df['fecha'].max()
        
        # Filtrar datos para la fecha específica
        df_fecha = df[df['fecha'] == fecha].copy()
        
        if len(df_fecha) == 0:
            logger.warning(f"No hay datos para la fecha {fecha}")
            return pd.DataFrame()
        
        # Preparar features
        X, _ = self.prepare_features(df_fecha, target_col="incendio_ocurrencia")
        
        # Predecir
        risk_scores = self.predict(X, return_proba=True)
        
        # Crear DataFrame con resultados
        risk_map = pd.DataFrame({
            'comuna': df_fecha['comuna'].values,
            'fecha': fecha,
            'riesgo_probabilidad': risk_scores,
            'riesgo_categoria': pd.cut(
                risk_scores,
                bins=[0, 0.3, 0.6, 1.0],
                labels=['Bajo', 'Medio', 'Alto']
            )
        })
        
        return risk_map.sort_values('riesgo_probabilidad', ascending=False)
    
    def save_model(self, filename: Optional[str] = None):
        """Guarda el modelo entrenado"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"fire_risk_model_{self.model_type}_{self.task}_{timestamp}.pkl"
        
        filepath = MODELS_DIR / filename
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'task': self.task,
            'feature_importance': self.feature_importance
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Modelo guardado en {filepath}")
        return filepath
    
    def load_model(self, filepath: str):
        """Carga un modelo pre-entrenado"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        self.task = model_data['task']
        self.feature_importance = model_data.get('feature_importance')
        
        logger.info(f"Modelo cargado desde {filepath}")

