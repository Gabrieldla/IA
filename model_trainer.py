"""
Módulo para entrenar el modelo XGBoost con optimización
"""
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import os


class ModelTrainer:
    """Entrena y optimiza el modelo XGBoost"""
    
    def __init__(self):
        self.model = None
        self.best_params = None
        self.metrics = {}
        self.training_log = []
        self.training_output = []  # Nuevo: outputs detallados
        
    def train_model(self, X, y, optimize=True):
        """Entrena el modelo con o sin optimización"""
        self._log("Dividiendo datos en entrenamiento y prueba (80-20)")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.training_output.append({
            'step': 'split',
            'name': 'División de datos',
            'output': f'Train: {len(X_train)} muestras | Test: {len(X_test)} muestras (80/20 split)'
        })
        
        if optimize:
            self._log("Iniciando optimización de hiperparámetros...")
            self.training_output.append({
                'step': 'optimization_start',
                'name': 'Búsqueda de hiperparámetros',
                'output': 'RandomizedSearchCV iniciado con 20 combinaciones y 5-fold cross-validation'
            })
            self.model = self._optimize_hyperparameters(X_train, y_train)
        else:
            self._log("Entrenando modelo con parámetros por defecto...")
            self.model = xgb.XGBRegressor(
                n_estimators=1000,
                learning_rate=0.05,
                max_depth=5,
                random_state=42
            )
            self.model.fit(X_train, y_train)
            self.training_output.append({
                'step': 'training',
                'name': 'Entrenamiento',
                'output': 'Modelo entrenado con parámetros por defecto'
            })
        
        # Evaluar modelo
        self._log("Evaluando modelo en conjunto de prueba...")
        y_pred = self.model.predict(X_test)
        self.metrics = self._evaluate_model(X_test, y_test)
        
        return self.model, self.metrics, X_test, y_test, y_pred
    
    def _optimize_hyperparameters(self, X_train, y_train):
        """Optimiza hiperparámetros usando RandomizedSearchCV"""
        param_dist = {
            'n_estimators': [500, 1000, 1500],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 1.0],
            'gamma': [0, 0.1],
            'reg_alpha': [0, 0.1, 1],
            'reg_lambda': [0.5, 1, 2]
        }
        
        xgb_model = xgb.XGBRegressor(random_state=42, eval_metric='rmse')
        
        random_search = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=param_dist,
            n_iter=20,
            cv=5,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1,
            random_state=42,
            verbose=0
        )
        
        self._log("Probando 20 combinaciones de hiperparámetros con validación cruzada...")
        random_search.fit(X_train, y_train)
        
        self.best_params = random_search.best_params_
        self._log(f"Mejores parámetros encontrados: {self.best_params}")
        
        # Guardar output de optimización
        params_str = "\n".join([f"  • {k}: {v}" for k, v in self.best_params.items()])
        self.training_output.append({
            'step': 'optimization_result',
            'name': 'Mejores hiperparámetros',
            'output': f'Optimización completada. Parámetros seleccionados:\n{params_str}'
        })
        
        self.training_output.append({
            'step': 'best_score',
            'name': 'Score de validación cruzada',
            'output': f'Mejor RMSE en CV: {-random_search.best_score_:.4f}'
        })
        
        return random_search.best_estimator_
    
    def _evaluate_model(self, X_test, y_test):
        """Evalúa el modelo y calcula métricas"""
        # Predecir
        y_pred = self.model.predict(X_test)
        
        # Convertir a escala original
        y_test_orig = np.expm1(y_test)
        y_pred_orig = np.expm1(y_pred)
        
        metrics = {
            'r2': r2_score(y_test_orig, y_pred_orig),
            'rmse': np.sqrt(mean_squared_error(y_test_orig, y_pred_orig)),
            'mae': mean_absolute_error(y_test_orig, y_pred_orig)
        }
        
        self._log(f"R² Score: {metrics['r2']:.4f}")
        self._log(f"RMSE: ${metrics['rmse']:,.2f}")
        self._log(f"MAE: ${metrics['mae']:,.2f}")
        
        # Guardar output de evaluación
        self.training_output.append({
            'step': 'evaluation',
            'name': 'Evaluación del modelo',
            'output': f'''Métricas en conjunto de prueba:
  • R² Score: {metrics['r2']:.4f} ({metrics['r2']*100:.2f}% de varianza explicada)
  • RMSE: ${metrics['rmse']:,.0f} (error cuadrático medio)
  • MAE: ${metrics['mae']:,.0f} (error absoluto promedio)'''
        })
        
        return metrics
    
    def save_model(self, model_dir, features_list, label_encoders):
        """Guarda el modelo y metadatos"""
        os.makedirs(model_dir, exist_ok=True)
        
        # Guardar modelo
        model_path = os.path.join(model_dir, 'modelo_casas.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Guardar encoders
        encoders_path = os.path.join(model_dir, 'label_encoders.pkl')
        with open(encoders_path, 'wb') as f:
            pickle.dump(label_encoders, f)
        
        # Guardar features
        features_path = os.path.join(model_dir, 'features.pkl')
        with open(features_path, 'wb') as f:
            pickle.dump(features_list, f)
        
        # Guardar métricas
        metrics_path = os.path.join(model_dir, 'metrics.pkl')
        with open(metrics_path, 'wb') as f:
            pickle.dump(self.metrics, f)
        
        self._log(f"Modelo guardado en {model_dir}")
        
        return {
            'model_path': model_path,
            'encoders_path': encoders_path,
            'features_path': features_path,
            'metrics_path': metrics_path
        }
    
    @staticmethod
    def load_model(model_dir):
        """Carga el modelo guardado"""
        model_path = os.path.join(model_dir, 'modelo_casas.pkl')
        encoders_path = os.path.join(model_dir, 'label_encoders.pkl')
        features_path = os.path.join(model_dir, 'features.pkl')
        metrics_path = os.path.join(model_dir, 'metrics.pkl')
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(encoders_path, 'rb') as f:
            encoders = pickle.load(f)
        
        with open(features_path, 'rb') as f:
            features = pickle.load(f)
        
        with open(metrics_path, 'rb') as f:
            metrics = pickle.load(f)
        
        return model, encoders, features, metrics
    
    def predict(self, features_dict):
        """Hace una predicción con el modelo"""
        if self.model is None:
            raise ValueError("Modelo no entrenado. Entrenar primero o cargar un modelo guardado.")
        
        # Convertir dict a DataFrame para predicción
        import pandas as pd
        X_pred = pd.DataFrame([features_dict])
        
        # Predecir (en escala log)
        y_pred_log = self.model.predict(X_pred)
        
        # Convertir a escala original
        y_pred = np.expm1(y_pred_log)[0]
        
        return y_pred
    
    def _log(self, message):
        """Registra un mensaje del entrenamiento"""
        self.training_log.append(message)
    
    def get_training_log(self):
        """Retorna el log del entrenamiento"""
        return self.training_log
    
    def get_feature_importance(self, feature_names):
        """Obtiene la importancia de las características"""
        if self.model is None:
            return []
        
        importance = self.model.feature_importances_
        feature_importance = [
            {'feature': name, 'importance': float(imp)}
            for name, imp in zip(feature_names, importance)
        ]
        # Ordenar por importancia
        feature_importance.sort(key=lambda x: x['importance'], reverse=True)
        
        return feature_importance
