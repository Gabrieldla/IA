"""
Módulo para procesar y limpiar datos de precios de casas
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class HousePriceDataProcessor:
    """Procesa datos de casas para entrenamiento del modelo"""
    
    def __init__(self):
        self.label_encoders = {}
        self.steps_log = []
        self.steps_output = []  # Nuevo: guardar outputs de cada paso
        self.step_charts_data = {}  # Nuevo: datos para generar gráficos de cada paso
        self.initial_info = {}  # Info inicial del dataset
        
    def get_initial_info(self, df):
        """Obtiene información inicial del dataset - EXACTO como notebook"""
        import pandas as pd
        
        # Cantidad de nulos y sus tipos
        na_count = df.isna().sum().sort_values(ascending=False)
        na_count = na_count[na_count > 0]
        
        na_df = pd.DataFrame({
            'Nulos': na_count,
            'Tipo': df[na_count.index].dtypes
        })
        
        # Top 15 nulos con porcentaje
        na_count_full = df.isna().sum().sort_values(ascending=False)
        na_pct = (df.isna().mean()*100).sort_values(ascending=False)
        top15_nulls = pd.DataFrame({"nulos": na_count_full, "%": na_pct}).head(15)
        
        self.initial_info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'head': df.head(20),
            'na_df': na_df,
            'top15_nulls': top15_nulls,
            'total_nulls': int(df.isna().sum().sum())
        }
        
        return self.initial_info
        
    def process_data(self, df):
        """Procesa el dataset completo - SOLO LO ESENCIAL"""
        df_ready = df.copy()
        initial_rows = len(df_ready)
        initial_nulls = df_ready.isnull().sum().sum()
        
        # Paso 1: Rellenar TODOS los nulos primero (antes de filtrar columnas)
        nulls_numeric = 0
        nulls_categorical = 0
        
        for col in df_ready.columns:
            if df_ready[col].isnull().any():
                if df_ready[col].dtype in ['int64', 'float64']:
                    # Numéricos: rellenar con 0
                    nulls_numeric += df_ready[col].isnull().sum()
                    df_ready[col] = df_ready[col].fillna(0)
                else:
                    # Categóricos: rellenar con 'None'
                    nulls_categorical += df_ready[col].isnull().sum()
                    df_ready[col] = df_ready[col].fillna('None')
        
        total_nulls_filled = nulls_numeric + nulls_categorical
        
        # Guardar datos para gráfico de nulos rellenados
        self.step_charts_data['step1_nulls'] = {
            'nulls_numeric': nulls_numeric,
            'nulls_categorical': nulls_categorical
        }
        
        self.steps_output.append({
            'step': 1,
            'name': 'Rellenar valores faltantes',
            'output': f'''{total_nulls_filled} valores nulos rellenados:
  • Numéricos: {nulls_numeric} → 0
  • Categóricos: {nulls_categorical} → 'None'
Dataset limpio: 0 valores nulos restantes ✓'''
        })
        
        # Calcular correlaciones ANTES de filtrar (para mostrar en gráfico)
        numeric_df = df_ready.select_dtypes(include=[np.number])
        if 'SalePrice' in numeric_df.columns:
            correlations = numeric_df.corr()['SalePrice'].sort_values(ascending=False)
            # Guardar para gráfico
            self.step_charts_data['step2_correlation'] = correlations.drop('SalePrice').head(15)
        
        # Ahora filtrar solo las columnas que vamos a usar
        features_a_usar = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 
                          'TotalBsmtSF', '1stFlrSF', 'FullBath', 'YearBuilt', 
                          'YearRemodAdd', 'TotRmsAbvGrd', 'SalePrice']
        
        # Mantener solo las columnas necesarias
        columnas_existentes = [col for col in features_a_usar if col in df_ready.columns]
        df_ready = df_ready[columnas_existentes]
        
        self.steps_output.append({
            'step': 2,
            'name': 'Selección de características',
            'output': f'Dataset reducido de {len(df.columns)} a {len(columnas_existentes)} columnas (TOP 10 + SalePrice)\n\nCaracterísticas seleccionadas: {", ".join(features_a_usar[:-1])}'
        })
        
        # Paso 4: Remover outliers extremos
        rows_before = len(df_ready)
        if 'GrLivArea' in df_ready.columns and 'SalePrice' in df_ready.columns:
            # Guardar datos ANTES para gráfico
            grliv_before = df_ready['GrLivArea'].copy()
            price_before = df_ready['SalePrice'].copy()
            
            threshold = df_ready['GrLivArea'].quantile(0.995)
            df_ready = df_ready[df_ready['GrLivArea'] <= threshold]
            
            # Guardar datos DESPUÉS para gráfico
            grliv_after = df_ready['GrLivArea'].copy()
            price_after = df_ready['SalePrice'].copy()
            
            self.step_charts_data['step4_outliers'] = {
                'grliv_before': grliv_before,
                'price_before': price_before,
                'grliv_after': grliv_after,
                'price_after': price_after,
                'threshold': threshold
            }
        else:
            threshold = 0
        
        outliers_removed = rows_before - len(df_ready)
        
        self.steps_output.append({
            'step': 4,
            'name': 'Eliminar outliers',
            'output': f'{outliers_removed} outliers extremos eliminados (áreas >99.5 percentil = {threshold:.0f} pies²)'
        })
        
        # Paso 5: Transformar variable objetivo
        if "SalePrice" in df_ready.columns:
            # Guardar datos ANTES para gráfico
            price_original = df_ready["SalePrice"].copy()
            
            df_ready["SalePrice_log"] = np.log1p(df_ready["SalePrice"])
            
            # Guardar datos para gráfico
            self.step_charts_data['step5_log'] = {
                'price_original': price_original,
                'price_log': df_ready["SalePrice_log"].copy()
            }
            
            self.steps_output.append({
                'step': 5,
                'name': 'Transformación logarítmica',
                'output': f'SalePrice transformado a escala logarítmica para normalidad\n(rango: {df_ready["SalePrice_log"].min():.2f} - {df_ready["SalePrice_log"].max():.2f})'
            })
        
        return df_ready
    
    def encode_categorical(self, df):
        """Codifica variables categóricas - NO NECESARIO, solo usamos numéricas"""
        return df.copy()
    
    def prepare_features(self, df_model):
        """Prepara las características para el modelo - Solo las más importantes"""
        # TOP 10 características más correlacionadas con el precio
        features_seleccionadas = [
            'OverallQual',    # Calidad general (correlación ~0.79)
            'GrLivArea',      # Área habitable (correlación ~0.71)
            'GarageCars',     # Capacidad garaje (correlación ~0.64)
            'GarageArea',     # Área garaje (correlación ~0.62)
            'TotalBsmtSF',    # Área sótano (correlación ~0.61)
            '1stFlrSF',       # Área primer piso (correlación ~0.61)
            'FullBath',       # Baños completos (correlación ~0.56)
            'YearBuilt',      # Año construcción (correlación ~0.52)
            'YearRemodAdd',   # Año remodelación (correlación ~0.51)
            'TotRmsAbvGrd',   # Total habitaciones (correlación ~0.53)
        ]
        
        features_disponibles = [f for f in features_seleccionadas if f in df_model.columns]
        self._log_step(f"Seleccionadas {len(features_disponibles)} características TOP (máxima correlación con precio)")
        
        X = df_model[features_disponibles]
        y = df_model["SalePrice_log"] if "SalePrice_log" in df_model.columns else None
        
        return X, y, features_disponibles
    
    def _log_step(self, message):
        """Registra un paso del procesamiento"""
        self.steps_log.append(message)
    
    def get_steps_log(self):
        """Retorna el log de pasos"""
        return self.steps_log
    
    def get_data_stats(self, df_original, df_processed):
        """Obtiene estadísticas de los datos"""
        stats = {
            'original_shape': df_original.shape,
            'processed_shape': df_processed.shape,
            'original_nulls': int(df_original.isna().sum().sum()),
            'processed_nulls': int(df_processed.isna().sum().sum()),
            'rows_removed': df_original.shape[0] - df_processed.shape[0]
        }
        return stats
