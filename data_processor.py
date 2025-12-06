"""
Módulo para procesar y limpiar datos de precios de casas
"""
import numpy as np
import pandas as pd


class HousePriceDataProcessor:
    """Procesa datos de casas para entrenamiento del modelo"""
    
    def __init__(self):
        self.steps_output = []  # Guardar outputs de cada paso para mostrar en UI
        self.step_charts_data = {}  # Datos para generar gráficos de cada paso
        self.initial_info = {}  # Info inicial del dataset
        
    def obtener_info_inicial(self, df):
        """Obtiene información inicial del dataset - EXACTO como notebook"""
        import pandas as pd
        
        # Cantidad de nulos y sus tipos
        conteo_nulos = df.isna().sum().sort_values(ascending=False)
        conteo_nulos = conteo_nulos[conteo_nulos > 0]
        
        df_nulos = pd.DataFrame({
            'Nulos': conteo_nulos,
            'Tipo': df[conteo_nulos.index].dtypes
        })
        
        # Top 15 nulos con porcentaje
        conteo_nulos_completo = df.isna().sum().sort_values(ascending=False)
        porcentaje_nulos = (df.isna().mean()*100).sort_values(ascending=False)
        top15_nulos = pd.DataFrame({"nulos": conteo_nulos_completo, "%": porcentaje_nulos}).head(15)
        
        self.initial_info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'head': df.head(20),
            'na_df': df_nulos,
            'top15_nulls': top15_nulos,
            'total_nulls': int(df.isna().sum().sum())
        }
        
        return self.initial_info
        
    def procesar_datos(self, df):
        """Procesa el dataset completo - SOLO LO ESENCIAL"""
        df_listo = df.copy()
        filas_iniciales = len(df_listo)
        nulos_iniciales = df_listo.isnull().sum().sum()
        
        # Paso 1: Rellenar TODOS los nulos primero (antes de filtrar columnas)
        nulos_numericos = 0
        nulos_categoricos = 0
        
        for col in df_listo.columns:
            if df_listo[col].isnull().any():
                if df_listo[col].dtype in ['int64', 'float64']:
                    # Numéricos: rellenar con 0
                    nulos_numericos += df_listo[col].isnull().sum()
                    df_listo[col] = df_listo[col].fillna(0)
                else:
                    # Categóricos: rellenar con 'None'
                    nulos_categoricos += df_listo[col].isnull().sum()
                    df_listo[col] = df_listo[col].fillna('None')
        
        total_nulos_rellenados = nulos_numericos + nulos_categoricos
        
        # Guardar datos para gráfico de nulos rellenados
        self.step_charts_data['step1_nulls'] = {
            'nulos_numericos': nulos_numericos,
            'nulos_categoricos': nulos_categoricos
        }
        
        self.steps_output.append({
            'step': 1,
            'name': 'PASO 1: Rellenar valores faltantes',
            'output': f'''{total_nulos_rellenados} valores nulos rellenados:
  • Numéricos: {nulos_numericos} → 0
  • Categóricos: {nulos_categoricos} → 'None'
Dataset limpio: 0 valores nulos restantes ✓'''
        })
        
        # Calcular correlaciones ANTES de filtrar (para mostrar en gráfico)
        df_numerico = df_listo.select_dtypes(include=[np.number])
        if 'SalePrice' in df_numerico.columns:
            correlaciones = df_numerico.corr()['SalePrice'].sort_values(ascending=False)
            # Guardar para gráfico
            self.step_charts_data['step2_correlation'] = correlaciones.drop('SalePrice').head(15)
            
            self.steps_output.append({
                'step': 2,
                'name': 'PASO 2: Análisis de correlaciones',
                'output': f'Correlaciones calculadas para todas las variables numéricas\nTop 3 características con mayor correlación:\n  • {correlaciones.index[1]}: {correlaciones.iloc[1]:.3f}\n  • {correlaciones.index[2]}: {correlaciones.iloc[2]:.3f}\n  • {correlaciones.index[3]}: {correlaciones.iloc[3]:.3f}'
            })
        
        # Ahora filtrar solo las columnas que vamos a usar
        top_10_caracteristicas = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 
                                  'TotalBsmtSF', '1stFlrSF', 'FullBath', 'YearBuilt', 
                                  'YearRemodAdd', 'TotRmsAbvGrd']
        
        # Mantener solo las columnas necesarias
        columnas_existentes = [col for col in top_10_caracteristicas if col in df_listo.columns]
        df_listo = df_listo[columnas_existentes + ['SalePrice']]
        
        self.steps_output.append({
            'step': 3,
            'name': 'PASO 3: Selección de características',
            'output': f'Dataset reducido de {len(df.columns)} a {len(df_listo.columns)} columnas (TOP 10 + SalePrice)\n\nCaracterísticas seleccionadas: {", ".join(top_10_caracteristicas)}'
        })
        
        # Paso 4: Remover outliers extremos
        filas_antes = len(df_listo)
        if 'GrLivArea' in df_listo.columns and 'SalePrice' in df_listo.columns:
            # Guardar datos ANTES para gráfico
            area_antes = df_listo['GrLivArea'].copy()
            precio_antes = df_listo['SalePrice'].copy()
            
            umbral = df_listo['GrLivArea'].quantile(0.995)
            df_listo = df_listo[df_listo['GrLivArea'] <= umbral]
            df_listo = df_listo.reset_index(drop=True)
            
            # Guardar datos DESPUÉS para gráfico
            area_despues = df_listo['GrLivArea'].copy()
            precio_despues = df_listo['SalePrice'].copy()
            
            self.step_charts_data['step4_outliers'] = {
                'area_antes': area_antes,
                'precio_antes': precio_antes,
                'area_despues': area_despues,
                'precio_despues': precio_despues,
                'umbral': umbral
            }
        else:
            umbral = 0
        
        outliers_eliminados = filas_antes - len(df_listo)
        
        self.steps_output.append({
            'step': 4,
            'name': 'PASO 4: Remover outliers extremos',
            'output': f'{outliers_eliminados} outliers eliminados (áreas >{umbral:.0f} pies²)'
        })
        
        # Paso 5: Transformar variable objetivo
        if "SalePrice" in df_listo.columns:
            # Guardar datos ANTES para gráfico
            precio_original = df_listo["SalePrice"].copy()
            
            df_listo["SalePrice_log"] = np.log1p(df_listo["SalePrice"])
            
            # Guardar datos para gráfico
            self.step_charts_data['step5_log'] = {
                'precio_original': precio_original,
                'precio_log': df_listo["SalePrice_log"].copy()
            }
            
            self.steps_output.append({
                'step': 5,
                'name': 'PASO 5: Transformación logarítmica',
                'output': f'SalePrice transformado a escala logarítmica\n  Rango original: ${precio_original.min():,.0f} - ${precio_original.max():,.0f}\n  Rango log: {df_listo["SalePrice_log"].min():.2f} - {df_listo["SalePrice_log"].max():.2f}'
            })
        
        return df_listo
    
    def preparar_caracteristicas(self, df_modelo):
        """Prepara las características para el modelo - Solo las más importantes"""
        # TOP 10 características más correlacionadas con el precio
        top_10_caracteristicas = [
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
        
        caracteristicas_disponibles = [c for c in top_10_caracteristicas if c in df_modelo.columns]
        
        X = df_modelo[caracteristicas_disponibles]
        y = df_modelo["SalePrice_log"] if "SalePrice_log" in df_modelo.columns else None
        
        return X, y, caracteristicas_disponibles
