"""
Utilidades para crear gráficos con matplotlib
"""
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI para servidor web
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import pandas as pd
import numpy as np


def plot_to_base64(fig):
    """Convierte una figura de matplotlib a base64 para HTML"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{img_base64}"


def create_nulls_chart(df):
    """Crea gráfico de valores nulos"""
    na_count = df.isna().sum().sort_values(ascending=False)
    na_count = na_count[na_count > 0].head(10)
    
    if len(na_count) == 0:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=na_count.values, y=na_count.index, ax=ax, palette='viridis')
    ax.set_title('Top 10 Columnas con Valores Nulos', fontsize=14, fontweight='bold')
    ax.set_xlabel('Cantidad de Nulos')
    ax.set_ylabel('Columna')
    ax.grid(axis='x', alpha=0.3)
    
    return plot_to_base64(fig)


def create_price_distribution(df):
    """Crea gráfico de distribución de precios"""
    if 'SalePrice' not in df.columns:
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Precio original
    sns.histplot(df['SalePrice'], bins=50, kde=True, ax=axes[0], color='steelblue')
    axes[0].set_title('Distribución de SalePrice', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Precio ($)')
    axes[0].set_ylabel('Frecuencia')
    
    # Precio transformado (log)
    if 'SalePrice_log' in df.columns:
        sns.histplot(df['SalePrice_log'], bins=50, kde=True, ax=axes[1], color='coral')
        axes[1].set_title('Distribución de SalePrice (Log)', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Log(Precio)')
        axes[1].set_ylabel('Frecuencia')
    
    plt.tight_layout()
    return plot_to_base64(fig)


def create_correlation_heatmap(df, target='SalePrice'):
    """Crea heatmap de correlaciones con el precio"""
    if target not in df.columns:
        return None
    
    # Seleccionar solo columnas numéricas
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Calcular correlación con target
    corr = numeric_df.corr()[target].sort_values(ascending=False).head(15)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['green' if x > 0 else 'red' for x in corr.values]
    ax.barh(corr.index, corr.values, color=colors, alpha=0.7)
    ax.set_xlabel('Correlación con SalePrice')
    ax.set_title('Top 15 Características más Correlacionadas', fontsize=12, fontweight='bold')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return plot_to_base64(fig)


def create_feature_importance_chart(feature_importance):
    """Crea gráfico de importancia de características"""
    if not feature_importance:
        return None
    
    # Tomar top 15
    top_features = feature_importance[:15]
    
    features = [f['feature'] for f in top_features]
    importance = [f['importance'] for f in top_features]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=importance, y=features, ax=ax, palette='viridis')
    ax.set_title('Importancia de Características (Top 15)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Importancia')
    ax.set_ylabel('Característica')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return plot_to_base64(fig)


def create_predictions_vs_real_chart(y_test, y_pred):
    """Crea gráfico de predicciones vs valores reales - EXACTO como notebook"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Convertir a escala original
    y_test_orig = np.expm1(y_test)
    y_pred_orig = np.expm1(y_pred)
    
    # Scatter plot
    ax.scatter(y_test_orig, y_pred_orig, alpha=0.6, s=50, color='#60a5fa', edgecolors='black', linewidth=0.5)
    
    # Línea de predicción perfecta
    valor_min = min(y_test_orig.min(), y_pred_orig.min())
    valor_max = max(y_test_orig.max(), y_pred_orig.max())
    ax.plot([valor_min, valor_max], [valor_min, valor_max], 'r--', lw=2, label='Predicción perfecta')
    
    ax.set_xlabel('Precio Real ($)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precio Predicho ($)', fontsize=12, fontweight='bold')
    ax.set_title('Gráfico: Predicciones vs Valores Reales', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return plot_to_base64(fig)


def create_residuals_chart(y_pred, residuos):
    """Crea gráficos de residuos (scatter + histogram) - EXACTO como notebook"""
    fig, ejes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Convertir a escala original
    y_pred_orig = np.expm1(y_pred)
    residuos_orig = np.expm1(residuos + np.log1p(y_pred_orig)) - y_pred_orig
    
    # Gráfico 1: Scatter de residuos
    ejes[0].scatter(y_pred_orig, residuos_orig, alpha=0.6, s=50, color='#4ade80', edgecolors='black', linewidth=0.5)
    ejes[0].axhline(0, color='red', linestyle='--', linewidth=2)
    ejes[0].set_xlabel('Predicciones', fontsize=12, fontweight='bold')
    ejes[0].set_ylabel('Residuos', fontsize=12, fontweight='bold')
    ejes[0].set_title('Residuos vs Predicciones', fontsize=13, fontweight='bold')
    ejes[0].grid(alpha=0.3)
    
    # Gráfico 2: Histograma de residuos
    ejes[1].hist(residuos_orig, bins=50, color='#60a5fa', alpha=0.7, edgecolor='black')
    ejes[1].axvline(0, color='red', linestyle='--', linewidth=2, label='Media')
    ejes[1].set_xlabel('Residuos', fontsize=12, fontweight='bold')
    ejes[1].set_ylabel('Frecuencia', fontsize=12, fontweight='bold')
    ejes[1].set_title('Distribución de Residuos', fontsize=13, fontweight='bold')
    ejes[1].legend()
    ejes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    return plot_to_base64(fig)


def create_prediction_comparison_chart(features_dict, df_processed):
    """Crea gráfico comparando características de la casa vs promedio del dataset"""
    if df_processed is None or df_processed.empty:
        return None
    
    # Seleccionar características numéricas principales
    main_features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', '1stFlrSF', 'FullBath']
    
    # Filtrar solo las que existen en ambos
    available_features = [f for f in main_features if f in features_dict and f in df_processed.columns]
    
    if not available_features:
        return None
    
    # Calcular promedios del dataset
    dataset_means = df_processed[available_features].mean()
    input_values = [features_dict.get(f, 0) for f in available_features]
    
    # Normalizar valores (cada característica en escala 0-100)
    normalization = {
        'OverallQual': 10,
        'GrLivArea': 5000,
        'GarageCars': 4,
        'TotalBsmtSF': 3000,
        '1stFlrSF': 3000,
        'FullBath': 4,
        'YearBuilt': 2024,
        'YearRemodAdd': 2024,
        'TotRmsAbvGrd': 15
    }
    
    input_normalized = [(input_values[i] / normalization.get(f, 1)) * 100 for i, f in enumerate(available_features)]
    dataset_normalized = [(dataset_means[f] / normalization.get(f, 1)) * 100 for f in available_features]
    
    # Crear gráfico
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(available_features))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, input_normalized, width, label='Tu Casa', color='#4ade80', alpha=0.9, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, dataset_normalized, width, label='Promedio Dataset', color='#60a5fa', alpha=0.9, edgecolor='black', linewidth=1.5)
    
    # Agregar valores reales encima de las barras
    for i, (bar1, bar2, val1, val2) in enumerate(zip(bars1, bars2, input_values, dataset_means)):
        # Valor de tu casa
        ax.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 2,
                f'{val1:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#006239')
        # Valor promedio
        ax.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 2,
                f'{val2:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#1e40af')
    
    ax.set_xlabel('Características', fontsize=12, fontweight='bold')
    ax.set_ylabel('Escala Normalizada (0-100)', fontsize=12, fontweight='bold')
    ax.set_title('Comparación Normalizada: Tu Casa vs Promedio del Mercado', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(available_features, rotation=0, ha='center', fontsize=11)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 120)
    
    # Agregar línea de referencia en 50%
    ax.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='50% de escala')
    
    plt.tight_layout()
    return plot_to_base64(fig)


def create_price_position_chart(predicted_price, df_processed):
    """Crea gráfico mostrando la posición del precio predicho en la distribución"""
    if df_processed is None or 'SalePrice' not in df_processed.columns:
        return None
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Histograma de precios del dataset
    ax.hist(df_processed['SalePrice'], bins=50, color='#60a5fa', alpha=0.6, edgecolor='black')
    
    # Línea vertical para la predicción
    ax.axvline(predicted_price, color='#4ade80', linestyle='--', linewidth=3, label=f'Predicción: ${predicted_price:,.0f}')
    
    # Líneas para cuartiles
    q25 = df_processed['SalePrice'].quantile(0.25)
    q50 = df_processed['SalePrice'].quantile(0.50)
    q75 = df_processed['SalePrice'].quantile(0.75)
    
    ax.axvline(q25, color='gray', linestyle=':', linewidth=1.5, alpha=0.5, label=f'Q1: ${q25:,.0f}')
    ax.axvline(q50, color='gray', linestyle=':', linewidth=1.5, alpha=0.5, label=f'Mediana: ${q50:,.0f}')
    ax.axvline(q75, color='gray', linestyle=':', linewidth=1.5, alpha=0.5, label=f'Q3: ${q75:,.0f}')
    
    ax.set_xlabel('Precio ($)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frecuencia', fontsize=11, fontweight='bold')
    ax.set_title('Posición de tu Predicción en el Mercado', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return plot_to_base64(fig)


def create_feature_contribution_chart(features_dict, feature_importance):
    """Crea gráfico mostrando la contribución de cada característica en esta predicción"""
    if not feature_importance or not features_dict:
        return None
    
    # Filtrar solo características con importancia > 0.01
    significant_features = [f for f in feature_importance if f['importance'] > 0.01][:8]
    
    if not significant_features:
        return None
    
    feature_names = [f['feature'] for f in significant_features]
    importance_values = [f['importance'] for f in significant_features]
    input_values = [features_dict.get(f['feature'], 0) for f in significant_features]
    
    # Crear gráfico con importancia real del modelo
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Usar directamente la importancia del modelo (más confiable)
    colors = ['#4ade80' if imp > np.mean(importance_values) else '#60a5fa' for imp in importance_values]
    bars = ax.barh(feature_names, importance_values, color=colors, alpha=0.8)
    
    ax.set_xlabel('Importancia en el Modelo (0-1)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Característica', fontsize=11, fontweight='bold')
    ax.set_title('Importancia de Características Según el Modelo XGBoost', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Agregar valores de entrada como anotaciones en el lado derecho
    max_importance = max(importance_values)
    for i, (bar, val, imp) in enumerate(zip(bars, input_values, importance_values)):
        # Mostrar valor ingresado
        ax.text(max_importance * 1.05, bar.get_y() + bar.get_height()/2, 
                f'Tu valor: {val:.0f}', 
                ha='left', va='center', fontsize=9, color='#4ade80', fontweight='bold')
        # Mostrar porcentaje de importancia
        ax.text(bar.get_width() * 0.5, bar.get_y() + bar.get_height()/2, 
                f'{imp*100:.1f}%', 
                ha='center', va='center', fontsize=9, color='white', fontweight='bold')
    
    plt.tight_layout()
    return plot_to_base64(fig)


def create_nulls_filled_chart(nulls_numeric, nulls_categorical):
    """Crea gráfico de pastel mostrando nulos rellenados por tipo"""
    total_filled = nulls_numeric + nulls_categorical
    
    if total_filled == 0:
        return None
    
    fig, ax = plt.subplots(figsize=(8, 6))
    categories = ['Numéricos', 'Categóricos']
    values = [nulls_numeric, nulls_categorical]
    colors = ['#60a5fa', '#4ade80']
    
    wedges, texts, autotexts = ax.pie(values, labels=categories, autopct='%1.1f%%',
                                        colors=colors, startangle=90,
                                        textprops={'fontsize': 12, 'fontweight': 'bold'})
    ax.set_title(f'{total_filled:,} Nulos Rellenados por Tipo', fontsize=14, fontweight='bold')
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(14)
    
    plt.tight_layout()
    return plot_to_base64(fig)


def create_outliers_comparison_chart(grliv_before, price_before, grliv_after, price_after, threshold):
    """Crea gráfico comparando datos antes y después de eliminar outliers"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # ANTES
    axes[0].scatter(grliv_before, price_before, alpha=0.6, s=30, color='#ef4444', edgecolors='black', linewidth=0.5)
    axes[0].axvline(threshold, color='#4ade80', linestyle='--', linewidth=2, label=f'Límite: {threshold:.0f}')
    axes[0].set_xlabel('GrLivArea (pies²)', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('SalePrice ($)', fontsize=11, fontweight='bold')
    axes[0].set_title(f'ANTES: {len(grliv_before):,} casas (con outliers)', fontsize=13, fontweight='bold')
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    
    # DESPUÉS
    axes[1].scatter(grliv_after, price_after, alpha=0.6, s=30, color='#4ade80', edgecolors='black', linewidth=0.5)
    axes[1].set_xlabel('GrLivArea (pies²)', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('SalePrice ($)', fontsize=11, fontweight='bold')
    axes[1].set_title(f'DESPUÉS: {len(grliv_after):,} casas (sin outliers)', fontsize=13, fontweight='bold')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    return plot_to_base64(fig)


def create_log_transformation_chart(price_original, price_log):
    """Crea gráfico comparando distribución original vs logarítmica"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # ORIGINAL
    axes[0].hist(price_original, bins=50, color='#60a5fa', alpha=0.7, edgecolor='black')
    axes[0].axvline(price_original.mean(), color='red', linestyle='--', linewidth=2, label=f'Media: ${price_original.mean():,.0f}')
    axes[0].set_xlabel('SalePrice ($)', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Frecuencia', fontsize=11, fontweight='bold')
    axes[0].set_title('Distribución ORIGINAL (Asimétrica)', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # LOG
    axes[1].hist(price_log, bins=50, color='#4ade80', alpha=0.7, edgecolor='black')
    axes[1].axvline(price_log.mean(), color='red', linestyle='--', linewidth=2, label=f'Media: {price_log.mean():.2f}')
    axes[1].set_xlabel('Log(SalePrice)', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Frecuencia', fontsize=11, fontweight='bold')
    axes[1].set_title('Distribución LOGARÍTMICA (Normalizada)', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    return plot_to_base64(fig)


def create_correlation_bar_chart(correlation_series):
    """Crea gráfico de barras horizontales de correlaciones - EXACTO como notebook"""
    if correlation_series is None or len(correlation_series) == 0:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 7))
    colors_bar = ['#4ade80' if x > 0.5 else '#60a5fa' if x > 0 else 'red' for x in correlation_series.values]
    ax.barh(correlation_series.index, correlation_series.values, color=colors_bar, alpha=0.8, edgecolor='black')
    ax.set_xlabel('Correlación con SalePrice', fontsize=12, fontweight='bold')
    ax.set_title('Top 15 Features por Correlación', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    for i, (idx, val) in enumerate(correlation_series.items()):
        ax.text(val + 0.01, i, f'{val:.3f}', va='center', fontweight='bold')
    
    plt.tight_layout()
    return plot_to_base64(fig)


def create_top_nulls_chart(df):
    """Crea gráfico de Top 15 nulos - EXACTO como notebook"""
    na_count = df.isna().sum().sort_values(ascending=False)
    na_count_top = na_count.head(15)
    
    if len(na_count_top) == 0:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=na_count_top.values, y=na_count_top.index, palette='Reds_r', ax=ax)
    ax.set_title("Top 15 columnas con más nulos", fontsize=14, fontweight='bold')
    ax.set_xlabel("Cantidad de nulos")
    plt.tight_layout()
    
    return plot_to_base64(fig)
