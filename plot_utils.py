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


def crear_grafico_importancia_caracteristicas(importancia_caracteristicas):
    """Crea gráfico de importancia de características"""
    if not importancia_caracteristicas:
        return None
    
    # Tomar top 15
    top_caracteristicas = importancia_caracteristicas[:15]
    
    caracteristicas = [f['feature'] for f in top_caracteristicas]
    importancia = [f['importance'] for f in top_caracteristicas]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=importancia, y=caracteristicas, ax=ax, palette='viridis')
    ax.set_title('Importancia de Características (Top 15)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Importancia')
    ax.set_ylabel('Característica')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return plot_to_base64(fig)


def crear_grafico_predicciones_vs_reales(y_test, y_pred):
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


def crear_grafico_residuos(y_pred, residuos):
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


def crear_grafico_nulos_rellenados(nulos_numericos, nulos_categoricos):
    """Crea gráfico de pastel mostrando nulos rellenados por tipo"""
    total_rellenados = nulos_numericos + nulos_categoricos
    
    if total_rellenados == 0:
        return None
    
    fig, ax = plt.subplots(figsize=(8, 6))
    categorias = ['Numéricos', 'Categóricos']
    valores = [nulos_numericos, nulos_categoricos]
    colores = ['#60a5fa', '#4ade80']
    
    wedges, texts, autotexts = ax.pie(valores, labels=categorias, autopct='%1.1f%%',
                                        colors=colores, startangle=90,
                                        textprops={'fontsize': 12, 'fontweight': 'bold'})
    ax.set_title(f'{total_rellenados:,} Nulos Rellenados por Tipo', fontsize=14, fontweight='bold')
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(14)
    
    plt.tight_layout()
    return plot_to_base64(fig)


def crear_grafico_comparacion_outliers(area_antes, precio_antes, area_despues, precio_despues, umbral):
    """Crea gráfico comparando datos antes y después de eliminar outliers"""
    fig, ejes = plt.subplots(1, 2, figsize=(16, 6))
    
    # ANTES
    ejes[0].scatter(area_antes, precio_antes, alpha=0.6, s=30, color='#ef4444', edgecolors='black', linewidth=0.5)
    ejes[0].axvline(umbral, color='#4ade80', linestyle='--', linewidth=2, label=f'Límite: {umbral:.0f}')
    ejes[0].set_xlabel('GrLivArea (pies²)', fontsize=11, fontweight='bold')
    ejes[0].set_ylabel('SalePrice ($)', fontsize=11, fontweight='bold')
    ejes[0].set_title(f'ANTES: {len(area_antes):,} casas (con outliers)', fontsize=13, fontweight='bold')
    ejes[0].grid(alpha=0.3)
    ejes[0].legend()
    
    # DESPUÉS
    ejes[1].scatter(area_despues, precio_despues, alpha=0.6, s=30, color='#4ade80', edgecolors='black', linewidth=0.5)
    ejes[1].set_xlabel('GrLivArea (pies²)', fontsize=11, fontweight='bold')
    ejes[1].set_ylabel('SalePrice ($)', fontsize=11, fontweight='bold')
    ejes[1].set_title(f'DESPUÉS: {len(area_despues):,} casas (sin outliers)', fontsize=13, fontweight='bold')
    ejes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    return plot_to_base64(fig)


def crear_grafico_transformacion_log(precio_original, precio_log):
    """Crea gráfico comparando distribución original vs logarítmica"""
    fig, ejes = plt.subplots(1, 2, figsize=(16, 6))
    
    # ORIGINAL
    ejes[0].hist(precio_original, bins=50, color='#60a5fa', alpha=0.7, edgecolor='black')
    ejes[0].axvline(precio_original.mean(), color='red', linestyle='--', linewidth=2, label=f'Media: ${precio_original.mean():,.0f}')
    ejes[0].set_xlabel('SalePrice ($)', fontsize=11, fontweight='bold')
    ejes[0].set_ylabel('Frecuencia', fontsize=11, fontweight='bold')
    ejes[0].set_title('Distribución ORIGINAL (Asimétrica)', fontsize=13, fontweight='bold')
    ejes[0].legend()
    ejes[0].grid(alpha=0.3)
    
    # LOG
    ejes[1].hist(precio_log, bins=50, color='#4ade80', alpha=0.7, edgecolor='black')
    ejes[1].axvline(precio_log.mean(), color='red', linestyle='--', linewidth=2, label=f'Media: {precio_log.mean():.2f}')
    ejes[1].set_xlabel('Log(SalePrice)', fontsize=11, fontweight='bold')
    ejes[1].set_ylabel('Frecuencia', fontsize=11, fontweight='bold')
    ejes[1].set_title('Distribución LOGARÍTMICA (Normalizada)', fontsize=13, fontweight='bold')
    ejes[1].legend()
    ejes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    return plot_to_base64(fig)


def crear_grafico_barras_correlacion(serie_correlacion):
    """Crea gráfico de barras horizontales de correlaciones - EXACTO como notebook"""
    if serie_correlacion is None or len(serie_correlacion) == 0:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 7))
    colores_barras = ['#4ade80' if x > 0.5 else '#60a5fa' if x > 0 else 'red' for x in serie_correlacion.values]
    ax.barh(serie_correlacion.index, serie_correlacion.values, color=colores_barras, alpha=0.8, edgecolor='black')
    ax.set_xlabel('Correlación con SalePrice', fontsize=12, fontweight='bold')
    ax.set_title('Top 15 Características por Correlación', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    for i, (idx, val) in enumerate(serie_correlacion.items()):
        ax.text(val + 0.01, i, f'{val:.3f}', va='center', fontweight='bold')
    
    plt.tight_layout()
    return plot_to_base64(fig)


def crear_grafico_top_nulos(df):
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


def crear_boxplot_saleprice(df):
    """Crea boxplot de SalePrice"""
    if 'SalePrice' not in df.columns:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(y=df['SalePrice'], ax=ax, color='#60a5fa')
    ax.set_title('Boxplot de SalePrice', fontsize=14, fontweight='bold')
    ax.set_ylabel('Precio ($)', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return plot_to_base64(fig)


def crear_grafico_cuartiles_saleprice(df):
    """Crea gráfico de distribución por cuartiles de SalePrice"""
    if 'SalePrice' not in df.columns:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    cuartiles = ['Q1\n(0-25%)', 'Q2\n(25-50%)', 'Q3\n(50-75%)', 'Q4\n(75-100%)']
    q1 = df['SalePrice'].quantile(0.25)
    q2 = df['SalePrice'].quantile(0.50)
    q3 = df['SalePrice'].quantile(0.75)
    
    conteos = [
        len(df[df['SalePrice'] <= q1]),
        len(df[(df['SalePrice'] > q1) & (df['SalePrice'] <= q2)]),
        len(df[(df['SalePrice'] > q2) & (df['SalePrice'] <= q3)]),
        len(df[df['SalePrice'] > q3])
    ]
    
    colores_cuartiles = ['#ef4444', '#f59e0b', '#4ade80', '#60a5fa']
    barras = ax.bar(cuartiles, conteos, color=colores_cuartiles, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_title('Distribución por Cuartiles de SalePrice', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cantidad de Casas', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Agregar valores encima de las barras
    for barra, conteo in zip(barras, conteos):
        altura = barra.get_height()
        ax.text(barra.get_x() + barra.get_width()/2., altura + 5,
               f'{conteo}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return plot_to_base64(fig)

