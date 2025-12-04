"""
House Price Prediction - FastHTML Application
Simple web interface siguiendo exactamente el flujo de la notebook
"""
from fasthtml.common import *
from starlette.responses import RedirectResponse
import pandas as pd
import pickle
from pathlib import Path

from data_processor import HousePriceDataProcessor
from model_trainer import ModelTrainer
from plot_utils import *

# Configuración
app, rt = fast_app(live=True, pico=False)
UPLOAD_DIR = Path("uploads")
MODELS_DIR = Path("models")
UPLOAD_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Estado global
app_state = {
    'df_original': None,
    'df_processed': None,
    'processor': None,
    'trainer': None,
    'model': None,
    'features': None,
    'metrics': None
}

# Estilos CSS
CSS = """
* { margin: 0; padding: 0; box-sizing: border-box; }
body { 
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; 
    background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 50%, #0f0f0f 100%);
    background-attachment: fixed;
    color: #e2e8f0; 
    line-height: 1.6;
    min-height: 100vh;
}
.container { 
    max-width: 100%;
    width: 100%;
    margin: 0;
    padding: 0;
}
.header { 
    background: linear-gradient(135deg, #006239 0%, #004d2d 100%);
    padding: 3rem 2rem; 
    text-align: center; 
    margin: 0;
    box-shadow: 0 8px 32px rgba(0, 98, 57, 0.4);
    border-bottom: 3px solid #4ade80;
}
.header h1 { 
    font-size: 3rem; 
    color: #ffffff; 
    font-weight: 800; 
    margin-bottom: 0.5rem;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
}
.header p { 
    color: #4ade80; 
    font-size: 1.2rem; 
    font-weight: 500;
    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2);
}

.section { 
    background: rgba(23, 23, 23, 0.8);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(74, 222, 128, 0.2); 
    padding: 3rem 2rem; 
    margin: 0;
    min-height: calc(100vh - 220px);
}
.section-title { 
    font-size: 2.25rem; 
    color: #ffffff; 
    margin-bottom: 0.75rem; 
    font-weight: 700;
    background: linear-gradient(135deg, #4ade80 0%, #22c55e 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.section-subtitle { 
    color: #9ca3af; 
    margin-bottom: 2.5rem;
    font-size: 1.1rem;
}

.form-group { margin-bottom: 1.5rem; }
label { 
    display: block; 
    font-weight: 600; 
    color: #e2e8f0; 
    margin-bottom: 0.5rem;
    font-size: 0.95rem;
}
input[type="file"], input[type="number"] {
    width: 100%;
    padding: 1rem;
    border: 2px solid rgba(74, 222, 128, 0.3);
    background: rgba(15, 15, 15, 0.8);
    color: #e2e8f0;
    border-radius: 8px;
    font-size: 1rem;
    transition: all 0.3s ease;
}
input:focus {
    outline: none;
    border-color: #4ade80;
    background: rgba(26, 26, 26, 0.9);
    box-shadow: 0 0 0 3px rgba(74, 222, 128, 0.1);
}

.btn {
    background: linear-gradient(135deg, #006239 0%, #004d2d 100%);
    color: #ffffff;
    border: none;
    padding: 1rem 2.5rem;
    font-size: 0.9rem;
    font-weight: 700;
    border-radius: 8px;
    cursor: pointer;
    width: 100%;
    max-width: 350px;
    text-transform: uppercase;
    transition: all 0.3s ease;
    display: block;
    margin: 0 auto;
    box-shadow: 0 4px 15px rgba(0, 98, 57, 0.3);
    letter-spacing: 0.5px;
}
.btn:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(74, 222, 128, 0.5);
    background: linear-gradient(135deg, #4ade80 0%, #22c55e 100%);
    color: #0a0a0a;
}
.btn-block {
    max-width: 100%;
}

.step-box {
    background: rgba(26, 26, 26, 0.6);
    border-left: 4px solid #4ade80;
    border-radius: 12px;
    padding: 2.5rem;
    margin-bottom: 2rem;
    backdrop-filter: blur(5px);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
}
.step-title {
    color: #ffffff;
    font-size: 1.5rem;
    margin-bottom: 1.25rem;
    border-bottom: 2px solid rgba(74, 222, 128, 0.5);
    padding-bottom: 0.75rem;
    font-weight: 700;
}
.step-output {
    background: rgba(15, 15, 15, 0.8);
    padding: 1.5rem;
    border-radius: 8px;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(74, 222, 128, 0.2);
}
.step-output pre {
    color: #4ade80;
    font-size: 1rem;
    margin: 0;
    white-space: pre-wrap;
    line-height: 1.6;
}

.chart-box {
    background: rgba(15, 15, 15, 0.6);
    padding: 2rem;
    border-radius: 12px;
    margin-bottom: 2rem;
    max-height: 600px;
    overflow-y: auto;
    border: 1px solid rgba(74, 222, 128, 0.2);
}
.chart-box img {
    max-width: 100%;
    height: auto;
    border-radius: 8px;
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 2rem;
    margin: 2.5rem 0;
}
.metric-card {
    background: rgba(26, 26, 26, 0.8);
    border: 2px solid rgba(74, 222, 128, 0.4);
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    transition: all 0.3s ease;
    backdrop-filter: blur(10px);
}
.metric-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 30px rgba(74, 222, 128, 0.3);
    border-color: #4ade80;
}
.metric-value {
    font-size: 2.5rem;
    font-weight: 800;
    color: #4ade80;
    margin-bottom: 0.5rem;
    text-shadow: 0 0 20px rgba(74, 222, 128, 0.3);
}
.metric-label {
    color: #9ca3af;
    font-size: 0.95rem;
    font-weight: 500;
}

.alert-info {
    background: rgba(26, 58, 82, 0.8);
    border: 2px solid rgba(59, 130, 246, 0.5);
    color: #60a5fa;
    padding: 1.5rem;
    border-radius: 10px;
    margin: 1rem 0;
    backdrop-filter: blur(10px);
}
.alert-success {
    background: rgba(13, 79, 51, 0.8);
    border: 2px solid rgba(74, 222, 128, 0.5);
    color: #4ade80;
    padding: 1.5rem;
    border-radius: 10px;
    margin: 1rem 0;
    backdrop-filter: blur(10px);
}

.navbar {
    position: sticky;
    top: 0;
    background: rgba(10, 10, 10, 0.95);
    border-bottom: 3px solid #4ade80;
    padding: 1.5rem 0;
    z-index: 1000;
    margin: 0;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
    backdrop-filter: blur(15px);
}
.nav-links {
    display: flex;
    gap: 0.75rem;
    justify-content: center;
    flex-wrap: wrap;
    align-items: center;
    max-width: 100%;
    margin: 0 auto;
    padding: 0 1rem;
}
.nav-link {
    color: #e2e8f0;
    text-decoration: none;
    padding: 0.75rem 1.5rem;
    border-radius: 8px;
    background: rgba(26, 26, 26, 0.8);
    border: 2px solid rgba(74, 222, 128, 0.3);
    transition: all 0.3s ease;
    font-size: 0.9rem;
    font-weight: 600;
    cursor: pointer;
    position: relative;
    overflow: hidden;
    white-space: nowrap;
    backdrop-filter: blur(5px);
}
.nav-link::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(74, 222, 128, 0.2), transparent);
    transition: left 0.5s;
}
.nav-link:hover::before {
    left: 100%;
}
.nav-link:hover {
    background: linear-gradient(135deg, #4ade80 0%, #22c55e 100%);
    color: #0a0a0a;
    border-color: #4ade80;
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(74, 222, 128, 0.4);
}
.nav-link.active {
    background: linear-gradient(135deg, #006239 0%, #004d2d 100%);
    color: #ffffff;
    border-color: #4ade80;
    box-shadow: 0 0 20px rgba(74, 222, 128, 0.3);
}
.nav-link.active::after {
    content: '✓';
    margin-left: 0.5rem;
    color: #4ade80;
    font-weight: bold;
}

.tab-pane {
    display: none;
}
.tab-pane.active {
    display: block;
}

.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1.5rem;
    margin: 2rem 0;
}
.stat-item {
    background: rgba(26, 26, 26, 0.6);
    border: 1px solid rgba(74, 222, 128, 0.3);
    border-radius: 10px;
    padding: 1.5rem;
    text-align: center;
    backdrop-filter: blur(5px);
}
.stat-value {
    font-size: 1.75rem;
    font-weight: 700;
    color: #4ade80;
    text-shadow: 0 0 10px rgba(74, 222, 128, 0.3);
}
.stat-label {
    font-size: 0.9rem;
    color: #9ca3af;
    margin-top: 0.25rem;
}

.form-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 1.5rem;
}

.table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
    background: rgba(15, 15, 15, 0.6);
    border-radius: 8px;
    overflow: hidden;
}
.table th {
    background: rgba(26, 26, 26, 0.9);
    color: #4ade80;
    padding: 1rem 0.75rem;
    text-align: left;
    border-bottom: 2px solid #4ade80;
    font-weight: 600;
}
.table td {
    padding: 0.75rem;
    border-bottom: 1px solid rgba(74, 222, 128, 0.1);
    color: #e2e8f0;
}
.table tr:hover {
    background: rgba(26, 26, 26, 0.5);
}

/* Scrollbar personalizado */
::-webkit-scrollbar {
    width: 10px;
    height: 10px;
}
::-webkit-scrollbar-track {
    background: rgba(15, 15, 15, 0.5);
}
::-webkit-scrollbar-thumb {
    background: rgba(74, 222, 128, 0.5);
    border-radius: 5px;
}
::-webkit-scrollbar-thumb:hover {
    background: #4ade80;
}
"""

@rt("/")
def get():
    """Página principal"""
    
    # Verificar si hay datos cargados
    has_data = app_state.get('processor') is not None
    has_model = app_state.get('model') is not None
    
    # Obtener mensaje de éxito y tab activo
    success_msg = app_state.pop('success_message', None)
    active_tab = app_state.pop('active_tab', 'upload')
    
    return Html(
        Head(
            Title("Predicción de Precios de Casas - XGBoost"),
            Meta(charset="UTF-8"),
            Meta(name="viewport", content="width=device-width, initial-scale=1.0"),
            Style(CSS)
        ),
        Body(
            Div(
                Div(
                    H1("Sistema de Predicción de Precios de Casas"),
                    P("Machine Learning con XGBoost - Pipeline Completo"),
                    cls="header"
                ),
                
                # Menú de navegación completo (TODOS LOS BOTONES SIEMPRE VISIBLES)
                Div(
                    Div(
                        Button("Cargar Datos", onclick="showTab('upload')", id="btn-upload", cls="nav-link active", type="button"),
                        Button("Información Dataset", onclick="showTab('info')", id="btn-info", cls="nav-link", type="button"),
                        Button("Características", onclick="showTab('features')", id="btn-features", cls="nav-link", type="button"),
                        Button("Limpieza y Análisis", onclick="showTab('cleaning')", id="btn-cleaning", cls="nav-link", type="button"),
                        Button("Entrenamiento", onclick="showTab('training')", id="btn-training", cls="nav-link", type="button"),
                        Button("Evaluación", onclick="showTab('evaluation')", id="btn-evaluation", cls="nav-link", type="button"),
                        Button("Predicción", onclick="showTab('prediction')", id="btn-prediction", cls="nav-link", type="button"),
                        cls="nav-links"
                    ),
                    cls="navbar"
                ),
                
                # Mensaje de éxito
                Div(
                    P(f"{success_msg}", style="color: #4ade80; font-size: 1.1rem; font-weight: 600;"),
                    cls="alert-success",
                    style="margin-bottom: 2rem;"
                ) if success_msg else None,
                
                # Tab: Cargar datos (SIEMPRE DISPONIBLE)
                Div(
                    H2("Cargar Dataset", cls="section-title"),
                    P("Sube tu archivo CSV con datos de casas", cls="section-subtitle"),
                    
                    Div(
                        Form(
                            Div(
                                Label("Seleccionar Archivo CSV", style="font-size: 1.1rem; margin-bottom: 1rem; color: #4ade80;"),
                                Div(
                                    Input(
                                        type="file", 
                                        name="file", 
                                        accept=".csv", 
                                        required=True,
                                        id="fileUpload",
                                        style="padding: 1.25rem; background: rgba(10, 10, 10, 0.6); border: 2px solid rgba(74, 222, 128, 0.4); border-radius: 10px; font-size: 1rem; cursor: pointer;"
                                    ),
                                    style="margin-bottom: 2rem;"
                                ),
                                cls="form-group"
                            ),
                            Button(
                                "Cargar y Procesar Datos", 
                                type="submit", 
                                cls="btn btn-block",
                                style="padding: 1.25rem 3rem; font-size: 1rem; border-radius: 10px; letter-spacing: 1px;"
                            ),
                            action="/load_data",
                            method="post",
                            enctype="multipart/form-data",
                            style="max-width: 600px; margin: 0 auto;"
                        ),
                        style="background: rgba(15, 15, 15, 0.5); padding: 3rem; border-radius: 16px; border: 2px solid rgba(74, 222, 128, 0.3); box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);"
                    ),
                    
                    Script("""
                        const fileUpload = document.getElementById('fileUpload');
                        if (fileUpload) {
                            fileUpload.addEventListener('change', function(e) {
                                if (e.target.files.length > 0) {
                                    this.style.borderColor = '#4ade80';
                                    this.style.background = 'rgba(74, 222, 128, 0.1)';
                                }
                            });
                        }
                    """),
                    
                    id="upload",
                    cls=f"section tab-pane{' active' if active_tab == 'upload' else ''}"
                ),
                
                # TODAS las tabs siempre presentes
                generate_info_tab(app_state, has_data, active_tab),
                generate_features_tab(app_state, has_data, active_tab),
                generate_cleaning_tab(app_state, has_data, active_tab),
                generate_training_tab(app_state, has_data, active_tab),
                generate_evaluation_tab(app_state, has_model, active_tab),
                generate_prediction_tab(app_state, has_model, active_tab),
                
                Script(f"""
                    function showTab(tabId) {{
                        document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
                        document.querySelectorAll('.nav-link').forEach(b => b.classList.remove('active'));
                        document.getElementById(tabId).classList.add('active');
                        document.getElementById('btn-' + tabId).classList.add('active');
                    }}
                    // Activar tab correcto al cargar
                    window.addEventListener('load', function() {{
                        showTab('{active_tab}');
                    }});
                """),
                
                cls="container"
            )
        )
    )


def generate_info_tab(state, has_data, active_tab='upload'):
    """Genera tab de información del dataset"""
    info = state.get('initial_info', {})
    top15_chart = state.get('top15_chart', '')
    
    if not has_data or not info:
        return Div(
            H2("Información del Dataset", cls="section-title"),
            Div(
                P("Primero debes cargar un dataset", style="font-size: 1.2rem; color: #fbbf24;"),
                P("Ve a la pestaña 'Cargar Datos' y sube un archivo CSV", style="color: #898989;"),
                cls="alert-info"
            ),
            id="info",
            cls=f"section tab-pane{' active' if active_tab == 'info' else ''}"
        )
    
    return Div(
        H2("Información del Dataset", cls="section-title"),
        
        Div(
            H3("Dimensiones:", style="color: #4ade80; margin-bottom: 0.5rem;"),
            P(f"{info['shape'][0]:,} filas × {info['shape'][1]} columnas", 
              style="font-size: 1.1rem; color: #e2e8f0;"),
            cls="step-box"
        ),
        
        Div(
            H3("Columnas del Dataset:", style="color: #4ade80; margin-bottom: 1rem;"),
            Div(
                P(f"Total: {len(info['columns'])} columnas"),
                Div(
                    *[Span(col, style="display: inline-block; background: #1a1a1a; padding: 0.4rem 0.8rem; margin: 0.2rem; border-radius: 4px; font-size: 0.85rem; border: 1px solid #292929;") 
                      for col in info['columns']],
                    style="margin-top: 0.5rem;"
                ),
                style="color: #e2e8f0;"
            ),
            cls="step-box"
        ),
        
        Div(
            H3("Primeros 20 registros:", style="color: #4ade80; margin-bottom: 1rem;"),
            Div(
                NotStr(info['head'].to_html(classes="table", border=0, index=False)),
                style="overflow-x: auto; max-height: 400px; overflow-y: auto;"
            ),
            cls="step-box",
            style="margin-bottom: 2rem;"
        ),
        
        Div(
            H3("Cantidad de nulos y sus tipos:", style="color: #4ade80; margin-bottom: 1rem;"),
            Div(
                NotStr(info['na_df'].to_html(classes="table", border=0)),
                style="overflow-x: auto; max-height: 400px; overflow-y: auto;"
            ),
            cls="step-box"
        ),
        
        Div(
            H3("Top 15 columnas con más nulos:", style="color: #4ade80; margin-bottom: 1rem;"),
            Div(
                Img(src=top15_chart),
                cls="chart-box"
            ),
            cls="step-box"
        ),
        
        id="info",
        cls=f"section tab-pane{' active' if active_tab == 'info' else ''}"
    )


def generate_features_tab(state, has_data, active_tab='upload'):
    """Genera tab de características estadísticas (.describe())"""
    df = state.get('df_original')
    
    if not has_data or df is None:
        return Div(
            H2("Características Estadísticas", cls="section-title"),
            Div(
                P("Primero debes cargar un dataset", style="font-size: 1.2rem; color: #fbbf24;"),
                P("Ve a la pestaña 'Cargar Datos' y sube un archivo CSV", style="color: #898989;"),
                cls="alert-info"
            ),
            id="features",
            cls=f"section tab-pane{' active' if active_tab == 'features' else ''}"
        )
    
    # Obtener describe de columnas numéricas
    df_describe = df.describe()
    
    # Generar gráficos estadísticos de SalePrice
    from plot_utils import create_saleprice_boxplot, create_saleprice_quartiles_chart
    
    boxplot_chart = create_saleprice_boxplot(df)
    quartiles_chart = create_saleprice_quartiles_chart(df)
    
    return Div(
        H2("Características Estadísticas", cls="section-title"),
        P("Resumen estadístico de las variables numéricas del dataset", cls="section-subtitle"),
        
        # Tabla de estadísticas descriptivas
        Div(
            H3("Estadísticas Descriptivas (.describe())", style="color: #4ade80; margin-bottom: 1rem;"),
            Div(
                NotStr(df_describe.to_html(classes="table", border=0)),
                style="overflow-x: auto; max-height: 600px; overflow-y: auto;"
            ),
            cls="step-box"
        ),
        
        # Boxplot de SalePrice
        Div(
            H3("Boxplot de SalePrice", style="color: #4ade80; margin-bottom: 1rem;"),
            Img(src=boxplot_chart, style="width: 100%; max-width: 800px; margin: 1rem auto; display: block;") if boxplot_chart else P("No hay datos de SalePrice"),
            cls="step-box"
        ) if boxplot_chart else None,
        
        # Distribución por Cuartiles
        Div(
            H3("Distribución por Cuartiles de SalePrice", style="color: #4ade80; margin-bottom: 1rem;"),
            Img(src=quartiles_chart, style="width: 100%; max-width: 800px; margin: 1rem auto; display: block;") if quartiles_chart else P("No hay datos de SalePrice"),
            cls="step-box"
        ) if quartiles_chart else None,
        
        id="features",
        cls=f"section tab-pane{' active' if active_tab == 'features' else ''}"
    )


def generate_cleaning_tab(state, has_data, active_tab='upload'):
    """Genera tab de limpieza y análisis"""
    processor = state.get('processor')
    charts = state.get('charts', {})
    
    if not has_data or not processor:
        return Div(
            H2("Limpieza y Análisis", cls="section-title"),
            Div(
                P("Primero debes cargar un dataset", style="font-size: 1.2rem; color: #fbbf24;"),
                P("Ve a la pestaña 'Cargar Datos' y sube un archivo CSV", style="color: #898989;"),
                cls="alert-info"
            ),
            id="cleaning",
            cls=f"section tab-pane{' active' if active_tab == 'cleaning' else ''}"
        )
    
    df = state.get('df_original')
    df_processed = state.get('df_processed')
    
    return Div(
        H2("Limpieza y Análisis", cls="section-title"),
        P("Pipeline de preprocesamiento y análisis de correlaciones", cls="section-subtitle"),
        
        # Mostrar cada paso
        *[Div(
            H3(f"PASO {step['step']}: {step['name']}", cls="step-title"),
            Div(
                Pre(step['output']),
                cls="step-output"
            ),
            Div(
                Img(src=charts.get('steps', {}).get(f"step{step['step']}", "")),
                cls="chart-box"
            ) if f"step{step['step']}" in charts.get('steps', {}) else None,
            cls="step-box"
        ) for step in processor.steps_output],
        
        # Resumen
        Div(
            Div(
                Div(
                    Div(f"{len(df):,}", cls="metric-value"),
                    Div("Registros Iniciales", cls="metric-label"),
                    cls="metric-card"
                ),
                Div(
                    Div(f"{len(df) - len(df_processed):,}", cls="metric-value"),
                    Div("Filas Eliminadas", cls="metric-label"),
                    cls="metric-card"
                ),
                Div(
                    Div(f"{len(df_processed):,}", cls="metric-value"),
                    Div("Dataset Final", cls="metric-label"),
                    cls="metric-card"
                ),
                cls="metric-grid"
            ),
            cls="alert-success"
        ),
        
        id="cleaning",
        cls="section tab-pane"
    )


def generate_training_tab(state, has_data, active_tab='upload'):
    """Genera tab de entrenamiento"""
    if not has_data:
        return Div(
            H2("Entrenar Modelo XGBoost", cls="section-title"),
            Div(
                P("Primero debes cargar un dataset", style="font-size: 1.2rem; color: #fbbf24;"),
                P("Ve a la pestaña 'Cargar Datos' y sube un archivo CSV", style="color: #898989;"),
                cls="alert-info"
            ),
            id="training",
            cls=f"section tab-pane{' active' if active_tab == 'training' else ''}"
        )
    
    return Div(
        H2("Entrenar Modelo XGBoost", cls="section-title"),
        P("Optimización de hiperparámetros con RandomizedSearchCV", cls="section-subtitle"),
        
        Div(
            P("Características seleccionadas para el modelo:"),
            Div(
                *[Span(feat, style="display: inline-block; background: #1a1a1a; padding: 0.4rem 0.8rem; margin: 0.2rem; border-radius: 4px; font-size: 0.85rem; border: 1px solid #4ade80;") 
                  for feat in ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'YearBuilt', 'YearRemodAdd', 'TotRmsAbvGrd']],
                style="margin-top: 0.5rem; margin-bottom: 1.5rem;"
            ),
            cls="step-box"
        ),
        
        Form(
            Button("Iniciar Entrenamiento del Modelo", type="submit", cls="btn btn-block"),
            action="/train",
            method="post"
        ),
        id="training",
        cls=f"section tab-pane{' active' if active_tab == 'training' else ''}"
    )



def generate_evaluation_tab(state, has_model, active_tab='upload'):
    """Genera tab de evaluación"""
    trainer = state.get('trainer')
    metrics = state.get('metrics')
    importance_chart = state.get('importance_chart')
    predictions_chart = state.get('predictions_chart')
    residuals_chart = state.get('residuals_chart')
    
    if not has_model or not trainer or not metrics:
        return Div(
            H2("Evaluación del Modelo", cls="section-title"),
            Div(
                P("Primero debes entrenar el modelo", style="font-size: 1.2rem; color: #fbbf24;"),
                P("Ve a la pestaña 'Entrenamiento' y entrena el modelo", style="color: #898989;"),
                cls="alert-info"
            ),
            id="evaluation",
            cls=f"section tab-pane{' active' if active_tab == 'evaluation' else ''}"
        )
    
    return Div(
        H2("Evaluación del Modelo", cls="section-title"),
        
        # Outputs de entrenamiento
        *[Div(
            H3(output['name'], cls="step-title"),
            Div(
                Pre(output['output']),
                cls="step-output"
            ),
            cls="step-box"
        ) for output in trainer.training_output],
        
        # Métricas
        Div(
            Div(
                Div(
                    Div(f"{metrics['r2']:.4f}", cls="metric-value"),
                    Div("R² Score", cls="metric-label"),
                    cls="metric-card"
                ),
                Div(
                    Div(f"${metrics['rmse']:,.0f}", cls="metric-value"),
                    Div("RMSE", cls="metric-label"),
                    cls="metric-card"
                ),
                Div(
                    Div(f"${metrics['mae']:,.0f}", cls="metric-value"),
                    Div("MAE", cls="metric-label"),
                    cls="metric-card"
                ),
                cls="metric-grid"
            ),
            cls="alert-success"
        ),
        
        # Gráfico: Predicciones vs Valores Reales
        Div(
            H3("Predicciones vs Valores Reales", cls="step-title"),
            Div(
                Img(src=predictions_chart),
                cls="chart-box"
            ),
            cls="step-box"
        ) if predictions_chart else None,
        
        # Gráfico: Residuos
        Div(
            H3("Análisis de Residuos", cls="step-title"),
            Div(
                Img(src=residuals_chart),
                cls="chart-box"
            ),
            cls="step-box"
        ) if residuals_chart else None,
        
        # Importancia de características
        Div(
            H3("Importancia de Características", cls="step-title"),
            Div(
                Img(src=importance_chart),
                cls="chart-box"
            ),
            cls="step-box"
        ),
        
        id="evaluation",
        cls=f"section tab-pane{' active' if active_tab == 'evaluation' else ''}"
    )


def generate_prediction_tab(state, has_model, active_tab='upload'):
    """Genera tab de predicción"""
    features = state.get('features')
    last_prediction = state.get('last_prediction')
    
    # Traducción de campos al español
    traduccion = {
        'OverallQual': 'Calidad General',
        'GrLivArea': 'Área Habitable (pies²)',
        'GarageCars': 'Capacidad Garaje (autos)',
        'GarageArea': 'Área Garaje (pies²)',
        'TotalBsmtSF': 'Área Sótano Total (pies²)',
        '1stFlrSF': 'Área 1er Piso (pies²)',
        'FullBath': 'Baños Completos',
        'YearBuilt': 'Año de Construcción',
        'YearRemodAdd': 'Año de Remodelación',
        'TotRmsAbvGrd': 'Habitaciones Totales'
    }
    
    if not has_model or not features:
        return Div(
            H2("Hacer Predicción", cls="section-title"),
            Div(
                P("Primero debes entrenar el modelo", style="font-size: 1.2rem; color: #fbbf24;"),
                P("Ve a la pestaña 'Entrenamiento' y entrena el modelo", style="color: #898989;"),
                cls="alert-info"
            ),
            id="prediction",
            cls=f"section tab-pane{' active' if active_tab == 'prediction' else ''}"
        )
    
    content = [
        H2("Hacer Predicción", cls="section-title"),
        P("Ingresa las características de una casa para predecir su precio", cls="section-subtitle"),
    ]
    
    # Mostrar última predicción si existe
    if last_prediction:
        content.append(
            Div(
                H3("Última Predicción:", cls="step-title", style="color: #4ade80;"),
                Div(
                    H4("Características:", style="color: #4ade80; margin-bottom: 0.5rem;"),
                    *[P(f"{k}: {v}", style="color: #e2e8f0;") for k, v in last_prediction['features'].items()],
                    cls="step-box"
                ),
                Div(
                    Div(
                        Div(f"${last_prediction['price']:,.0f}", cls="metric-value", style="font-size: 2.5rem;"),
                        Div("Precio Predicho", cls="metric-label"),
                        cls="metric-card"
                    ),
                    style="max-width: 400px; margin: 1.5rem auto;"
                ),
                cls="alert-success",
                style="margin-bottom: 2rem;"
            )
        )
    
    # Valores por defecto y rangos para cada característica
    valores_defaults = {
        'OverallQual': {'default': 7, 'min': 1, 'max': 10, 'step': 1, 'placeholder': 'Ej: 7 (1-10)'},
        'GrLivArea': {'default': 1500, 'min': 334, 'max': 5642, 'step': 1, 'placeholder': 'Ej: 1500 (334-5642)'},
        'GarageCars': {'default': 2, 'min': 0, 'max': 4, 'step': 1, 'placeholder': 'Ej: 2 (0-4)'},
        'GarageArea': {'default': 480, 'min': 0, 'max': 1418, 'step': 1, 'placeholder': 'Ej: 480 (0-1418)'},
        'TotalBsmtSF': {'default': 1000, 'min': 0, 'max': 6110, 'step': 1, 'placeholder': 'Ej: 1000 (0-6110)'},
        '1stFlrSF': {'default': 1000, 'min': 334, 'max': 4692, 'step': 1, 'placeholder': 'Ej: 1000 (334-4692)'},
        'FullBath': {'default': 2, 'min': 0, 'max': 3, 'step': 1, 'placeholder': 'Ej: 2 (0-3)'},
        'YearBuilt': {'default': 2000, 'min': 1872, 'max': 2010, 'step': 1, 'placeholder': 'Ej: 2000 (1872-2010)'},
        'YearRemodAdd': {'default': 2000, 'min': 1950, 'max': 2010, 'step': 1, 'placeholder': 'Ej: 2000 (1950-2010)'},
        'TotRmsAbvGrd': {'default': 6, 'min': 2, 'max': 14, 'step': 1, 'placeholder': 'Ej: 6 (2-14)'}
    }
    
    # Formulario de nueva predicción con validaciones
    content.append(
        Div(
            H3("Nueva Predicción:", cls="step-title", style="color: #4ade80;"),
            Div(
                P("Ejemplo de casa típica precargado. Modifica los valores según necesites.", 
                  style="color: #fbbf24; margin-bottom: 1rem; font-size: 0.95rem;"),
                cls="alert-info"
            ),
            Form(
                Div(
                    *[Div(
                        Label(traduccion.get(feat, feat)),
                        Input(
                            type="number", 
                            name=feat, 
                            required=True, 
                            step=str(valores_defaults[feat]['step']),
                            min=str(valores_defaults[feat]['min']),
                            max=str(valores_defaults[feat]['max']),
                            value=str(valores_defaults[feat]['default']),
                            placeholder=valores_defaults[feat]['placeholder']
                        ),
                        cls="form-group"
                    ) for feat in features if feat in valores_defaults],
                    cls="form-grid"
                ),
                Button("Calcular Precio Predicho", type="submit", cls="btn btn-block"),
                action="/predict",
                method="post"
            ),
            cls="step-box"
        )
    )
    
    return Div(
        *content,
        id="prediction",
        cls=f"section tab-pane{' active' if active_tab == 'prediction' else ''}"
    )


@rt("/load_data")
async def post(file: UploadFile):
    """Carga y procesa el dataset"""
    try:
        # Guardar archivo
        file_path = UPLOAD_DIR / file.filename
        file_path.write_bytes(await file.read())
        
        # Leer CSV
        df = pd.read_csv(file_path)
        app_state['df_original'] = df
        
        # Obtener información inicial
        processor = HousePriceDataProcessor()
        initial_info = processor.get_initial_info(df)
        
        # Procesar datos
        df_processed = processor.process_data(df)
        
        app_state['df_processed'] = df_processed
        app_state['processor'] = processor
        
        # Generar gráficos de cada paso
        charts = {}
        
        # PASO 1: Nulos
        if 'step1_nulls' in processor.step_charts_data:
            data = processor.step_charts_data['step1_nulls']
            charts['step1'] = create_nulls_filled_chart(
                data['nulos_numericos'],
                data['nulos_categoricos']
            )
        
        # PASO 2: Correlación
        if 'step2_correlation' in processor.step_charts_data:
            corr_data = processor.step_charts_data['step2_correlation']
            charts['step2'] = create_correlation_bar_chart(corr_data)
        
        # PASO 4: Outliers
        if 'step4_outliers' in processor.step_charts_data:
            data = processor.step_charts_data['step4_outliers']
            charts['step4'] = create_outliers_comparison_chart(
                data['area_antes'],
                data['precio_antes'],
                data['area_despues'],
                data['precio_despues'],
                data['umbral']
            )
        
        # PASO 5: Log transformation
        if 'step5_log' in processor.step_charts_data:
            data = processor.step_charts_data['step5_log']
            charts['step5'] = create_log_transformation_chart(
                data['precio_original'],
                data['precio_log']
            )
        
        # Estadísticas del dataset
        info = initial_info
        
        # Gráfico top 15 nulos
        top15_chart = create_top_nulls_chart(df)
        
        app_state['charts'] = {'steps': charts}
        app_state['initial_info'] = info
        app_state['top15_chart'] = top15_chart
        app_state['success_message'] = 'Dataset cargado y procesado exitosamente'
        app_state['active_tab'] = 'info'
        
        # REDIRIGIR A LA MISMA PÁGINA PRINCIPAL PARA MOSTRAR PESTAÑAS
        return RedirectResponse('/', status_code=303)
        
    except Exception as e:
        return Div(
            H1("Error"),
            P(f"Error al procesar archivo: {str(e)}"),
            A("Volver", href="/"),
            cls="alert-info"
        )


@rt("/train")
async def post():
    """Entrena el modelo - PASO 2"""
    try:
        if app_state['processor'] is None:
            return RedirectResponse('/', status_code=303)
        
        processor = app_state['processor']
        df_processed = app_state['df_processed']
        
        # Preparar features
        df_encoded = processor.encode_categorical(df_processed)
        X, y, features = processor.prepare_features(df_encoded)
        
        # Entrenar modelo
        trainer = ModelTrainer()
        model, metrics, X_test, y_test, y_pred = trainer.train_model(X, y)
        
        # Guardar modelo
        trainer.save_model(str(MODELS_DIR), features, processor.label_encoders)
        
        # Guardar en estado
        app_state['trainer'] = trainer
        app_state['model'] = model
        app_state['features'] = features
        app_state['metrics'] = metrics
        
        # Gráficos
        feature_importance = trainer.get_feature_importance(features)
        importance_chart = create_feature_importance_chart(feature_importance)
        predictions_chart = create_predictions_vs_real_chart(y_test, y_pred)
        residuos = y_test - y_pred
        residuals_chart = create_residuals_chart(y_pred, residuos)
        
        app_state['importance_chart'] = importance_chart
        app_state['predictions_chart'] = predictions_chart
        app_state['residuals_chart'] = residuals_chart
        app_state['success_message'] = f'Modelo entrenado exitosamente | R²: {metrics["r2"]:.4f}'
        app_state['active_tab'] = 'evaluation'
        
        # REDIRIGIR A LA PÁGINA PRINCIPAL
        return RedirectResponse('/', status_code=303)
        
    except Exception as e:
        return Div(
            H1("Error"),
            P(f"Error al entrenar: {str(e)}"),
            A("Volver", href="/"),
            cls="alert-info"
        )


@rt("/predict")
async def post(request):
    """Hace predicción"""
    try:
        if app_state['model'] is None:
            return RedirectResponse('/', status_code=303)
        
        # Obtener datos del formulario
        form_data = await request.form()
        features_dict = {k: float(v) for k, v in form_data.items()}
        
        # Predecir
        trainer = app_state['trainer']
        prediction = trainer.predict(features_dict)
        
        # Guardar en estado
        app_state['last_prediction'] = {
            'features': features_dict,
            'price': prediction
        }
        
        # Redirigir a la página principal
        return RedirectResponse('/', status_code=303)
        
    except Exception as e:
        return Div(
            H1("Error"),
            P(f"Error en predicción: {str(e)}"),
            A("Volver", href="/"),
            cls="alert-info"
        )


# Iniciar servidor
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5001))
    serve(host="0.0.0.0", port=port)
