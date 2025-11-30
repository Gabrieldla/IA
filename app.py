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
    font-family: 'Inter', -apple-system, sans-serif; 
    background: #0a0a0a; 
    color: #e2e8f0; 
    line-height: 1.6; 
}
.container { max-width: 1400px; margin: 0 auto; padding: 2rem; }
.header { 
    background: linear-gradient(135deg, #006239 0%, #004d2d 100%);
    padding: 2.5rem; 
    text-align: center; 
    border-radius: 12px; 
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(0, 98, 57, 0.3);
}
.header h1 { 
    font-size: 2.5rem; 
    color: #ffffff; 
    font-weight: 800; 
    margin-bottom: 0.5rem;
}
.header p { color: #4ade80; font-size: 1.1rem; font-weight: 500; }

.section { 
    background: #171717; 
    border: 1px solid #292929; 
    border-radius: 8px; 
    padding: 2.5rem; 
    margin-bottom: 2rem; 
}
.section-title { 
    font-size: 1.75rem; 
    color: #ffffff; 
    margin-bottom: 0.5rem; 
    font-weight: 700; 
}
.section-subtitle { color: #898989; margin-bottom: 2rem; }

.form-group { margin-bottom: 1.5rem; }
label { 
    display: block; 
    font-weight: 600; 
    color: #e2e8f0; 
    margin-bottom: 0.5rem; 
}
input[type="file"], input[type="number"] {
    width: 100%;
    padding: 1rem;
    border: 2px solid #292929;
    background: #0f0f0f;
    color: #e2e8f0;
    border-radius: 6px;
    font-size: 1rem;
}
input:focus {
    outline: none;
    border-color: #4ade80;
    background: #1a1a1a;
}

.btn {
    background: linear-gradient(135deg, #006239 0%, #004d2d 100%);
    color: #ffffff;
    border: none;
    padding: 0.875rem 2rem;
    font-size: 0.85rem;
    font-weight: 600;
    border-radius: 6px;
    cursor: pointer;
    width: 100%;
    max-width: 300px;
    text-transform: uppercase;
    transition: all 0.3s;
    display: block;
    margin: 0 auto;
}
.btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(74, 222, 128, 0.5);
}
.btn-block {
    max-width: 100%;
}

.step-box {
    background: #1a1a1a;
    border-left: 4px solid #4ade80;
    border-radius: 8px;
    padding: 2rem;
    margin-bottom: 2rem;
}
.step-title {
    color: #ffffff;
    font-size: 1.3rem;
    margin-bottom: 1rem;
    border-bottom: 2px solid #4ade80;
    padding-bottom: 0.5rem;
}
.step-output {
    background: #0f0f0f;
    padding: 1.25rem;
    border-radius: 6px;
    margin-bottom: 1.5rem;
}
.step-output pre {
    color: #4ade80;
    font-size: 1rem;
    margin: 0;
    white-space: pre-wrap;
}

.chart-box {
    background: #0f0f0f;
    padding: 1.5rem;
    border-radius: 6px;
    margin-bottom: 2rem;
    max-height: 500px;
    overflow-y: auto;
}
.chart-box img {
    max-width: 100%;
    height: auto;
    border-radius: 6px;
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1.5rem;
    margin: 2rem 0;
}
.metric-card {
    background: #1a1a1a;
    border: 2px solid #4ade80;
    border-radius: 8px;
    padding: 1.5rem;
    text-align: center;
}
.metric-value {
    font-size: 2rem;
    font-weight: 800;
    color: #4ade80;
    margin-bottom: 0.5rem;
}
.metric-label {
    color: #898989;
    font-size: 0.9rem;
}

.alert-info {
    background: #1a3a52;
    border: 2px solid #3b82f6;
    color: #60a5fa;
    padding: 1.5rem;
    border-radius: 8px;
    margin: 1rem 0;
}
.alert-success {
    background: #0d4f33;
    border: 2px solid #4ade80;
    color: #4ade80;
    padding: 1.5rem;
    border-radius: 8px;
    margin: 1rem 0;
}

.navbar {
    position: sticky;
    top: 0;
    background: #171717;
    border-bottom: 2px solid #4ade80;
    padding: 1rem 0;
    z-index: 100;
    margin-bottom: 2rem;
}
.nav-links {
    display: flex;
    gap: 0.5rem;
    justify-content: center;
    flex-wrap: wrap;
}
.nav-link {
    color: #e2e8f0;
    text-decoration: none;
    padding: 0.6rem 1.2rem;
    border-radius: 6px;
    background: #1a1a1a;
    border: 1px solid #292929;
    transition: all 0.3s;
    font-size: 0.85rem;
    cursor: pointer;
}
.nav-link:hover {
    background: #4ade80;
    color: #0a0a0a;
    border-color: #4ade80;
}
.nav-link.active {
    background: #006239;
    color: #ffffff;
    border-color: #4ade80;
}

.tab-pane {
    display: none;
}
.tab-pane.active {
    display: block;
}

.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1rem;
    margin: 1.5rem 0;
}
.stat-item {
    background: #1a1a1a;
    border: 1px solid #292929;
    border-radius: 6px;
    padding: 1rem;
    text-align: center;
}
.stat-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #4ade80;
}
.stat-label {
    font-size: 0.85rem;
    color: #898989;
    margin-top: 0.25rem;
}

.form-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1rem;
}

.table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
}
.table th {
    background: #1a1a1a;
    color: #4ade80;
    padding: 0.75rem;
    text-align: left;
    border-bottom: 2px solid #4ade80;
}
.table td {
    padding: 0.5rem 0.75rem;
    border-bottom: 1px solid #292929;
    color: #e2e8f0;
}
.table tr:hover {
    background: #1a1a1a;
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
                    Form(
                        Div(
                            Label("Seleccionar Archivo CSV:"),
                            Input(type="file", name="file", accept=".csv", required=True),
                            cls="form-group"
                        ),
                        Button("Cargar y Procesar Datos", type="submit", cls="btn btn-block"),
                        action="/load_data",
                        method="post",
                        enctype="multipart/form-data"
                    ),
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
    
    return Div(
        H2("Características Estadísticas", cls="section-title"),
        P("Resumen estadístico de las variables numéricas del dataset", cls="section-subtitle"),
        
        Div(
            H3("Estadísticas Descriptivas (.describe()):", style="color: #4ade80; margin-bottom: 1rem;"),
            Div(
                NotStr(df_describe.to_html(classes="table", border=0)),
                style="overflow-x: auto; max-height: 600px; overflow-y: auto;"
            ),
            cls="step-box"
        ),
        
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
    
    # Formulario de nueva predicción
    content.append(
        Div(
            H3("Nueva Predicción:", cls="step-title", style="color: #4ade80;"),
            Form(
                Div(
                    *[Div(
                        Label(traduccion.get(feat, feat)),
                        Input(type="number", name=feat, required=True, step="any", placeholder=f"Ingrese {traduccion.get(feat, feat).lower()}"),
                        cls="form-group"
                    ) for feat in features],
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
                data['nulls_numeric'],
                data['nulls_categorical']
            )
        
        # PASO 2: Correlación
        if 'step2_correlation' in processor.step_charts_data:
            corr_data = processor.step_charts_data['step2_correlation']
            charts['step2'] = create_correlation_bar_chart(corr_data)
        
        # PASO 4: Outliers
        if 'step4_outliers' in processor.step_charts_data:
            data = processor.step_charts_data['step4_outliers']
            charts['step4'] = create_outliers_comparison_chart(
                data['grliv_before'],
                data['price_before'],
                data['grliv_after'],
                data['price_after'],
                data['threshold']
            )
        
        # PASO 5: Log transformation
        if 'step5_log' in processor.step_charts_data:
            data = processor.step_charts_data['step5_log']
            charts['step5'] = create_log_transformation_chart(
                data['price_original'],
                data['price_log']
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
        model, metrics, X_test, y_test, y_pred = trainer.train_model(X, y, optimize=True)
        
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
    serve()
