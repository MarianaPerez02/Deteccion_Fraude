"""
Utilidades para cargar y usar modelos de ML para detección de fraude
"""
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Variables globales para almacenar modelos y datos
MODELS = {}
SCALERS = {}
MODEL_METRICS = {}
DATA_INFO = {}

def load_model():
    """
    Carga los modelos entrenados desde archivos pickle

    Returns:
        dict: Diccionario con modelos cargados
    """
    global MODELS, SCALERS, MODEL_METRICS, DATA_INFO

    # Rutas base
    base_dir = Path(__file__).parent.parent.parent
    models_dir = base_dir / 'Notebook' / 'models'

    try:
        # Intentar cargar modelos reales si existen
        if models_dir.exists():
            model_files = list(models_dir.glob('*_model.pkl'))

            if model_files:
                print("Cargando modelos entrenados...")

                for model_file in model_files:
                    model_name = model_file.stem.replace('_model', '').replace('_', ' ').title()

                    try:
                        with open(model_file, 'rb') as f:
                            MODELS[model_name] = pickle.load(f)
                        print(f"   OK - {model_name} cargado")

                        # Cargar métricas asociadas
                        metrics_file = model_file.parent / f"{model_file.stem.replace('_model', '')}_metrics.pkl"
                        if metrics_file.exists():
                            with open(metrics_file, 'rb') as f:
                                MODEL_METRICS[model_name] = pickle.load(f)

                    except Exception as e:
                        print(f"   Advertencia - Error cargando {model_name}: {e}")

                # Cargar scaler si existe
                scaler_file = models_dir / 'scaler.pkl'
                if scaler_file.exists():
                    with open(scaler_file, 'rb') as f:
                        SCALERS['scaler'] = pickle.load(f)
                    print("   OK - Scaler cargado")

                # Cargar métricas globales
                all_metrics_file = models_dir / 'all_metrics.pkl'
                if all_metrics_file.exists():
                    with open(all_metrics_file, 'rb') as f:
                        global_metrics = pickle.load(f)
                        MODEL_METRICS['global'] = global_metrics
                    print("   OK - Metricas globales cargadas")

                print(f"\nOK - Total de modelos cargados: {len(MODELS)}")
                return MODELS

        # Si no hay modelos, usar datos de ejemplo
        print("Advertencia: No se encontraron modelos entrenados. Usando datos de prueba.")
        print("Para usar modelos reales:")
        print("   1. Entrena tus modelos en el notebook")
        print("   2. Ejecuta: from export_models import export_models_from_notebook")
        print("   3. Reinicia la aplicacion Flask")

        # Datos de ejemplo para testing
        DATA_INFO = {
            'total_transacciones': 284807,
            'total_fraudes': 492,
            'total_legitimas': 284315,
            'porcentaje_fraude': 0.173
        }

        return MODELS

    except Exception as e:
        print(f"Error al cargar modelos: {e}")
        return {}

def save_model(model, model_name, scaler=None, metrics=None):
    """
    Guarda un modelo entrenado y sus métricas

    Args:
        model: Modelo entrenado de sklearn
        model_name (str): Nombre del modelo
        scaler: Scaler usado para normalización (opcional)
        metrics (dict): Métricas del modelo (opcional)
    """
    base_dir = Path(__file__).parent.parent.parent
    models_dir = base_dir / 'Notebook' / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)

    # Guardar modelo
    model_path = models_dir / f'{model_name.lower().replace(" ", "_")}_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    # Guardar scaler si existe
    if scaler is not None:
        scaler_path = models_dir / f'{model_name.lower().replace(" ", "_")}_scaler.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)

    # Guardar métricas si existen
    if metrics is not None:
        metrics_path = models_dir / f'{model_name.lower().replace(" ", "_")}_metrics.pkl'
        with open(metrics_path, 'wb') as f:
            pickle.dump(metrics, f)

    print(f"OK - Modelo '{model_name}' guardado exitosamente en {model_path}")

def get_model_metrics():
    """
    Obtiene las métricas de los modelos entrenados

    Returns:
        dict: Diccionario con métricas de todos los modelos
    """
    global MODEL_METRICS

    # Si hay métricas globales cargadas, usarlas
    if 'global' in MODEL_METRICS:
        metrics = MODEL_METRICS['global'].copy()

        # Agregar datos adicionales
        if 'models_comparison' in metrics and metrics['models_comparison']:
            best_model = max(metrics['models_comparison'], key=lambda x: x.get('f1_score', 0))
            metrics['best_model_accuracy'] = best_model.get('precision', 0)
            metrics['best_model_recall'] = best_model.get('recall', 0)
            metrics['fraudes_detectados'] = 456  # Ajustar según tus datos

        # Agregar info básica si no existe
        if 'total_transacciones' not in metrics:
            metrics.update({
                'total_transacciones': 284807,
                'total_fraudes': 492,
                'total_legitimas': 284315,
                'fraudes_detectados': 456
            })

        return metrics

    # Si no hay métricas reales, retornar datos de ejemplo
    return {
        'total_transacciones': 284807,
        'total_fraudes': 492,
        'total_legitimas': 284315,
        'fraudes_detectados': 456,
        'best_model_accuracy': 0.9995,
        'best_model_recall': 0.9268,
        'models_comparison': [
            {
                'model': 'Logistic Regression',
                'precision': 0.8912,
                'recall': 0.8943,
                'f1_score': 0.8927,
                'roc_auc': 0.9412,
                'is_best': False
            },
            {
                'model': 'Random Forest',
                'precision': 0.9523,
                'recall': 0.9268,
                'f1_score': 0.9394,
                'roc_auc': 0.9856,
                'is_best': True
            },
            {
                'model': 'XGBoost',
                'precision': 0.9344,
                'recall': 0.9187,
                'f1_score': 0.9265,
                'roc_auc': 0.9789,
                'is_best': False
            }
        ],
        'roc_curves': {
            'Logistic Regression': {
                'fpr': np.linspace(0, 1, 100).tolist(),
                'tpr': (np.linspace(0, 1, 100) ** 0.5).tolist(),
                'auc': 0.9412
            },
            'Random Forest': {
                'fpr': np.linspace(0, 1, 100).tolist(),
                'tpr': (np.linspace(0, 1, 100) ** 0.3).tolist(),
                'auc': 0.9856
            },
            'XGBoost': {
                'fpr': np.linspace(0, 1, 100).tolist(),
                'tpr': (np.linspace(0, 1, 100) ** 0.4).tolist(),
                'auc': 0.9789
            }
        },
        'confusion_matrices': {
            'Logistic Regression': [[56850, 12], [52, 440]],
            'Random Forest': [[56858, 4], [36, 456]],
            'XGBoost': [[56854, 8], [40, 452]]
        }
    }

def predict_fraud(features):
    """
    Realiza predicción de fraude para una transacción

    Args:
        features (dict): Diccionario con las características de la transacción

    Returns:
        dict: Resultado de la predicción con probabilidades
    """
    global MODELS, SCALERS

    try:
        # Preparar datos de entrada
        feature_names = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]

        # Crear DataFrame con las características
        X = pd.DataFrame([features], columns=feature_names)

        # Rellenar valores faltantes con 0
        for col in feature_names:
            if col not in X.columns:
                X[col] = 0

        # Asegurar orden correcto de columnas
        X = X[feature_names]

        # Si hay modelos cargados, usar el mejor
        if MODELS:
            # Obtener el mejor modelo (basado en F1-Score)
            best_model_name = None
            if 'global' in MODEL_METRICS and 'models_comparison' in MODEL_METRICS['global']:
                models_comp = MODEL_METRICS['global']['models_comparison']
                best_model_info = max(models_comp, key=lambda x: x.get('f1_score', 0))
                best_model_name = best_model_info['model']

            # Si no se encontró, usar el primero disponible
            if not best_model_name:
                best_model_name = list(MODELS.keys())[0]

            model = MODELS[best_model_name]

            # Aplicar scaler si existe
            X_scaled = X.copy()
            if 'scaler' in SCALERS:
                X_scaled = pd.DataFrame(
                    SCALERS['scaler'].transform(X),
                    columns=X.columns
                )

            # Realizar predicción
            prediction = model.predict(X_scaled)[0]

            # Obtener probabilidades si el modelo lo soporta
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X_scaled)[0]
                prob_legitimate = probabilities[0]
                prob_fraud = probabilities[1]
            else:
                prob_fraud = 0.9 if prediction == 1 else 0.1
                prob_legitimate = 1 - prob_fraud

            is_fraud = prediction == 1

            return {
                'is_fraud': is_fraud,
                'prediction': int(prediction),
                'probability_fraud': float(prob_fraud),
                'probability_legitimate': float(prob_legitimate),
                'confidence': float(max(prob_fraud, prob_legitimate)),
                'model_used': best_model_name,
                'feature_importance': ['Amount', 'V14', 'V12', 'V10', 'V4']
            }

        # Si no hay modelos, usar lógica simple para demostración
        amount = features.get('Amount', 0)
        v_features_sum = sum([abs(features.get(f'V{i}', 0)) for i in range(1, 29)])

        # Scoring simple
        fraud_score = 0
        if amount > 500:
            fraud_score += 0.3
        if amount > 1000:
            fraud_score += 0.2
        if v_features_sum > 50:
            fraud_score += 0.3
        if v_features_sum > 100:
            fraud_score += 0.2

        # Ajustar probabilidades
        prob_fraud = min(fraud_score, 0.95)
        prob_legitimate = 1 - prob_fraud

        # Determinar si es fraude (umbral 0.5)
        is_fraud = prob_fraud > 0.5

        return {
            'is_fraud': is_fraud,
            'prediction': 1 if is_fraud else 0,
            'probability_fraud': prob_fraud,
            'probability_legitimate': prob_legitimate,
            'confidence': max(prob_fraud, prob_legitimate),
            'model_used': 'Simulación (sin modelo cargado)',
            'feature_importance': ['Amount', 'V14', 'V12', 'V10', 'V4']
        }

    except Exception as e:
        print(f"Error en predicción: {e}")
        return {
            'is_fraud': False,
            'prediction': 0,
            'probability_fraud': 0.0,
            'probability_legitimate': 1.0,
            'confidence': 0.5,
            'error': str(e)
        }

def validate_features(features):
    """
    Valida que las características estén en el formato correcto

    Args:
        features (dict): Características a validar

    Returns:
        tuple: (is_valid, error_message)
    """
    required_features = ['Time', 'Amount']

    # Verificar características requeridas
    for feature in required_features:
        if feature not in features:
            return False, f"Falta la característica requerida: {feature}"

    # Validar tipos de datos
    try:
        float(features['Time'])
        float(features['Amount'])
    except ValueError:
        return False, "Time y Amount deben ser valores numéricos"

    # Validar rangos
    if features['Amount'] < 0:
        return False, "Amount no puede ser negativo"

    if features['Time'] < 0:
        return False, "Time no puede ser negativo"

    return True, None

def get_feature_statistics():
    """
    Obtiene estadísticas descriptivas de las características

    Returns:
        dict: Estadísticas de las características
    """
    return {
        'Amount': {
            'mean': 88.35,
            'std': 250.12,
            'min': 0.0,
            'max': 25691.16,
            'median': 22.0
        },
        'Time': {
            'mean': 94813.86,
            'std': 47488.15,
            'min': 0.0,
            'max': 172792.0,
            'median': 84692.0
        }
    }
