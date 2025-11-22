"""
Script para exportar modelos entrenados desde el notebook al formato requerido por la webapp

Ejecutar este script después de entrenar los modelos en el notebook
"""
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, auc

def export_models_from_notebook(models_dict, X_test, y_test, scaler=None):
    """
    Exporta modelos entrenados y sus métricas para la webapp

    Args:
        models_dict (dict): Diccionario con modelos entrenados {nombre: modelo}
        X_test: Datos de prueba
        y_test: Etiquetas de prueba
        scaler: Scaler usado para normalización (opcional)

    Ejemplo de uso en el notebook:
        from export_models import export_models_from_notebook

        models = {
            'Logistic Regression': lr_model,
            'Random Forest': rf_model,
            'XGBoost': xgb_model
        }

        export_models_from_notebook(models, X_test, y_test, scaler)
    """
    # Crear carpeta de modelos si no existe
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("EXPORTANDO MODELOS PARA LA WEBAPP")
    print("=" * 60)

    all_metrics = {
        'models_comparison': [],
        'roc_curves': {},
        'confusion_matrices': {}
    }

    for model_name, model in models_dict.items():
        print(f"\n📦 Procesando: {model_name}")

        # Guardar modelo
        model_filename = model_name.lower().replace(' ', '_')
        model_path = models_dir / f'{model_filename}_model.pkl'

        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"   ✅ Modelo guardado en: {model_path}")

        # Calcular métricas
        try:
            from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred

            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_proba)

            # Guardar métricas
            metrics = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc
            }

            metrics_path = models_dir / f'{model_filename}_metrics.pkl'
            with open(metrics_path, 'wb') as f:
                pickle.dump(metrics, f)
            print(f"   ✅ Métricas guardadas en: {metrics_path}")

            # Agregar a comparación
            all_metrics['models_comparison'].append({
                'model': model_name,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'is_best': False  # Se marcará después
            })

            # Curva ROC
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            all_metrics['roc_curves'][model_name] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'auc': roc_auc
            }

            # Matriz de confusión
            cm = confusion_matrix(y_test, y_pred)
            all_metrics['confusion_matrices'][model_name] = cm.tolist()

            print(f"   📊 Precision: {precision:.4f}")
            print(f"   📊 Recall: {recall:.4f}")
            print(f"   📊 F1-Score: {f1:.4f}")
            print(f"   📊 ROC-AUC: {roc_auc:.4f}")

        except Exception as e:
            print(f"   ⚠️ Error calculando métricas: {e}")

    # Marcar el mejor modelo (basado en F1-Score)
    if all_metrics['models_comparison']:
        best_idx = max(range(len(all_metrics['models_comparison'])),
                      key=lambda i: all_metrics['models_comparison'][i]['f1_score'])
        all_metrics['models_comparison'][best_idx]['is_best'] = True

    # Guardar scaler si existe
    if scaler is not None:
        scaler_path = models_dir / 'scaler.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"\n   ✅ Scaler guardado en: {scaler_path}")

    # Guardar todas las métricas
    all_metrics_path = models_dir / 'all_metrics.pkl'
    with open(all_metrics_path, 'wb') as f:
        pickle.dump(all_metrics, f)
    print(f"\n   ✅ Todas las métricas guardadas en: {all_metrics_path}")

    print("\n" + "=" * 60)
    print("✅ EXPORTACIÓN COMPLETADA EXITOSAMENTE")
    print("=" * 60)
    print("\nAhora puedes ejecutar la webapp con:")
    print("  cd webapp")
    print("  python app.py")
    print("=" * 60)

def load_exported_metrics():
    """
    Carga las métricas exportadas para usarlas en la webapp

    Returns:
        dict: Métricas de todos los modelos
    """
    models_dir = Path('models')
    all_metrics_path = models_dir / 'all_metrics.pkl'

    if all_metrics_path.exists():
        with open(all_metrics_path, 'rb') as f:
            return pickle.load(f)
    else:
        print("⚠️ No se encontraron métricas exportadas")
        return None

if __name__ == "__main__":
    print("Este script debe ejecutarse desde un notebook después de entrenar los modelos")
    print("\nEjemplo de uso:")
    print("""
    from export_models import export_models_from_notebook

    # Diccionario con tus modelos entrenados
    models = {
        'Logistic Regression': lr_model,
        'Random Forest': rf_model,
        'XGBoost': xgb_model
    }

    # Exportar
    export_models_from_notebook(models, X_test, y_test, scaler)
    """)
