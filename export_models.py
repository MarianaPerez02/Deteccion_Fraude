"""
Script para exportar modelos entrenados desde el notebook al formato requerido por la webapp

Ejecutar este script después de entrenar los modelos en el notebook
"""
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, auc

def export_models_from_notebook(models_dict, X_test, y_test, scaler=None, X_train=None, y_train=None):
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
    # Usar ruta absoluta basada en la ubicación de export_models.py
    # export_models.py está en la raíz del proyecto
    project_root = Path(__file__).resolve().parent
    models_dir = project_root / 'Notebook' / 'models'

    # NO crear directorios nuevos, solo verificar que existe
    if not models_dir.exists():
        raise FileNotFoundError(
            f"La carpeta {models_dir} no existe. "
            f"Por favor crea la carpeta Notebook/models en la raíz del proyecto."
        )

    print(f"Exportando modelos a: {models_dir}")

    print("=" * 60)
    print("EXPORTANDO MODELOS PARA LA WEBAPP")
    print("=" * 60)

    all_metrics = {
        'models_comparison': [],
        'roc_curves': {},
        'confusion_matrices': {},
        'feature_importance': {},
        'data_analysis': {},
        'correlation_matrix': {},
        'class_balance': {}
    }

    # Guardar predicciones para calcular fraudes detectados
    model_predictions = {}

    for model_name, model in models_dict.items():
        print(f"\nProcesando: {model_name}")

        # Guardar modelo
        model_filename = model_name.lower().replace(' ', '_')
        model_path = models_dir / f'{model_filename}_model.pkl'

        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"   OK - Modelo guardado en: {model_path}")

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
            print(f"   OK - Metricas guardadas en: {metrics_path}")

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

            # Guardar predicciones para calcular fraudes detectados después
            model_predictions[model_name] = y_pred

            # Extraer feature importance si el modelo lo soporta
            if hasattr(model, 'feature_importances_'):
                feature_names = [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount']
                importances = model.feature_importances_

                # Crear lista de (feature, importance) y ordenar
                feature_imp_list = list(zip(feature_names, importances))
                feature_imp_list.sort(key=lambda x: x[1], reverse=True)

                # Guardar top 15
                all_metrics['feature_importance'][model_name] = {
                    'features': [f[0] for f in feature_imp_list[:15]],
                    'importances': [float(f[1]) for f in feature_imp_list[:15]]
                }

            print(f"   Precision: {precision:.4f}")
            print(f"   Recall: {recall:.4f}")
            print(f"   F1-Score: {f1:.4f}")
            print(f"   ROC-AUC: {roc_auc:.4f}")

        except Exception as e:
            print(f"   Advertencia - Error calculando metricas: {e}")

    # Marcar el mejor modelo (basado en F1-Score) y calcular fraudes detectados
    best_model_name = None
    fraudes_detectados_best = 0

    if all_metrics['models_comparison']:
        best_idx = max(range(len(all_metrics['models_comparison'])),
                      key=lambda i: all_metrics['models_comparison'][i]['f1_score'])
        all_metrics['models_comparison'][best_idx]['is_best'] = True
        best_model_name = all_metrics['models_comparison'][best_idx]['model']

        # Calcular fraudes detectados por el mejor modelo
        if best_model_name in model_predictions:
            fraudes_detectados_best = int(model_predictions[best_model_name].sum())

    # Guardar scaler si existe
    if scaler is not None:
        scaler_path = models_dir / 'scaler.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"\n   OK - Scaler guardado en: {scaler_path}")

    # Calcular estadísticas de los datos de prueba
    total_test = len(y_test)
    total_fraudes_test = int(y_test.sum())
    total_legitimas_test = total_test - total_fraudes_test

    all_metrics['total_transacciones'] = total_test
    all_metrics['total_fraudes'] = total_fraudes_test
    all_metrics['total_legitimas'] = total_legitimas_test
    all_metrics['fraudes_detectados'] = fraudes_detectados_best

    # Agregar análisis de distribución de Amount y Time por clase si X_train y y_train están disponibles
    if X_train is not None and y_train is not None:
        try:
            # Convertir a DataFrame si es necesario
            if isinstance(X_train, np.ndarray):
                feature_names = [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount']
                X_train_df = pd.DataFrame(X_train, columns=feature_names)
            else:
                X_train_df = X_train

            # Separar por clase
            fraude_mask = y_train == 1
            no_fraude_mask = y_train == 0

            # Análisis de Amount
            amount_fraude = X_train_df.loc[fraude_mask, 'Amount']
            amount_no_fraude = X_train_df.loc[no_fraude_mask, 'Amount']

            all_metrics['data_analysis']['amount_distribution'] = {
                'fraude': {
                    'mean': float(np.mean(amount_fraude)),
                    'median': float(np.median(amount_fraude)),
                    'std': float(np.std(amount_fraude)),
                    'min': float(np.min(amount_fraude)),
                    'max': float(np.max(amount_fraude))
                },
                'no_fraude': {
                    'mean': float(np.mean(amount_no_fraude)),
                    'median': float(np.median(amount_no_fraude)),
                    'std': float(np.std(amount_no_fraude)),
                    'min': float(np.min(amount_no_fraude)),
                    'max': float(np.max(amount_no_fraude))
                }
            }

            # Análisis de Time
            time_fraude = X_train_df.loc[fraude_mask, 'Time']
            time_no_fraude = X_train_df.loc[no_fraude_mask, 'Time']

            all_metrics['data_analysis']['time_distribution'] = {
                'fraude': {
                    'mean': float(np.mean(time_fraude)),
                    'median': float(np.median(time_fraude)),
                    'std': float(np.std(time_fraude)),
                    'min': float(np.min(time_fraude)),
                    'max': float(np.max(time_fraude))
                },
                'no_fraude': {
                    'mean': float(np.mean(time_no_fraude)),
                    'median': float(np.median(time_no_fraude)),
                    'std': float(np.std(time_no_fraude)),
                    'min': float(np.min(time_no_fraude)),
                    'max': float(np.max(time_no_fraude))
                }
            }

            # Matriz de correlación
            corr_matrix = X_train_df.corr()
            all_metrics['correlation_matrix'] = {
                'values': corr_matrix.values.tolist(),
                'columns': corr_matrix.columns.tolist()
            }

            # Correlación con variable objetivo
            X_train_with_class = X_train_df.copy()
            X_train_with_class['Class'] = y_train
            correlations_with_class = X_train_with_class.corr()['Class'].drop('Class')
            correlations_sorted = correlations_with_class.abs().sort_values(ascending=False)

            all_metrics['data_analysis']['correlation_with_class'] = {
                'features': correlations_sorted.index.tolist()[:20],  # Top 20
                'correlations': correlations_sorted.values.tolist()[:20]
            }

            # Balanceo de clases (antes y después)
            all_metrics['class_balance'] = {
                'before': {
                    'no_fraude': int((y_train == 0).sum()),
                    'fraude': int((y_train == 1).sum())
                },
                'after': {
                    'no_fraude': int((y_train == 0).sum()),
                    'fraude': int((y_train == 1).sum())
                },
                'total_before': int(len(y_train)),
                'total_after': int(len(y_train))
            }

            print("\n   OK - Analisis de distribucion de Amount y Time agregado")
            print("   OK - Matriz de correlacion agregada")
            print("   OK - Correlacion con variable objetivo agregada")
            print("   OK - Informacion de balanceo de clases agregada")

        except Exception as e:
            print(f"\n   Advertencia - Error en analisis de datos: {e}")

    # Guardar todas las métricas
    all_metrics_path = models_dir / 'all_metrics.pkl'
    with open(all_metrics_path, 'wb') as f:
        pickle.dump(all_metrics, f)
    print(f"\n   OK - Todas las metricas guardadas en: {all_metrics_path}")

    print("\n" + "=" * 60)
    print("EXPORTACION COMPLETADA EXITOSAMENTE")
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
    # Usar ruta absoluta basada en la ubicación de export_models.py
    project_root = Path(__file__).resolve().parent
    models_dir = project_root / 'Notebook' / 'models'
    all_metrics_path = models_dir / 'all_metrics.pkl'

    if all_metrics_path.exists():
        with open(all_metrics_path, 'rb') as f:
            return pickle.load(f)
    else:
        print("Advertencia - No se encontraron metricas exportadas")
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
        'XGBoost': xgb_model,
        'LightGBM': lgbm_model  # Agregar LightGBM aqui
    }

    # Exportar (incluir X_train_resampled y y_train_resampled para analisis adicional)
    export_models_from_notebook(models, X_test, y_test, scaler, X_train_resampled, y_train_resampled)
    """)
