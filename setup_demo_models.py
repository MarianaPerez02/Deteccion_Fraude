"""
Script para crear modelos de demostración si no tienes datos para entrenar
Este script crea modelos ficticios solo para testing de la webapp
"""
import pickle
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)

def create_demo_models():
    """Crea modelos de demostración para testing"""

    print("=" * 60)
    print("CREANDO MODELOS DE DEMOSTRACIÓN")
    print("=" * 60)

    # Crear carpeta models
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)

    # Crear datos sintéticos para entrenar modelos demo
    print("\nGenerando datos sinteticos...")
    X, y = make_classification(
        n_samples=1000,
        n_features=30,
        n_informative=20,
        n_redundant=5,
        n_classes=2,
        weights=[0.98, 0.02],
        random_state=42
    )

    # Dividir en train/test
    print("Dividiendo datos en train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Crear scaler
    print("Creando scaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Guardar scaler
    scaler_path = models_dir / 'scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"   OK - Scaler guardado en: {scaler_path}")

    # Entrenar Logistic Regression
    print("\nEntrenando Logistic Regression...")
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)

    lr_path = models_dir / 'logistic_regression_model.pkl'
    with open(lr_path, 'wb') as f:
        pickle.dump(lr_model, f)
    print(f"   OK - Modelo guardado en: {lr_path}")

    # Calcular métricas REALES para LR
    lr_pred = lr_model.predict(X_test_scaled)
    lr_pred_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
    lr_metrics = {
        'precision': precision_score(y_test, lr_pred),
        'recall': recall_score(y_test, lr_pred),
        'f1_score': f1_score(y_test, lr_pred),
        'roc_auc': roc_auc_score(y_test, lr_pred_proba)
    }
    lr_metrics_path = models_dir / 'logistic_regression_metrics.pkl'
    with open(lr_metrics_path, 'wb') as f:
        pickle.dump(lr_metrics, f)

    # Entrenar Random Forest
    print("\nEntrenando Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)

    rf_path = models_dir / 'random_forest_model.pkl'
    with open(rf_path, 'wb') as f:
        pickle.dump(rf_model, f)
    print(f"   OK - Modelo guardado en: {rf_path}")

    # Calcular métricas REALES para RF
    rf_pred = rf_model.predict(X_test_scaled)
    rf_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
    rf_metrics = {
        'precision': precision_score(y_test, rf_pred),
        'recall': recall_score(y_test, rf_pred),
        'f1_score': f1_score(y_test, rf_pred),
        'roc_auc': roc_auc_score(y_test, rf_pred_proba)
    }
    rf_metrics_path = models_dir / 'random_forest_metrics.pkl'
    with open(rf_metrics_path, 'wb') as f:
        pickle.dump(rf_metrics, f)

    # Entrenar Decision Tree
    print("\nEntrenando Decision Tree...")
    dt_model = DecisionTreeClassifier(random_state=42, max_depth=10)
    dt_model.fit(X_train_scaled, y_train)

    dt_path = models_dir / 'decision_tree_model.pkl'
    with open(dt_path, 'wb') as f:
        pickle.dump(dt_model, f)
    print(f"   OK - Modelo guardado en: {dt_path}")

    # Calcular métricas REALES para DT
    dt_pred = dt_model.predict(X_test_scaled)
    dt_pred_proba = dt_model.predict_proba(X_test_scaled)[:, 1]
    dt_metrics = {
        'precision': precision_score(y_test, dt_pred),
        'recall': recall_score(y_test, dt_pred),
        'f1_score': f1_score(y_test, dt_pred),
        'roc_auc': roc_auc_score(y_test, dt_pred_proba)
    }
    dt_metrics_path = models_dir / 'decision_tree_metrics.pkl'
    with open(dt_metrics_path, 'wb') as f:
        pickle.dump(dt_metrics, f)

    # Entrenar Gradient Boosting
    print("\nEntrenando Gradient Boosting...")
    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_model.fit(X_train_scaled, y_train)

    gb_path = models_dir / 'gradient_boosting_model.pkl'
    with open(gb_path, 'wb') as f:
        pickle.dump(gb_model, f)
    print(f"   OK - Modelo guardado en: {gb_path}")

    # Calcular métricas REALES para GB
    gb_pred = gb_model.predict(X_test_scaled)
    gb_pred_proba = gb_model.predict_proba(X_test_scaled)[:, 1]
    gb_metrics = {
        'precision': precision_score(y_test, gb_pred),
        'recall': recall_score(y_test, gb_pred),
        'f1_score': f1_score(y_test, gb_pred),
        'roc_auc': roc_auc_score(y_test, gb_pred_proba)
    }
    gb_metrics_path = models_dir / 'gradient_boosting_metrics.pkl'
    with open(gb_metrics_path, 'wb') as f:
        pickle.dump(gb_metrics, f)

    # Entrenar SVM
    print("\nEntrenando SVM...")
    svm_model = SVC(kernel='rbf', probability=True, random_state=42)
    svm_model.fit(X_train_scaled, y_train)

    svm_path = models_dir / 'svm_model.pkl'
    with open(svm_path, 'wb') as f:
        pickle.dump(svm_model, f)
    print(f"   OK - Modelo guardado en: {svm_path}")

    # Calcular métricas REALES para SVM
    svm_pred = svm_model.predict(X_test_scaled)
    svm_pred_proba = svm_model.predict_proba(X_test_scaled)[:, 1]
    svm_metrics = {
        'precision': precision_score(y_test, svm_pred),
        'recall': recall_score(y_test, svm_pred),
        'f1_score': f1_score(y_test, svm_pred),
        'roc_auc': roc_auc_score(y_test, svm_pred_proba)
    }
    svm_metrics_path = models_dir / 'svm_metrics.pkl'
    with open(svm_metrics_path, 'wb') as f:
        pickle.dump(svm_metrics, f)

    # Crear métricas globales
    print("\nCreando metricas globales...")

    all_metrics = {
        'total_transacciones': 284807,
        'total_fraudes': 492,
        'total_legitimas': 284315,
        'fraudes_detectados': 456,
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
                'model': 'Decision Tree',
                'precision': 0.8654,
                'recall': 0.8521,
                'f1_score': 0.8587,
                'roc_auc': 0.9123,
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
                'model': 'Gradient Boosting',
                'precision': 0.9344,
                'recall': 0.9187,
                'f1_score': 0.9265,
                'roc_auc': 0.9789,
                'is_best': False
            },
            {
                'model': 'SVM',
                'precision': 0.9012,
                'recall': 0.8867,
                'f1_score': 0.8939,
                'roc_auc': 0.9523,
                'is_best': False
            }
        ],
        'roc_curves': {
            'Logistic Regression': {
                'fpr': np.linspace(0, 1, 100).tolist(),
                'tpr': (np.linspace(0, 1, 100) ** 0.5).tolist(),
                'auc': 0.9412
            },
            'Decision Tree': {
                'fpr': np.linspace(0, 1, 100).tolist(),
                'tpr': (np.linspace(0, 1, 100) ** 0.6).tolist(),
                'auc': 0.9123
            },
            'Random Forest': {
                'fpr': np.linspace(0, 1, 100).tolist(),
                'tpr': (np.linspace(0, 1, 100) ** 0.3).tolist(),
                'auc': 0.9856
            },
            'Gradient Boosting': {
                'fpr': np.linspace(0, 1, 100).tolist(),
                'tpr': (np.linspace(0, 1, 100) ** 0.4).tolist(),
                'auc': 0.9789
            },
            'SVM': {
                'fpr': np.linspace(0, 1, 100).tolist(),
                'tpr': (np.linspace(0, 1, 100) ** 0.45).tolist(),
                'auc': 0.9523
            }
        },
        'confusion_matrices': {
            'Logistic Regression': [[56850, 12], [52, 440]],
            'Decision Tree': [[56840, 22], [72, 420]],
            'Random Forest': [[56858, 4], [36, 456]],
            'Gradient Boosting': [[56854, 8], [40, 452]],
            'SVM': [[56848, 14], [56, 436]]
        }
    }

    all_metrics_path = models_dir / 'all_metrics.pkl'
    with open(all_metrics_path, 'wb') as f:
        pickle.dump(all_metrics, f)
    print(f"   OK - Metricas globales guardadas en: {all_metrics_path}")

    print("\n" + "=" * 60)
    print("MODELOS DE DEMOSTRACION CREADOS EXITOSAMENTE")
    print("=" * 60)
    print("\nArchivos creados:")
    print("  - models/scaler.pkl")
    print("  - models/logistic_regression_model.pkl")
    print("  - models/decision_tree_model.pkl")
    print("  - models/random_forest_model.pkl")
    print("  - models/gradient_boosting_model.pkl")
    print("  - models/svm_model.pkl")
    print("  - models/all_metrics.pkl")
    print(f"\nTotal de modelos: 5")
    print("\nAhora puedes ejecutar la webapp:")
    print("  cd webapp")
    print("  python app.py")
    print("=" * 60)

if __name__ == "__main__":
    create_demo_models()
