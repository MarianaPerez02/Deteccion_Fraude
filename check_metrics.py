"""
Script para verificar qué datos contiene all_metrics.pkl
"""
import pickle
from pathlib import Path

project_root = Path(__file__).resolve().parent
models_dir = project_root / 'Notebook' / 'models'
all_metrics_path = models_dir / 'all_metrics.pkl'

print("=" * 60)
print("VERIFICANDO CONTENIDO DE all_metrics.pkl")
print("=" * 60)

if not all_metrics_path.exists():
    print(f"ERROR: No existe el archivo {all_metrics_path}")
else:
    with open(all_metrics_path, 'rb') as f:
        metrics = pickle.load(f)

    print(f"\nArchivo encontrado en: {all_metrics_path}")
    print(f"\nClaves principales en metrics:")
    for key in metrics.keys():
        print(f"  - {key}")

    print("\n" + "=" * 60)
    print("VERIFICANDO DATOS ESPECÍFICOS")
    print("=" * 60)

    # Verificar correlation_matrix
    print("\n1. correlation_matrix:")
    if 'correlation_matrix' in metrics:
        corr_data = metrics['correlation_matrix']
        if corr_data and 'values' in corr_data and 'columns' in corr_data:
            print(f"   OK - Tiene {len(corr_data['columns'])} columnas")
            print(f"   OK - Matriz de {len(corr_data['values'])}x{len(corr_data['values'][0])}")
        else:
            print("   FALTA - Estructura incorrecta o vacía")
    else:
        print("   FALTA - No existe esta clave")

    # Verificar data_analysis
    print("\n2. data_analysis:")
    if 'data_analysis' in metrics:
        data_analysis = metrics['data_analysis']
        print(f"   Claves en data_analysis: {list(data_analysis.keys())}")

        # time_distribution
        if 'time_distribution' in data_analysis:
            print("   OK - time_distribution existe")
        else:
            print("   FALTA - time_distribution no existe")

        # correlation_with_class
        if 'correlation_with_class' in data_analysis:
            corr_class = data_analysis['correlation_with_class']
            if 'features' in corr_class and 'correlations' in corr_class:
                print(f"   OK - correlation_with_class tiene {len(corr_class['features'])} features")
            else:
                print("   FALTA - correlation_with_class estructura incorrecta")
        else:
            print("   FALTA - correlation_with_class no existe")
    else:
        print("   FALTA - No existe data_analysis")

    # Verificar class_balance
    print("\n3. class_balance:")
    if 'class_balance' in metrics:
        balance = metrics['class_balance']
        print(f"   Claves en class_balance: {list(balance.keys())}")
        if 'before' in balance and 'after' in balance:
            print(f"   OK - Tiene 'before' y 'after'")
            print(f"   Before - No fraude: {balance['before'].get('no_fraude', 'N/A')}, Fraude: {balance['before'].get('fraude', 'N/A')}")
            print(f"   After - No fraude: {balance['after'].get('no_fraude', 'N/A')}, Fraude: {balance['after'].get('fraude', 'N/A')}")
        else:
            print("   FALTA - No tiene 'before' y/o 'after'")
    else:
        print("   FALTA - No existe class_balance")

    # Verificar feature_importance
    print("\n4. feature_importance:")
    if 'feature_importance' in metrics:
        feat_imp = metrics['feature_importance']
        print(f"   Modelos con feature importance: {list(feat_imp.keys())}")
    else:
        print("   FALTA - No existe feature_importance")

    print("\n" + "=" * 60)
