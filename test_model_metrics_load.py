"""
Script para verificar qué contiene MODEL_METRICS['global']
"""
import pickle
from pathlib import Path

project_root = Path(__file__).resolve().parent
models_dir = project_root / 'Notebook' / 'models'
all_metrics_path = models_dir / 'all_metrics.pkl'

print("=" * 60)
print("VERIFICANDO MODEL_METRICS['global']")
print("=" * 60)

# Cargar directamente el archivo
with open(all_metrics_path, 'rb') as f:
    global_metrics = pickle.load(f)

print("\nClaves en el archivo all_metrics.pkl:")
for key in global_metrics.keys():
    print(f"  - {key}")

# Simular lo que hace get_model_metrics()
metrics = global_metrics.copy()

print("\nDespués de .copy():")
for key in metrics.keys():
    print(f"  - {key}")

# Verificar específicamente las claves que necesitamos
print("\n" + "=" * 60)
print("VERIFICANDO CLAVES NECESARIAS")
print("=" * 60)

required_keys = [
    'feature_importance',
    'data_analysis',
    'correlation_matrix',
    'class_balance',
    'models_comparison'
]

for key in required_keys:
    if key in metrics:
        print(f"  OK - {key}")
        if key == 'data_analysis':
            print(f"       Sub-claves: {list(metrics[key].keys())}")
    else:
        print(f"  FALTA - {key}")
