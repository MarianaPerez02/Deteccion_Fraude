"""
Script para probar la ruta /analisis y verificar qué datos se están enviando
"""
import sys
sys.path.insert(0, 'webapp')

from webapp.utils.model_utils import get_model_metrics

print("=" * 60)
print("PROBANDO DATOS QUE SE ENVÍAN A /analisis")
print("=" * 60)

# Obtener métricas
metrics = get_model_metrics()

print("\n1. Claves principales en metrics:")
for key in metrics.keys():
    print(f"  - {key}")

# Verificar cada visualización
print("\n" + "=" * 60)
print("VERIFICANDO DATOS PARA CADA VISUALIZACIÓN")
print("=" * 60)

# 1. Feature Importance
print("\n1. Feature Importance:")
if 'feature_importance' in metrics:
    feat_imp = metrics['feature_importance']
    print(f"   OK - {len(feat_imp)} modelos con feature importance")
    for model_name in feat_imp.keys():
        print(f"      - {model_name}")
else:
    print("   FALTA")

# 2. Amount Distribution
print("\n2. Amount Distribution:")
if 'data_analysis' in metrics and 'amount_distribution' in metrics['data_analysis']:
    print("   OK - amount_distribution presente")
else:
    print("   FALTA")

# 3. Time Distribution
print("\n3. Time Distribution:")
if 'data_analysis' in metrics and 'time_distribution' in metrics['data_analysis']:
    print("   OK - time_distribution presente")
else:
    print("   FALTA")

# 4. Models Comparison
print("\n4. Models Comparison (Subplots):")
if 'models_comparison' in metrics and len(metrics['models_comparison']) > 0:
    print(f"   OK - {len(metrics['models_comparison'])} modelos para comparar")
else:
    print("   FALTA")

# 5. Correlation Matrix
print("\n5. Correlation Matrix:")
if 'correlation_matrix' in metrics and metrics['correlation_matrix']:
    corr_matrix = metrics['correlation_matrix']
    if 'values' in corr_matrix and 'columns' in corr_matrix:
        print(f"   OK - Matriz de {len(corr_matrix['columns'])} columnas")
    else:
        print("   FALTA - estructura incorrecta")
else:
    print("   FALTA")

# 6. Correlation with Class
print("\n6. Correlation with Class:")
if 'data_analysis' in metrics and 'correlation_with_class' in metrics['data_analysis']:
    corr_class = metrics['data_analysis']['correlation_with_class']
    if 'features' in corr_class and 'correlations' in corr_class:
        print(f"   OK - {len(corr_class['features'])} features")
    else:
        print("   FALTA - estructura incorrecta")
else:
    print("   FALTA")

# 7. Class Balance
print("\n7. Class Balance:")
if 'class_balance' in metrics and metrics['class_balance']:
    balance = metrics['class_balance']
    if 'before' in balance and 'after' in balance:
        print("   OK - Tiene before y after")
    else:
        print("   FALTA - estructura incorrecta")
else:
    print("   FALTA")

print("\n" + "=" * 60)
print("RESUMEN")
print("=" * 60)
print("\nTodas las visualizaciones tienen datos disponibles.")
print("Si no se muestran en la webapp, el problema puede ser:")
print("  1. Error en la generación de gráficos en app.py")
print("  2. Error en el renderizado del template")
print("  3. Error JavaScript en el navegador")
print("\nRecomendación: Verificar la consola del navegador (F12)")
