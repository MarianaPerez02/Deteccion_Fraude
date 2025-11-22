# 🚀 Inicio Rápido - Sistema de Detección de Fraude

## Opción 1: Usar Modelos de Demostración (Más Rápido)

Si quieres **probar la aplicación inmediatamente** sin entrenar modelos:

### 1. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 2. Crear modelos de demostración
```bash
python setup_demo_models.py
```

### 3. Ejecutar la aplicación
```bash
cd webapp
python app.py
```

### 4. Abrir en navegador
```
http://localhost:5000
```

---

## Opción 2: Usar tus Propios Modelos Entrenados

Si ya entrenaste modelos en tu notebook:

### En tu Jupyter Notebook, al final agrega:

```python
# Importar función de exportación
import sys
sys.path.append('..')
from export_models import export_models_from_notebook

# Tus modelos entrenados
models = {
    'Logistic Regression': lr_model,
    'Random Forest': rf_model,
    'XGBoost': xgb_model  # opcional
}

# Exportar todo
export_models_from_notebook(
    models_dict=models,
    X_test=X_test,
    y_test=y_test,
    scaler=scaler
)
```

Luego ejecuta:
```bash
cd webapp
python app.py
```

---

## ✅ Verificar que funciona

Cuando ejecutes `python app.py`, deberías ver:

```
📦 Cargando modelos entrenados...
   ✅ Logistic Regression cargado
   ✅ Random Forest cargado
   ✅ Scaler cargado
   ✅ Métricas globales cargadas

✅ Total de modelos cargados: 2
```

Si ves esta advertencia:
```
⚠️ Advertencia: No se encontraron modelos entrenados
```

Significa que necesitas ejecutar `python setup_demo_models.py` primero.

---

## 📊 Estructura de la Carpeta Models

Después de ejecutar `setup_demo_models.py` o exportar tus modelos:

```
models/
├── scaler.pkl                        # StandardScaler para normalización
├── logistic_regression_model.pkl     # Modelo Logistic Regression
├── logistic_regression_metrics.pkl   # Métricas del modelo
├── random_forest_model.pkl           # Modelo Random Forest
├── random_forest_metrics.pkl         # Métricas del modelo
└── all_metrics.pkl                   # Métricas globales y curvas ROC
```

---

## 🔍 Solución de Problemas

### Error: "No module named 'sklearn'"
```bash
pip install scikit-learn
```

### Error: "No module named 'flask'"
```bash
pip install -r requirements.txt
```

### Los modelos no se cargan
1. Verifica que la carpeta `models/` exista
2. Ejecuta `python setup_demo_models.py`
3. Reinicia la aplicación Flask

### Puerto 5000 en uso
Edita `webapp/app.py` y cambia:
```python
app.run(debug=True, host='0.0.0.0', port=8080)
```

---

## 📝 Notas Importantes

- **Modelos de demostración**: Son modelos ficticios entrenados con datos sintéticos. Solo sirven para testing de la webapp.
- **Modelos reales**: Debes entrenarlos en tu notebook con el dataset real de fraude.
- **Datos**: Los modelos de demo NO detectan fraude real, solo demuestran la funcionalidad.

---

**¡Listo!** Ahora tienes la aplicación funcionando sin la advertencia.
