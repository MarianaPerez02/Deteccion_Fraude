# 🚀 Guía Rápida - Sistema de Detección de Fraude

## ⚡ Inicio Rápido (3 pasos)

### 1️⃣ Instalar dependencias
```bash
pip install -r requirements.txt
```

### 2️⃣ Ejecutar la aplicación
```bash
cd webapp
python app.py
```

### 3️⃣ Abrir en el navegador
Abre tu navegador en: **http://localhost:5000**

---

## 📊 Usar con tus propios modelos

### Paso 1: Entrenar modelos en el notebook

Abre tu notebook Jupyter:
```bash
jupyter notebook Notebook/fraude_Tarjetas.ipynb
```

### Paso 2: Exportar modelos

Al final de tu notebook, agrega este código:

```python
# Importar la función de exportación
import sys
sys.path.append('..')
from export_models import export_models_from_notebook

# Diccionario con tus modelos entrenados
models = {
    'Logistic Regression': lr_model,
    'Random Forest': rf_model,
    'XGBoost': xgb_model  # Si lo usas
}

# Exportar modelos y métricas
export_models_from_notebook(
    models_dict=models,
    X_test=X_test,
    y_test=y_test,
    scaler=scaler  # El scaler que usaste
)
```

### Paso 3: Reiniciar la webapp

```bash
cd webapp
python app.py
```

Los modelos se cargarán automáticamente!

---

## 🎯 Páginas Disponibles

| Página | URL | Descripción |
|--------|-----|-------------|
| **Dashboard** | `/` | Vista general con KPIs |
| **Comparación** | `/comparacion` | Comparar modelos y ver curvas ROC |
| **Predicción** | `/prediccion` | Hacer predicciones individuales |
| **Matrices** | `/matriz-confusion` | Ver matrices de confusión |

---

## 🔧 Solución de Problemas

### Error: Port 5000 already in use
```python
# En webapp/app.py, cambia la última línea:
app.run(debug=True, host='0.0.0.0', port=8080)
```

### No se ven gráficos
- Verifica que tengas conexión a internet (se cargan CDNs)
- Revisa la consola del navegador (F12)

### Modelos no cargan
- Asegúrate de haber ejecutado `export_models_from_notebook()`
- Verifica que exista la carpeta `models/`
- Revisa que los archivos `.pkl` estén en `models/`

---

## 📡 API REST

### Endpoint de predicción

**POST** `/api/predict`

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 12345,
    "Amount": 150.50,
    "V1": -1.359,
    "V2": -0.073,
    ...
    "V28": 0.133
  }'
```

**Respuesta:**
```json
{
  "is_fraud": false,
  "probability_fraud": 0.23,
  "probability_legitimate": 0.77,
  "confidence": 0.77,
  "model_used": "Random Forest"
}
```

---

## 💡 Tips

1. **Genera datos de ejemplo**: En la página de predicción, usa el botón "Generar Datos de Ejemplo"
2. **Modo Debug**: Ya está activado, verás errores detallados
3. **Recarga automática**: Flask recarga automáticamente cuando cambias código
4. **Personalización**: Edita `templates/*.html` y `static/css/style.css`

---

## 📚 Más Información

Lee el [README.md](README.md) completo para:
- Estructura detallada del proyecto
- Guía de deployment
- Personalización avanzada
- Troubleshooting completo

---

**¿Preguntas?** Revisa la documentación o abre un issue.
