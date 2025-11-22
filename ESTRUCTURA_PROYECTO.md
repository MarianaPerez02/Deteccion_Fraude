# 📁 Estructura del Proyecto - Sistema de Detección de Fraude

## 🏗️ Arquitectura del Proyecto

```
Deteccion_Fraude/
│
├── 📂 webapp/                          # Aplicación web Flask
│   ├── app.py                          # Aplicación principal (rutas y lógica)
│   │
│   ├── 📂 templates/                   # Templates HTML
│   │   ├── base.html                   # Template base (navbar, footer)
│   │   ├── home.html                   # Dashboard principal
│   │   ├── comparacion.html            # Comparación de modelos
│   │   ├── prediccion.html             # Predicción individual
│   │   └── matriz_confusion.html       # Matrices de confusión
│   │
│   ├── 📂 static/                      # Archivos estáticos
│   │   ├── css/
│   │   │   └── style.css               # Estilos personalizados
│   │   └── js/                         # JavaScript (vacío por ahora)
│   │
│   └── 📂 utils/                       # Utilidades
│       ├── __init__.py
│       └── model_utils.py              # Funciones para ML
│
├── 📂 Notebook/                        # Jupyter Notebooks
│   └── fraude_Tarjetas.ipynb          # Análisis y entrenamiento
│
├── 📂 models/                          # Modelos entrenados (crear)
│   ├── random_forest_model.pkl         # Modelo Random Forest
│   ├── logistic_regression_model.pkl   # Modelo Logistic Regression
│   ├── xgboost_model.pkl              # Modelo XGBoost
│   ├── scaler.pkl                     # Scaler para normalización
│   └── all_metrics.pkl                # Métricas globales
│
├── 📄 export_models.py                 # Script para exportar modelos
├── 📄 requirements.txt                 # Dependencias Python
├── 📄 README.md                        # Documentación completa
├── 📄 GUIA_RAPIDA.md                  # Guía de inicio rápido
├── 📄 ESTRUCTURA_PROYECTO.md          # Este archivo
├── 📄 run.bat                         # Script de inicio (Windows)
└── 📄 .gitignore                      # Archivos a ignorar en Git

```

---

## 🔄 Flujo de Trabajo

### 1. Desarrollo y Entrenamiento
```
Notebook (fraude_Tarjetas.ipynb)
    ↓
Entrenar modelos
    ↓
Ejecutar export_models.py
    ↓
Modelos guardados en /models/
```

### 2. Ejecución de la Webapp
```
run.bat / python app.py
    ↓
Flask inicia en puerto 5000
    ↓
Carga modelos desde /models/
    ↓
Webapp lista para usar
```

### 3. Uso de la Aplicación
```
Usuario accede → http://localhost:5000
    ↓
┌─────────────────┬────────────────┬──────────────────┬────────────────┐
│   Dashboard     │  Comparación   │   Predicción     │    Matrices    │
│   (KPIs)        │  (Modelos)     │  (Individual)    │  (Confusión)   │
└─────────────────┴────────────────┴──────────────────┴────────────────┘
```

---

## 📋 Componentes Principales

### 🎨 Frontend (Templates HTML)

#### 1. **base.html**
- Template base con navbar y footer
- Importa Bootstrap 5 y Plotly
- Estructura común para todas las páginas

#### 2. **home.html**
- Dashboard principal
- KPIs: Total transacciones, fraudes, precisión, recall
- Gráfico de distribución
- Links rápidos a otras páginas

#### 3. **comparacion.html**
- Tabla comparativa de modelos
- Gráfico de barras con métricas
- Curvas ROC interactivas
- Explicación de métricas

#### 4. **prediccion.html**
- Formulario de entrada
- Campos para Time, Amount, V1-V28
- Botón para generar datos de ejemplo
- Visualización de resultados con probabilidades

#### 5. **matriz_confusion.html**
- Heatmaps de matrices de confusión
- Explicación de TP, TN, FP, FN
- Fórmulas de métricas

---

### ⚙️ Backend (Python)

#### 1. **app.py** - Servidor Flask
```python
Rutas principales:
- /                    → Dashboard (home.html)
- /comparacion         → Comparación de modelos
- /prediccion          → Formulario de predicción
- /matriz-confusion    → Matrices de confusión
- /api/predict         → API REST para predicciones
```

#### 2. **model_utils.py** - Utilidades ML
```python
Funciones principales:
- load_model()         → Carga modelos desde /models/
- get_model_metrics()  → Obtiene métricas
- predict_fraud()      → Realiza predicciones
- save_model()         → Guarda modelos
```

#### 3. **export_models.py** - Exportación
```python
- export_models_from_notebook()  → Exporta modelos del notebook
- Calcula y guarda métricas
- Genera curvas ROC
- Crea matrices de confusión
```

---

## 🎨 Diseño Visual

### Paleta de Colores
- **Azul (#007bff)**: Principal (navbar, headers)
- **Verde (#28a745)**: Transacciones legítimas
- **Rojo (#dc3545)**: Transacciones fraudulentas
- **Amarillo (#ffc107)**: Advertencias
- **Gris (#f8f9fa)**: Backgrounds

### Componentes Bootstrap
- Cards con shadow y hover effects
- Badges para métricas
- Alerts para mensajes
- Progress bars para confianza
- Accordion para features adicionales

---

## 📊 Visualizaciones Plotly

### 1. **Dashboard**
- Gráfico de barras: Distribución Legítimas vs Fraudulentas

### 2. **Comparación**
- Gráfico de barras agrupadas: Métricas por modelo
- Curvas ROC: Múltiples modelos superpuestos

### 3. **Predicción**
- Gráfico de barras: Probabilidades (Legítima vs Fraudulenta)

### 4. **Matrices**
- Heatmaps: Una por cada modelo

---

## 🔌 API REST

### Endpoint de Predicción
```http
POST /api/predict
Content-Type: application/json

{
  "Time": 12345,
  "Amount": 150.50,
  "V1": -1.359,
  ...
  "V28": 0.133
}
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

## 🔧 Configuración y Extensión

### Agregar nuevo modelo
1. Entrena el modelo en el notebook
2. Agrégalo al diccionario en `export_models_from_notebook()`
3. Ejecuta el script de exportación
4. Reinicia la webapp

### Cambiar estilos
- Edita `webapp/static/css/style.css`
- Colores, fuentes, animaciones

### Agregar nueva página
1. Crea ruta en `app.py`
2. Crea template en `templates/`
3. Agrega link en `base.html` navbar

### Personalizar gráficos
- Modifica las funciones en `app.py` que crean figuras de Plotly
- Documentación: https://plotly.com/python/

---

## 📦 Dependencias Principales

| Librería | Versión | Propósito |
|----------|---------|-----------|
| Flask | 3.0.0 | Framework web |
| Pandas | 2.1.4 | Manipulación de datos |
| NumPy | 1.26.2 | Cálculos numéricos |
| Scikit-learn | 1.3.2 | Machine Learning |
| Plotly | 5.18.0 | Visualizaciones |

---

## 🚀 Modo de Desarrollo vs Producción

### Desarrollo (Actual)
- `debug=True` en Flask
- Datos de ejemplo si no hay modelos
- Recarga automática de código
- Errores detallados en navegador

### Producción (Futuro)
- Usar Gunicorn o Waitress
- `debug=False`
- Variables de entorno (.env)
- Logging apropiado
- HTTPS
- Rate limiting en API

---

## 📝 Próximos Pasos Sugeridos

1. ✅ Entrenar modelos en el notebook
2. ✅ Exportar modelos con `export_models.py`
3. ✅ Ejecutar webapp y probar
4. 🔲 Agregar autenticación de usuarios
5. 🔲 Dashboard de monitoreo en tiempo real
6. 🔲 Exportar reportes PDF
7. 🔲 Integrar con base de datos
8. 🔲 Deploy en cloud (Heroku, AWS, Azure)

---

**Creado para el proyecto de Ciencia de Datos - Detección de Fraude**
