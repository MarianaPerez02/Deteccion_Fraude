# Sistema de Detección de Fraude con Tarjetas de Crédito

Aplicación web desarrollada con Flask y Plotly para la detección de transacciones fraudulentas mediante Machine Learning.

## Características

- **Dashboard Interactivo**: Visualización de KPIs y métricas principales
- **Comparación de Modelos**: Análisis comparativo de diferentes algoritmos de ML
- **Predicción en Tiempo Real**: Sistema de predicción individual de transacciones
- **Visualizaciones Interactivas**: Gráficos dinámicos con Plotly
- **Matrices de Confusión**: Análisis detallado del rendimiento de cada modelo

## Tecnologías Utilizadas

- **Backend**: Flask 3.0
- **Frontend**: HTML5, CSS3, Bootstrap 5
- **Visualización**: Plotly.js
- **Machine Learning**: Scikit-learn
- **Análisis de Datos**: Pandas, NumPy

## Estructura del Proyecto

```
Deteccion_Fraude/
│
├── webapp/
│   ├── app.py                 # Aplicación principal Flask
│   ├── templates/             # Plantillas HTML
│   │   ├── base.html
│   │   ├── home.html
│   │   ├── comparacion.html
│   │   ├── prediccion.html
│   │   └── matriz_confusion.html
│   ├── static/                # Archivos estáticos
│   │   └── css/
│   │       └── style.css
│   └── utils/                 # Utilidades
│       ├── __init__.py
│       └── model_utils.py     # Funciones para modelos ML
│
├── Notebook/                  # Jupyter notebooks de análisis
│   └── fraude_Tarjetas.ipynb
│
├── models/                    # Modelos entrenados (crear esta carpeta)
│
├── requirements.txt           # Dependencias del proyecto
└── README.md                  # Este archivo
```

## Instalación

### 1. Clonar el repositorio o navegar al directorio del proyecto

```bash
cd "c:\Users\yorie\Downloads\Ciencia de Datos\corte 3\Proyecto\Deteccion_Fraude"
```

### 2. Crear un entorno virtual (recomendado)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

## Uso

### 1. Entrenar los Modelos (Opcional)

Antes de ejecutar la aplicación, es recomendable entrenar tus modelos usando el notebook:

```bash
jupyter notebook Notebook/fraude_Tarjetas.ipynb
```

Después de entrenar los modelos, guárdalos en la carpeta `models/` usando el siguiente código en tu notebook:

```python
from webapp.utils.model_utils import save_model

# Ejemplo: Guardar un modelo Random Forest
save_model(
    model=rf_model,
    model_name='Random Forest',
    scaler=scaler,
    metrics=metrics
)
```

### 2. Ejecutar la Aplicación

```bash
cd webapp
python app.py
```

La aplicación estará disponible en: [http://localhost:5000](http://localhost:5000)

### 3. Navegar por la Aplicación

- **Inicio** (`/`): Dashboard principal con KPIs y overview
- **Comparación** (`/comparacion`): Comparación de modelos con gráficos ROC
- **Predicción** (`/prediccion`): Realizar predicciones individuales
- **Matrices** (`/matriz-confusion`): Ver matrices de confusión de cada modelo

## API Endpoints

La aplicación también expone un endpoint API para predicciones:

### POST /api/predict

Realiza una predicción de fraude.

**Request Body:**
```json
{
  "Time": 12345,
  "Amount": 150.50,
  "V1": -1.359,
  "V2": -0.073,
  ...
  "V28": 0.133
}
```

**Response:**
```json
{
  "is_fraud": false,
  "probability_fraud": 0.23,
  "probability_legitimate": 0.77,
  "confidence": 0.77,
  "model_used": "Random Forest"
}
```

## Integración con tus Modelos

Para integrar tus propios modelos entrenados:

1. **Entrenar tus modelos** en el notebook Jupyter
2. **Guardar los modelos** usando pickle o joblib:

```python
import pickle

# Guardar modelo
with open('../models/random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

# Guardar scaler
with open('../models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
```

3. **Modificar `model_utils.py`** para cargar tus modelos:

```python
def load_model():
    global MODELS, SCALERS

    models_dir = Path(__file__).parent.parent.parent / 'models'

    # Cargar Random Forest
    with open(models_dir / 'random_forest_model.pkl', 'rb') as f:
        MODELS['Random Forest'] = pickle.load(f)

    # Cargar Scaler
    with open(models_dir / 'scaler.pkl', 'rb') as f:
        SCALERS['scaler'] = pickle.load(f)

    return MODELS
```

## Personalización

### Cambiar el tema de colores

Edita el archivo [webapp/static/css/style.css](webapp/static/css/style.css)

### Agregar nuevas páginas

1. Crea una nueva ruta en [webapp/app.py](webapp/app.py)
2. Crea el template HTML en `webapp/templates/`
3. Agrega el enlace en la barra de navegación en [webapp/templates/base.html](webapp/templates/base.html)

### Modificar gráficos

Los gráficos se generan con Plotly en [webapp/app.py](webapp/app.py). Consulta la [documentación de Plotly](https://plotly.com/python/) para personalizarlos.

## Despliegue en Producción

### Opción 1: Usando Gunicorn (Linux/Mac)

```bash
pip install gunicorn
cd webapp
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Opción 2: Usando Waitress (Windows)

```bash
pip install waitress
cd webapp
waitress-serve --host=0.0.0.0 --port=5000 app:app
```

### Opción 3: Docker

Crea un `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

WORKDIR /app/webapp

CMD ["python", "app.py"]
```

Construir y ejecutar:

```bash
docker build -t fraud-detection .
docker run -p 5000:5000 fraud-detection
```

## Troubleshooting

### Error: Módulo no encontrado

```bash
# Asegúrate de estar en el entorno virtual
pip install -r requirements.txt
```

### Error: Puerto 5000 en uso

Cambia el puerto en [webapp/app.py](webapp/app.py):

```python
app.run(debug=True, host='0.0.0.0', port=8080)
```

### Modelos no cargados

Verifica que los archivos `.pkl` estén en la carpeta `models/` y que el código de carga esté correcto en `model_utils.py`.

## Contribuciones

Este proyecto fue desarrollado como parte de un proyecto académico de Ciencia de Datos.

## Licencia

Este proyecto es de código abierto y está disponible bajo la licencia MIT.

## Contacto

Para preguntas o sugerencias sobre este proyecto, por favor abre un issue en el repositorio.

---

**Nota**: Este proyecto utiliza datos simulados para demostración. Para uso en producción, asegúrate de entrenar los modelos con datos reales y validar su rendimiento adecuadamente.
