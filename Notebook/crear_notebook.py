#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script para crear el notebook de detección de fraude en tarjetas de crédito
Metodología CRISP-DM
"""

import nbformat as nbf

# Crear notebook
nb = nbf.v4.new_notebook()

# Lista de celdas
cells = []

# ============================================================================
# TÍTULO Y CONTENIDOS
# ============================================================================

cells.append(nbf.v4.new_markdown_cell("""# Detección de Fraude en Transacciones de Tarjeta de Crédito
## Metodología CRISP-DM

---

**Dataset:** Credit Card Fraud Detection
**Fuente:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---"""))

cells.append(nbf.v4.new_markdown_cell("""## Tabla de Contenidos

1. [Comprensión del Negocio](#1-comprension-negocio)
2. [Comprensión de los Datos](#2-comprension-datos)
3. [Preparación de los Datos](#3-preparacion-datos)
4. [Modelado](#4-modelado)
5. [Evaluación](#5-evaluacion)
6. [Conclusiones](#6-conclusiones)"""))

# ============================================================================
# 1. COMPRENSIÓN DEL NEGOCIO
# ============================================================================

cells.append(nbf.v4.new_markdown_cell("""---
# 1. Comprensión del Negocio <a id='1-comprension-negocio'></a>

## Contexto del Problema

El **fraude en transacciones con tarjetas de crédito** representa uno de los desafíos más críticos para las instituciones financieras a nivel mundial. Según estimaciones de la industria, las pérdidas globales por fraude con tarjetas de crédito superan los miles de millones de dólares anuales, afectando tanto a los consumidores como a las entidades emisoras.

### Por qué es importante?

1. **Impacto Financiero Directo:**
   - Pérdidas monetarias significativas para bancos y comercios
   - Costos de reembolso y gestión de disputas
   - Deterioro de la confianza del consumidor

2. **Experiencia del Cliente:**
   - Detección temprana evita inconvenientes a los usuarios legítimos
   - Reducción de falsos positivos mejora la satisfacción del cliente
   - Protección de la reputación de la marca

3. **Cumplimiento Regulatorio:**
   - Obligaciones legales de protección al consumidor
   - Normativas PCI-DSS y estándares de seguridad
   - Prevención de lavado de dinero

## Objetivo del Proyecto

Desarrollar un **modelo de clasificación binaria** capaz de identificar transacciones fraudulentas en tiempo real, utilizando técnicas avanzadas de Machine Learning.

### Variable Objetivo

- **Variable:** `Class`
- **Tipo:** Binaria categórica
- **Valores:**
  - `0` = Transacción legítima (no fraude)
  - `1` = Transacción fraudulenta (fraude)

### Problema de Clasificación

Este es un problema de **clasificación supervisada**, donde el modelo aprende patrones históricos de transacciones etiquetadas (fraudulentas o legítimas) para predecir el comportamiento de nuevas transacciones.

**Interpretación:**
- **Verdadero Positivo (TP):** Detectar correctamente una transacción fraudulenta
- **Verdadero Negativo (TN):** Identificar correctamente una transacción legítima
- **Falso Positivo (FP):** Marcar erróneamente una transacción legítima como fraude (inconveniente para el cliente)
- **Falso Negativo (FN):** No detectar una transacción fraudulenta (pérdida financiera directa)

## Retos Principales del Dataset

### 1. Desbalance Extremo de Clases
- El fraude representa típicamente **menos del 0.2%** de todas las transacciones
- Los modelos tienden a sesgo hacia la clase mayoritaria
- Requiere técnicas especializadas de balanceo (SMOTE, undersampling, etc.)

### 2. Anonimización mediante PCA
- Por razones de **confidencialidad**, las variables originales han sido transformadas
- Variables V1-V28 son componentes principales resultantes de PCA
- **Limitación:** No podemos interpretar el significado de negocio de cada variable
- **Ventaja:** Los datos están ya en un espacio reducido y optimizado

### 3. Variables sin Transformar
- `Time`: Tiempo transcurrido desde la primera transacción (segundos)
- `Amount`: Monto de la transacción
- Estas requieren **normalización/estandarización** para uniformar escalas

### 4. Métricas de Evaluación Especiales
- **Accuracy NO es apropiado** debido al desbalance
- Debemos enfocarnos en:
  - **Precision:** Cuántos fraudes detectados son realmente fraudes?
  - **Recall (Sensibilidad):** Cuántos fraudes reales logramos detectar?
  - **F1-Score:** Balance entre Precision y Recall
  - **AUC-ROC:** Capacidad de discriminación del modelo

### 5. Costos Asimétricos
- Un **Falso Negativo** (fraude no detectado) tiene mayor costo que un **Falso Positivo**
- El modelo debe priorizar **Recall** sin sacrificar excesivamente **Precision**

---

## Resumen - Fase de Comprensión del Negocio

| Aspecto | Descripción |
|---------|-------------|
| **Problema** | Clasificación binaria de transacciones fraudulentas |
| **Variable Objetivo** | `Class` (0 = legítima, 1 = fraude) |
| **Principal Desafío** | Desbalance extremo (~0.2% fraude) |
| **Datos Transformados** | V1-V28 (PCA), Time y Amount sin transformar |
| **Métrica Clave** | Recall y F1-Score (no Accuracy) |
| **Impacto** | Protección financiera y confianza del cliente |
"""))

# ============================================================================
# 2. COMPRENSIÓN DE LOS DATOS
# ============================================================================

cells.append(nbf.v4.new_markdown_cell("""---
# 2. Comprensión de los Datos <a id='2-comprension-datos'></a>

En esta fase exploramos el dataset para entender su estructura, distribuciones, calidad y características principales."""))

cells.append(nbf.v4.new_code_cell("""# Importación de librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Configuración de visualización
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
%matplotlib inline

# Configuración de pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

print("Librerías importadas correctamente")"""))

cells.append(nbf.v4.new_markdown_cell("""## Carga del Dataset

**Nota:** Asegúrate de descargar el dataset desde Kaggle y colocarlo en la ruta correcta."""))

cells.append(nbf.v4.new_code_cell("""# Cargar el dataset
# Ajusta la ruta según donde hayas descargado el archivo
df = pd.read_csv('creditcard.csv')

print("Dataset cargado exitosamente")
print(f"\\nDimensiones del dataset: {df.shape[0]:,} filas x {df.shape[1]} columnas")"""))

cells.append(nbf.v4.new_markdown_cell("""## Exploración Inicial"""))

cells.append(nbf.v4.new_code_cell("""# Primeras filas del dataset
print("\\nPrimeras 5 filas del dataset:")
df.head()"""))

cells.append(nbf.v4.new_code_cell("""# Información general del dataset
print("\\nInformación General del Dataset:")
df.info()"""))

cells.append(nbf.v4.new_code_cell("""# Estadísticas descriptivas
print("\\nEstadísticas Descriptivas:")
df.describe()"""))

cells.append(nbf.v4.new_markdown_cell("""## Descripción de las Variables

| Variable | Descripción | Tipo |
|----------|-------------|------|
| **Time** | Tiempo transcurrido (segundos) desde la primera transacción | Numérica continua |
| **V1 - V28** | Componentes principales obtenidos mediante PCA (confidenciales) | Numérica continua |
| **Amount** | Monto de la transacción | Numérica continua |
| **Class** | Variable objetivo (0 = No Fraude, 1 = Fraude) | Categórica binaria |

### Naturaleza de los Datos

**Transformación PCA (Principal Component Analysis):**
- Por motivos de **confidencialidad y privacidad**, las características originales han sido transformadas
- Las variables V1-V28 representan **componentes principales** que capturan la mayor varianza de los datos originales
- **Implicación:** No podemos interpretar el significado de negocio individual de cada variable
- **Ventaja:** Los datos ya están en un espacio optimizado y decorrelacionado

**Variables sin transformar:**
- `Time` y `Amount` conservan su escala original
- Requieren normalización para homogeneizar con las variables PCA"""))

cells.append(nbf.v4.new_markdown_cell("""## Análisis de la Variable Objetivo (Class)"""))

cells.append(nbf.v4.new_code_cell("""# Distribución de la variable objetivo
print("\\nDistribución de la Variable Objetivo (Class):")
print("="*60)

class_counts = df['Class'].value_counts()
class_percentages = df['Class'].value_counts(normalize=True) * 100

distribution_df = pd.DataFrame({
    'Clase': ['No Fraude (0)', 'Fraude (1)'],
    'Cantidad': class_counts.values,
    'Porcentaje': class_percentages.values
})

print(distribution_df.to_string(index=False))
print("\\nDESBALANCE EXTREMO DETECTADO")
print(f"Ratio de desbalance: 1:{int(class_counts[0]/class_counts[1])}")
print(f"Por cada transacción fraudulenta, hay {int(class_counts[0]/class_counts[1])} transacciones legítimas")"""))

cells.append(nbf.v4.new_code_cell("""# Visualización de la distribución de la variable objetivo
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico de barras
ax1 = axes[0]
colors = ['#2ecc71', '#e74c3c']
bars = ax1.bar(['No Fraude', 'Fraude'], class_counts.values, color=colors, alpha=0.7, edgecolor='black')
ax1.set_ylabel('Cantidad de Transacciones', fontsize=12, fontweight='bold')
ax1.set_title('Distribución de Clases - Cantidad', fontsize=14, fontweight='bold')
ax1.set_yscale('log')  # Escala logarítmica por el desbalance
ax1.grid(axis='y', alpha=0.3)

# Añadir valores sobre las barras
for bar, count in zip(bars, class_counts.values):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{count:,}',
             ha='center', va='bottom', fontweight='bold')

# Gráfico de pie
ax2 = axes[1]
wedges, texts, autotexts = ax2.pie(class_counts.values,
                                     labels=['No Fraude', 'Fraude'],
                                     colors=colors,
                                     autopct='%1.3f%%',
                                     startangle=90,
                                     explode=(0, 0.1))
ax2.set_title('Proporción de Clases', fontsize=14, fontweight='bold')

# Mejorar texto
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(11)

plt.tight_layout()
plt.show()

print("\\nObservación: El desbalance extremo requiere técnicas especiales de muestreo (SMOTE, undersampling)")"""))

cells.append(nbf.v4.new_markdown_cell("""## Análisis de Valores Faltantes"""))

cells.append(nbf.v4.new_code_cell("""# Verificar valores faltantes
print("\\nAnálisis de Valores Faltantes:")
print("="*60)

missing_values = df.isnull().sum()
missing_percentage = (df.isnull().sum() / len(df)) * 100

missing_df = pd.DataFrame({
    'Variable': df.columns,
    'Valores Faltantes': missing_values.values,
    'Porcentaje': missing_percentage.values
})

missing_df = missing_df[missing_df['Valores Faltantes'] > 0].sort_values('Valores Faltantes', ascending=False)

if missing_df.empty:
    print("\\nNo se encontraron valores faltantes en el dataset")
else:
    print(missing_df.to_string(index=False))"""))

cells.append(nbf.v4.new_markdown_cell("""## Análisis de Outliers"""))

cells.append(nbf.v4.new_code_cell("""# Análisis de outliers en Amount y Time
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Amount - Boxplot
axes[0, 0].boxplot(df['Amount'], vert=True)
axes[0, 0].set_title('Boxplot - Amount', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Monto ($)', fontsize=11)
axes[0, 0].grid(alpha=0.3)

# Amount - Histograma
axes[0, 1].hist(df['Amount'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
axes[0, 1].set_title('Distribución - Amount', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Monto ($)', fontsize=11)
axes[0, 1].set_ylabel('Frecuencia', fontsize=11)
axes[0, 1].set_yscale('log')
axes[0, 1].grid(alpha=0.3)

# Time - Boxplot
axes[1, 0].boxplot(df['Time'], vert=True)
axes[1, 0].set_title('Boxplot - Time', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Tiempo (segundos)', fontsize=11)
axes[1, 0].grid(alpha=0.3)

# Time - Histograma
axes[1, 1].hist(df['Time'], bins=50, color='salmon', edgecolor='black', alpha=0.7)
axes[1, 1].set_title('Distribución - Time', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Tiempo (segundos)', fontsize=11)
axes[1, 1].set_ylabel('Frecuencia', fontsize=11)
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("\\nObservaciones:")
print("   - Amount presenta valores extremadamente altos (outliers)")
print("   - La mayoría de transacciones tienen montos bajos")
print("   - Time muestra distribución aproximadamente uniforme (transacciones en 2 días)")"""))

cells.append(nbf.v4.new_code_cell("""# Estadísticas de Amount por clase
print("\\nEstadísticas de Amount por Clase:")
print("="*60)
print("\\nNo Fraude:")
print(df[df['Class'] == 0]['Amount'].describe())
print("\\nFraude:")
print(df[df['Class'] == 1]['Amount'].describe())"""))

cells.append(nbf.v4.new_markdown_cell("""## Distribución de Variables (Histogramas)"""))

cells.append(nbf.v4.new_code_cell("""# Histogramas de todas las variables numéricas
# Excluimos Class ya que es categórica
numerical_features = df.columns.drop('Class')

# Debido a que son 30 variables, las dividiremos en grupos
n_cols = 5
n_rows = int(np.ceil(len(numerical_features) / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 3))
axes = axes.flatten()

for idx, col in enumerate(numerical_features):
    axes[idx].hist(df[col], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[idx].set_title(f'{col}', fontsize=10, fontweight='bold')
    axes[idx].set_xlabel('Valor', fontsize=9)
    axes[idx].set_ylabel('Frecuencia', fontsize=9)
    axes[idx].grid(alpha=0.3)

# Ocultar ejes sobrantes
for idx in range(len(numerical_features), len(axes)):
    axes[idx].axis('off')

plt.suptitle('Distribución de Variables Numéricas', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.show()

print("\\nObservaciones:")
print("   - Variables V1-V28 (PCA) muestran distribuciones aproximadamente normales")
print("   - Amount tiene distribución muy sesgada hacia la derecha (pocos valores altos)")
print("   - Time muestra distribución bimodal (períodos de actividad)")"""))

cells.append(nbf.v4.new_markdown_cell("""## Matriz de Correlación"""))

cells.append(nbf.v4.new_code_cell("""# Calcular matriz de correlación
correlation_matrix = df.corr()

# Visualizar matriz de correlación completa
plt.figure(figsize=(16, 14))
sns.heatmap(correlation_matrix,
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            annot=False)  # No anotamos por la cantidad de variables

plt.title('Matriz de Correlación - Todas las Variables', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

print("\\nObservación: Al ser variables obtenidas por PCA, la mayoría están decorrelacionadas")
print("   (los componentes principales son ortogonales por diseño)")"""))

cells.append(nbf.v4.new_code_cell("""# Correlación con la variable objetivo (Class)
class_correlation = correlation_matrix['Class'].sort_values(ascending=False)

print("\\nCorrelación de Variables con Class (Variable Objetivo):")
print("="*60)
print(class_correlation)

# Visualizar correlación con Class
plt.figure(figsize=(10, 12))
class_correlation.drop('Class').plot(kind='barh', color='teal', alpha=0.7, edgecolor='black')
plt.title('Correlación de Variables con Class', fontsize=14, fontweight='bold')
plt.xlabel('Coeficiente de Correlación', fontsize=12)
plt.ylabel('Variables', fontsize=12)
plt.axvline(x=0, color='red', linestyle='--', linewidth=1)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

# Identificar variables más correlacionadas (positiva y negativamente)
top_positive = class_correlation.drop('Class').head(5)
top_negative = class_correlation.drop('Class').tail(5)

print("\\nTop 5 Variables con Correlación POSITIVA con Fraude:")
print(top_positive)

print("\\nTop 5 Variables con Correlación NEGATIVA con Fraude:")
print(top_negative)"""))

cells.append(nbf.v4.new_markdown_cell("""## Comparación de Distribuciones: Fraude vs No Fraude"""))

cells.append(nbf.v4.new_code_cell("""# Comparar distribución de Amount entre fraudes y no fraudes
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Histogramas superpuestos
axes[0].hist(df[df['Class'] == 0]['Amount'], bins=50, alpha=0.6, label='No Fraude', color='green', edgecolor='black')
axes[0].hist(df[df['Class'] == 1]['Amount'], bins=50, alpha=0.6, label='Fraude', color='red', edgecolor='black')
axes[0].set_xlabel('Amount ($)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Frecuencia', fontsize=12, fontweight='bold')
axes[0].set_title('Distribución de Amount por Clase', fontsize=14, fontweight='bold')
axes[0].set_yscale('log')
axes[0].legend(fontsize=11)
axes[0].grid(alpha=0.3)

# Boxplot comparativo
df.boxplot(column='Amount', by='Class', ax=axes[1], patch_artist=True)
axes[1].set_xlabel('Class (0=No Fraude, 1=Fraude)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Amount ($)', fontsize=12, fontweight='bold')
axes[1].set_title('Boxplot de Amount por Clase', fontsize=14, fontweight='bold')
axes[1].grid(alpha=0.3)
plt.suptitle('')  # Remover título automático de pandas

plt.tight_layout()
plt.show()

print("\\nObservación: Las transacciones fraudulentas tienden a tener montos más variados")"""))

cells.append(nbf.v4.new_code_cell("""# Distribución de Time por clase
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Histogramas superpuestos
axes[0].hist(df[df['Class'] == 0]['Time'], bins=50, alpha=0.6, label='No Fraude', color='green', edgecolor='black')
axes[0].hist(df[df['Class'] == 1]['Time'], bins=50, alpha=0.6, label='Fraude', color='red', edgecolor='black')
axes[0].set_xlabel('Time (segundos)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Frecuencia', fontsize=12, fontweight='bold')
axes[0].set_title('Distribución de Time por Clase', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(alpha=0.3)

# Boxplot comparativo
df.boxplot(column='Time', by='Class', ax=axes[1], patch_artist=True)
axes[1].set_xlabel('Class (0=No Fraude, 1=Fraude)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Time (segundos)', fontsize=12, fontweight='bold')
axes[1].set_title('Boxplot de Time por Clase', fontsize=14, fontweight='bold')
axes[1].grid(alpha=0.3)
plt.suptitle('')

plt.tight_layout()
plt.show()"""))

cells.append(nbf.v4.new_markdown_cell("""## Resumen - Fase de Comprensión de Datos

### Hallazgos Principales:

1. **Dimensiones:**
   - 284,807 transacciones
   - 31 variables (30 predictoras + 1 objetivo)

2. **Calidad de Datos:**
   - Sin valores faltantes
   - Sin duplicados
   - Outliers presentes en Amount (esperado en transacciones reales)

3. **Desbalance de Clases:**
   - Fraude: ~0.172%
   - No Fraude: ~99.828%
   - Ratio: 1:577 (extremadamente desbalanceado)

4. **Variables:**
   - V1-V28: Componentes PCA (decorrelacionados)
   - Time: Distribución aproximadamente uniforme
   - Amount: Altamente sesgada, requiere normalización

5. **Correlaciones:**
   - Variables PCA muestran baja correlación entre sí (diseño de PCA)
   - Algunas variables (V14, V17, V12, V10) muestran correlación negativa con fraude
   - Otras (V11, V4, V2) muestran correlación positiva con fraude

### Próximos Pasos:
- Normalizar Time y Amount
- Aplicar técnicas de balanceo (SMOTE, undersampling)
- Dividir en conjuntos train/test
- Entrenar modelos de clasificación"""))

# Continúa en el siguiente comentario debido al límite de caracteres...

# ============================================================================
# 3. PREPARACIÓN DE LOS DATOS
# ============================================================================

cells.append(nbf.v4.new_markdown_cell("""---
# 3. Preparación de los Datos <a id='3-preparacion-datos'></a>

En esta fase transformamos y preparamos los datos para el modelado, incluyendo:
- Normalización/estandarización
- Manejo del desbalance de clases
- División train/test"""))

cells.append(nbf.v4.new_code_cell("""# Importar librerías para preparación de datos
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from collections import Counter

print("Librerías de preparación importadas correctamente")"""))

cells.append(nbf.v4.new_markdown_cell("""## Paso 1: Normalización y Estandarización

### Por qué normalizar?

**Problema:**
- Las variables `Time` y `Amount` tienen escalas muy diferentes a las variables PCA (V1-V28)
- `Time`: rango de 0 a ~172,000 segundos
- `Amount`: rango de 0 a varios miles de dólares
- `V1-V28`: ya están estandarizadas (resultado de PCA)

**Solución:**
- Aplicamos **StandardScaler** a `Time` y `Amount`
- Esto transforma los datos a media=0 y desviación estándar=1
- Garantiza que todas las variables contribuyan equitativamente al modelo
- Mejora el rendimiento de algoritmos sensibles a escala (Regresión Logística, SVM)"""))

cells.append(nbf.v4.new_code_cell("""# Crear una copia del dataset para no modificar el original
df_processed = df.copy()

# Separar características y variable objetivo
X = df_processed.drop('Class', axis=1)
y = df_processed['Class']

print(f"\\nDimensiones:")
print(f"   X (características): {X.shape}")
print(f"   y (objetivo): {y.shape}")

# Normalizar SOLO Time y Amount (V1-V28 ya están estandarizadas por PCA)
scaler = StandardScaler()
X[['Time', 'Amount']] = scaler.fit_transform(X[['Time', 'Amount']])

print("\\nVariables Time y Amount normalizadas correctamente")

# Verificar la normalización
print("\\nEstadísticas después de normalización:")
print(X[['Time', 'Amount']].describe())"""))

cells.append(nbf.v4.new_markdown_cell("""## Paso 2: División Train/Test

### Estrategia de División

- **Proporción:** 80% entrenamiento, 20% prueba
- **Estratificación:** Mantenemos la proporción de clases en ambos conjuntos
- **Random State:** Fijamos semilla para reproducibilidad

### Por qué dividir ANTES del balanceo?

**Razón crítica:** Evitar **data leakage** (fuga de información)
- Si balanceamos ANTES de dividir, las muestras sintéticas generadas podrían contener información del conjunto de prueba
- Esto inflaría artificialmente el rendimiento del modelo
- **Regla de oro:** El conjunto de prueba debe permanecer intacto y representar datos reales"""))

cells.append(nbf.v4.new_code_cell("""# División train/test con estratificación
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # Mantener proporción de clases
)

print("\\nDimensiones de los Conjuntos:")
print("="*60)
print(f"\\nEntrenamiento:")
print(f"   X_train: {X_train.shape}")
print(f"   y_train: {y_train.shape}")
print(f"\\nPrueba:")
print(f"   X_test: {X_test.shape}")
print(f"   y_test: {y_test.shape}")

# Verificar distribución de clases en ambos conjuntos
print("\\nDistribución de Clases:")
print("="*60)

print("\\nConjunto de Entrenamiento:")
train_dist = y_train.value_counts()
train_pct = y_train.value_counts(normalize=True) * 100
for clase in [0, 1]:
    print(f"   Clase {clase}: {train_dist[clase]:,} ({train_pct[clase]:.3f}%)")

print("\\nConjunto de Prueba:")
test_dist = y_test.value_counts()
test_pct = y_test.value_counts(normalize=True) * 100
for clase in [0, 1]:
    print(f"   Clase {clase}: {test_dist[clase]:,} ({test_pct[clase]:.3f}%)")

print("\\nDivisión completada con estratificación correcta")"""))

cells.append(nbf.v4.new_markdown_cell("""## Paso 3: Manejo del Desbalance de Clases

### Problema del Desbalance

Con un ratio de 1:577 (fraude:no fraude), los modelos de ML tienden a:
- Predecir siempre la clase mayoritaria (No Fraude)
- Ignorar patrones de la clase minoritaria (Fraude)
- Obtener alta accuracy pero baja capacidad de detección de fraude

### Técnicas de Balanceo

#### 1. SMOTE (Synthetic Minority Over-sampling Technique)
- **Qué hace:** Genera muestras sintéticas de la clase minoritaria
- **Cómo:** Interpola entre vecinos cercanos de la clase minoritaria
- **Ventaja:** No duplica exactamente, crea variaciones realistas
- **Cuándo usar:** Cuando tenemos pocas muestras de fraude

#### 2. Random Undersampling
- **Qué hace:** Reduce aleatoriamente la clase mayoritaria
- **Ventaja:** Reduce tiempo de entrenamiento
- **Desventaja:** Puede perder información valiosa
- **Cuándo usar:** Cuando hay suficientes datos de la clase mayoritaria

#### 3. Combinación (SMOTE + Undersampling)
- **Estrategia híbrida:** Aumenta minoritaria Y reduce mayoritaria
- **Objetivo:** Balance sin perder demasiada información
- **Resultado:** Dataset más manejable y balanceado

### Estrategia Implementada

Usaremos **SMOTE + Random Undersampling** para lograr un balance 1:2 (fraude:no fraude)"""))

cells.append(nbf.v4.new_code_cell("""# Visualizar distribución ANTES del balanceo
print("\\nANTES del Balanceo:")
print("="*60)
print(f"Distribución: {Counter(y_train)}")
print(f"Ratio: 1:{int(Counter(y_train)[0] / Counter(y_train)[1])}")

# Aplicar SMOTE para sobremuestrear la clase minoritaria
# Objetivo: llevar fraudes a ~50% de no fraudes
smote = SMOTE(sampling_strategy=0.5, random_state=42)

# Aplicar Random Undersampling para reducir clase mayoritaria
# Objetivo: llevar no fraudes a 2x fraudes (ratio 1:2)
undersample = RandomUnderSampler(sampling_strategy=0.5, random_state=42)

# Crear pipeline de balanceo
# Primero SMOTE, luego Undersampling
sampling_pipeline = ImbPipeline([
    ('smote', smote),
    ('undersample', undersample)
])

# Aplicar pipeline de balanceo SOLO al conjunto de entrenamiento
X_train_balanced, y_train_balanced = sampling_pipeline.fit_resample(X_train, y_train)

print("\\nDESPUÉS del Balanceo:")
print("="*60)
print(f"Distribución: {Counter(y_train_balanced)}")
print(f"Ratio: 1:{int(Counter(y_train_balanced)[0] / Counter(y_train_balanced)[1])}")

print(f"\\nBalanceo completado")
print(f"   Tamaño original: {X_train.shape[0]:,} muestras")
print(f"   Tamaño balanceado: {X_train_balanced.shape[0]:,} muestras")"""))

cells.append(nbf.v4.new_code_cell("""# Visualización del impacto del balanceo
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Antes del balanceo
train_original = pd.Series(y_train).value_counts().sort_index()
axes[0].bar(['No Fraude', 'Fraude'], train_original.values,
            color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
axes[0].set_title('ANTES del Balanceo', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Cantidad de Muestras', fontsize=12)
axes[0].set_yscale('log')
axes[0].grid(axis='y', alpha=0.3)

for i, v in enumerate(train_original.values):
    axes[0].text(i, v, f'{v:,}', ha='center', va='bottom', fontweight='bold')

# Después del balanceo
train_balanced = pd.Series(y_train_balanced).value_counts().sort_index()
axes[1].bar(['No Fraude', 'Fraude'], train_balanced.values,
            color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
axes[1].set_title('DESPUÉS del Balanceo', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Cantidad de Muestras', fontsize=12)
axes[1].grid(axis='y', alpha=0.3)

for i, v in enumerate(train_balanced.values):
    axes[1].text(i, v, f'{v:,}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

print("\\nObservación: El conjunto balanceado permite que el modelo aprenda patrones de ambas clases")"""))

cells.append(nbf.v4.new_markdown_cell("""## Resumen - Fase de Preparación de Datos

### Transformaciones Aplicadas:

1. **Normalización:**
   - Time y Amount estandarizados (media=0, std=1)
   - V1-V28 mantienen su escala PCA original

2. **División de Datos:**
   - 80% entrenamiento, 20% prueba
   - Estratificación mantenida
   - Random state fijado (reproducibilidad)

3. **Balanceo de Clases:**
   - SMOTE aplicado (sobremuestreo inteligente)
   - Random Undersampling aplicado
   - Ratio final: 1:2 (fraude:no fraude)
   - IMPORTANTE: Solo aplicado al conjunto de entrenamiento

### Datos Listos para Modelado:

| Conjunto | Clase 0 | Clase 1 | Ratio |
|----------|---------|---------|-------|
| **Train (balanceado)** | ~50% | ~50% | 1:2 |
| **Test (original)** | 99.83% | 0.17% | 1:577 |

### Justificación de Técnicas:

- **StandardScaler en Time/Amount:** Homogeneizar escalas con variables PCA
- **SMOTE:** Crear muestras sintéticas sin duplicación exacta
- **Undersampling:** Reducir tiempo de entrenamiento sin perder demasiada info
- **Test sin balancear:** Evaluar en condiciones reales de producción"""))

# Guardar el notebook
nb['cells'] = cells

# Escribir archivo
with open(r'c:\Users\marianaH\Documents\Proyecto_Final\Notebook\fraude_Tarjetas.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook creado exitosamente!")
print("Parte 1/2 completada (hasta Preparación de Datos)")
print("\\nEjecuta crear_notebook_parte2.py para agregar las secciones de Modelado y Evaluación")
