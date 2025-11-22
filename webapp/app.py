from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder
import json
from utils.model_utils import load_model, predict_fraud, get_model_metrics
from sklearn.metrics import confusion_matrix, roc_curve, auc

app = Flask(__name__)
app.config['SECRET_KEY'] = 'tu-clave-secreta-aqui'

# Cargar modelo al iniciar la aplicación
model_data = load_model()

@app.route('/')
def home():
    """Página principal con overview del proyecto"""
    try:
        metrics = get_model_metrics()

        # KPIs principales
        kpis = {
            'total_transacciones': metrics.get('total_transacciones', 'N/A'),
            'fraudes_detectados': metrics.get('fraudes_detectados', 'N/A'),
            'accuracy': metrics.get('best_model_accuracy', 'N/A'),
            'recall': metrics.get('best_model_recall', 'N/A')
        }

        # Gráfico de distribución de clases
        legitimas = metrics.get('total_legitimas', 284315)
        fraudes = metrics.get('total_fraudes', 492)
        total = legitimas + fraudes

        fig_dist = go.Figure(data=[
            go.Bar(
                x=['Legítimas', 'Fraudulentas'],
                y=[legitimas, fraudes],
                marker_color=['#2ecc71', '#e74c3c'],
                text=[f'{legitimas:,}<br>({legitimas/total*100:.2f}%)',
                      f'{fraudes:,}<br>({fraudes/total*100:.2f}%)'],
                textposition='auto',
                textfont=dict(size=12, color='white')
            )
        ])
        fig_dist.update_layout(
            title={
                'text': 'Distribución de Transacciones en el Dataset',
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title='Tipo de Transacción',
            yaxis_title='Cantidad de Transacciones',
            template='plotly_white',
            height=400,
            showlegend=False,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        graph_dist_json = json.dumps(fig_dist, cls=PlotlyJSONEncoder)

        return render_template('home.html', kpis=kpis, graph_dist=graph_dist_json)
    except Exception as e:
        return render_template('home.html', error=str(e))

@app.route('/comparacion')
def comparacion():
    """Página de comparación de modelos"""
    try:
        metrics = get_model_metrics()
        models_comparison = metrics.get('models_comparison', [])

        if not models_comparison:
            return render_template('comparacion.html', error="No hay datos de comparación disponibles")

        # Crear DataFrame para facilitar visualización
        df_models = pd.DataFrame(models_comparison)

        # Gráfico de barras agrupadas para métricas
        fig_metrics = go.Figure()

        metrics_to_plot = ['precision', 'recall', 'f1_score', 'roc_auc']
        metric_names = {'precision': 'Precision', 'recall': 'Recall',
                       'f1_score': 'F1-Score', 'roc_auc': 'ROC-AUC'}

        for metric in metrics_to_plot:
            if metric in df_models.columns:
                fig_metrics.add_trace(go.Bar(
                    name=metric_names.get(metric, metric),
                    x=df_models['model'],
                    y=df_models[metric],
                    text=[f'{val:.4f}' for val in df_models[metric]],
                    textposition='outside',
                    textfont=dict(size=10)
                ))

        fig_metrics.update_layout(
            title={
                'text': 'Comparación de Métricas de Desempeño por Modelo',
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title='Modelo de Clasificación',
            yaxis_title='Score',
            barmode='group',
            template='plotly_white',
            yaxis_range=[0, 1.1],
            height=450,
            margin=dict(l=50, r=50, t=80, b=80),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        graph_metrics_json = json.dumps(fig_metrics, cls=PlotlyJSONEncoder)

        # Curva ROC (si está disponible)
        roc_data = metrics.get('roc_curves', {})
        fig_roc = go.Figure()

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        for idx, (model_name, roc_info) in enumerate(roc_data.items()):
            fig_roc.add_trace(go.Scatter(
                x=roc_info['fpr'],
                y=roc_info['tpr'],
                mode='lines',
                name=f'{model_name} (AUC = {roc_info["auc"]:.4f})',
                line=dict(width=2, color=colors[idx % len(colors)])
            ))

        # Línea diagonal de referencia
        fig_roc.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Clasificador Aleatorio (AUC = 0.5)',
            line=dict(dash='dash', color='gray', width=2)
        ))

        fig_roc.update_layout(
            title={
                'text': 'Curvas ROC - Receiver Operating Characteristic',
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title='Tasa de Falsos Positivos (FPR)',
            yaxis_title='Tasa de Verdaderos Positivos (TPR)',
            template='plotly_white',
            height=500,
            margin=dict(l=50, r=50, t=80, b=50),
            legend=dict(
                x=0.6,
                y=0.1,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='gray',
                borderwidth=1
            ),
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1])
        )
        graph_roc_json = json.dumps(fig_roc, cls=PlotlyJSONEncoder)

        return render_template(
            'comparacion.html',
            models=models_comparison,
            graph_metrics=graph_metrics_json,
            graph_roc=graph_roc_json
        )
    except Exception as e:
        return render_template('comparacion.html', error=str(e))

@app.route('/prediccion', methods=['GET', 'POST'])
def prediccion():
    """Página de predicción individual"""
    if request.method == 'POST':
        try:
            # Obtener datos del formulario
            features = {
                'Time': float(request.form.get('time', 0)),
                'Amount': float(request.form.get('amount', 0))
            }

            # Agregar features V1-V28
            for i in range(1, 29):
                feature_name = f'V{i}'
                features[feature_name] = float(request.form.get(feature_name, 0))

            # Realizar predicción
            result = predict_fraud(features)

            # Crear gráfico de probabilidades
            fig_prob = go.Figure(data=[
                go.Bar(
                    x=['Legítima', 'Fraudulenta'],
                    y=[result['probability_legitimate'], result['probability_fraud']],
                    marker_color=['#2ecc71', '#e74c3c'],
                    text=[f"{result['probability_legitimate']:.1%}", f"{result['probability_fraud']:.1%}"],
                    textposition='auto',
                    textfont=dict(size=14, color='white')
                )
            ])
            fig_prob.update_layout(
                title={
                    'text': 'Probabilidades de Clasificación',
                    'x': 0.5,
                    'xanchor': 'center'
                },
                yaxis_title='Probabilidad',
                template='plotly_white',
                yaxis_range=[0, 1],
                height=350,
                showlegend=False,
                margin=dict(l=50, r=50, t=60, b=50)
            )
            graph_prob_json = json.dumps(fig_prob, cls=PlotlyJSONEncoder)

            return render_template(
                'prediccion.html',
                result=result,
                graph_prob=graph_prob_json,
                features=features
            )
        except Exception as e:
            return render_template('prediccion.html', error=str(e))

    return render_template('prediccion.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint para predicciones"""
    try:
        data = request.get_json()
        result = predict_fraud(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/matriz-confusion')
def matriz_confusion():
    """Página con matrices de confusión de los modelos"""
    try:
        metrics = get_model_metrics()
        confusion_matrices = metrics.get('confusion_matrices', {})

        figures = []
        for model_name, cm in confusion_matrices.items():
            # Crear anotaciones personalizadas con porcentajes
            cm_array = np.array(cm)
            total = cm_array.sum()
            annotations = []

            for i in range(2):
                for j in range(2):
                    count = cm_array[i][j]
                    percentage = (count / total) * 100
                    annotations.append(
                        f'{count:,}<br>({percentage:.2f}%)'
                    )

            # Crear heatmap de la matriz de confusión
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Predicción: Legítima', 'Predicción: Fraude'],
                y=['Real: Legítima', 'Real: Fraude'],
                text=[[annotations[0], annotations[1]],
                      [annotations[2], annotations[3]]],
                texttemplate='%{text}',
                textfont=dict(size=12),
                colorscale='RdYlGn_r',
                showscale=True,
                colorbar=dict(title="Cantidad")
            ))
            fig.update_layout(
                title={
                    'text': f'Matriz de Confusión - {model_name}',
                    'x': 0.5,
                    'xanchor': 'center'
                },
                xaxis_title='Predicción del Modelo',
                yaxis_title='Valor Real',
                template='plotly_white',
                height=400,
                margin=dict(l=80, r=80, t=80, b=80)
            )
            figures.append({
                'model': model_name,
                'graph': json.dumps(fig, cls=PlotlyJSONEncoder)
            })

        return render_template('matriz_confusion.html', figures=figures)
    except Exception as e:
        return render_template('matriz_confusion.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
