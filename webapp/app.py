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

        # Obtener el nombre del mejor modelo
        best_model_name = 'N/A'
        if 'models_comparison' in metrics and metrics['models_comparison']:
            best_model = max(metrics['models_comparison'], key=lambda x: x.get('f1_score', 0))
            best_model_name = best_model.get('model', 'N/A')

        # KPIs principales
        kpis = {
            'total_transacciones': metrics.get('total_transacciones', 'N/A'),
            'fraudes_detectados': metrics.get('fraudes_detectados', 'N/A'),
            'accuracy': metrics.get('best_model_accuracy', 'N/A'),
            'recall': metrics.get('best_model_recall', 'N/A'),
            'best_model_name': best_model_name
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
                'text': 'Distribución de Transacciones en el Dataset de Prueba (Test Set)',
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

            # Crear escala de colores personalizada para mejor distinción
            # Verde oscuro para TN (True Negative - alto y bueno)
            # Verde claro para TP (True Positive - bueno pero menor cantidad)
            # Rojo claro para FP (False Positive - error)
            # Rojo oscuro para FN (False Negative - error crítico)
            colorscale = [
                [0, '#d73027'],      # Rojo oscuro (valores bajos)
                [0.33, '#fc8d59'],   # Naranja
                [0.66, '#fee090'],   # Amarillo claro
                [1, '#1a9850']       # Verde oscuro (valores altos)
            ]

            # Crear heatmap de la matriz de confusión
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Predicción: Legítima', 'Predicción: Fraude'],
                y=['Real: Legítima', 'Real: Fraude'],
                text=[[annotations[0], annotations[1]],
                      [annotations[2], annotations[3]]],
                texttemplate='%{text}',
                textfont=dict(size=14, color='white'),
                colorscale=colorscale,
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

@app.route('/analisis')
def analisis():
    """Página de análisis exploratorio con Feature Importance y distribuciones"""
    try:
        metrics = get_model_metrics()

        # Feature Importance
        feature_importance_data = metrics.get('feature_importance', {})
        fig_importance_list = []

        for model_name, importance_info in feature_importance_data.items():
            features = importance_info.get('features', [])
            importances = importance_info.get('importances', [])

            if features and importances:
                fig = go.Figure(data=[
                    go.Bar(
                        x=importances,
                        y=features,
                        orientation='h',
                        marker_color='#3498db',
                        text=[f'{val:.4f}' for val in importances],
                        textposition='outside',
                        textfont=dict(size=10)
                    )
                ])
                fig.update_layout(
                    title={
                        'text': f'Importancia de Características - {model_name}',
                        'x': 0.5,
                        'xanchor': 'center'
                    },
                    xaxis_title='Importancia',
                    yaxis_title='Característica',
                    template='plotly_white',
                    height=500,
                    margin=dict(l=80, r=80, t=80, b=50),
                    yaxis=dict(autorange='reversed')
                )
                fig_importance_list.append({
                    'model': model_name,
                    'graph': json.dumps(fig, cls=PlotlyJSONEncoder)
                })

        # Distribución de Amount por clase
        data_analysis = metrics.get('data_analysis', {})
        amount_dist = data_analysis.get('amount_distribution', {})

        fig_amount = None
        if amount_dist:
            fraude_stats = amount_dist.get('fraude', {})
            no_fraude_stats = amount_dist.get('no_fraude', {})

            # Crear gráfico de barras agrupadas para comparar estadísticas
            categories = ['Media', 'Mediana', 'Desv. Est.', 'Mínimo', 'Máximo']
            fraude_values = [
                fraude_stats.get('mean', 0),
                fraude_stats.get('median', 0),
                fraude_stats.get('std', 0),
                fraude_stats.get('min', 0),
                fraude_stats.get('max', 0)
            ]
            no_fraude_values = [
                no_fraude_stats.get('mean', 0),
                no_fraude_stats.get('median', 0),
                no_fraude_stats.get('std', 0),
                no_fraude_stats.get('min', 0),
                no_fraude_stats.get('max', 0)
            ]

            fig_amount = go.Figure(data=[
                go.Bar(
                    name='Fraude',
                    x=categories,
                    y=fraude_values,
                    marker_color='#e74c3c',
                    text=[f'${val:.2f}' for val in fraude_values],
                    textposition='outside',
                    textfont=dict(size=10)
                ),
                go.Bar(
                    name='No Fraude',
                    x=categories,
                    y=no_fraude_values,
                    marker_color='#2ecc71',
                    text=[f'${val:.2f}' for val in no_fraude_values],
                    textposition='outside',
                    textfont=dict(size=10)
                )
            ])
            fig_amount.update_layout(
                title={
                    'text': 'Estadísticas de Amount por Clase',
                    'x': 0.5,
                    'xanchor': 'center'
                },
                xaxis_title='Estadística',
                yaxis_title='Valor ($)',
                barmode='group',
                template='plotly_white',
                height=450,
                margin=dict(l=50, r=50, t=80, b=50),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            fig_amount = json.dumps(fig_amount, cls=PlotlyJSONEncoder)

        # Comparación de métricas por modelo (4 subplots)
        models_comparison = metrics.get('models_comparison', [])
        fig_metrics_comparison = None

        if models_comparison:
            from plotly.subplots import make_subplots

            df_models = pd.DataFrame(models_comparison)

            fig_metrics_comparison = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Precision', 'Recall', 'F1-Score', 'ROC-AUC')
            )

            metrics_list = [
                ('precision', 1, 1),
                ('recall', 1, 2),
                ('f1_score', 2, 1),
                ('roc_auc', 2, 2)
            ]

            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

            for metric, row, col in metrics_list:
                if metric in df_models.columns:
                    for idx, (model_name, value) in enumerate(zip(df_models['model'], df_models[metric])):
                        fig_metrics_comparison.add_trace(
                            go.Bar(
                                x=[model_name],
                                y=[value],
                                name=model_name,
                                marker_color=colors[idx % len(colors)],
                                text=f'{value:.4f}',
                                textposition='outside',
                                textfont=dict(size=10),
                                showlegend=(row == 1 and col == 1)
                            ),
                            row=row, col=col
                        )

            fig_metrics_comparison.update_layout(
                title={
                    'text': 'Comparación Detallada de Métricas por Modelo',
                    'x': 0.5,
                    'xanchor': 'center'
                },
                template='plotly_white',
                height=700,
                margin=dict(l=50, r=50, t=100, b=50)
            )
            fig_metrics_comparison.update_yaxes(range=[0, 1.1])
            fig_metrics_comparison = json.dumps(fig_metrics_comparison, cls=PlotlyJSONEncoder)

        # Distribución de Time por clase
        time_dist = data_analysis.get('time_distribution', {})
        fig_time = None

        if time_dist:
            fraude_stats = time_dist.get('fraude', {})
            no_fraude_stats = time_dist.get('no_fraude', {})

            categories = ['Media', 'Mediana', 'Desv. Est.', 'Mínimo', 'Máximo']
            fraude_values = [
                fraude_stats.get('mean', 0),
                fraude_stats.get('median', 0),
                fraude_stats.get('std', 0),
                fraude_stats.get('min', 0),
                fraude_stats.get('max', 0)
            ]
            no_fraude_values = [
                no_fraude_stats.get('mean', 0),
                no_fraude_stats.get('median', 0),
                no_fraude_stats.get('std', 0),
                no_fraude_stats.get('min', 0),
                no_fraude_stats.get('max', 0)
            ]

            fig_time = go.Figure(data=[
                go.Bar(
                    name='Fraude',
                    x=categories,
                    y=fraude_values,
                    marker_color='#e74c3c',
                    text=[f'{val:.2f}' for val in fraude_values],
                    textposition='outside',
                    textfont=dict(size=10)
                ),
                go.Bar(
                    name='No Fraude',
                    x=categories,
                    y=no_fraude_values,
                    marker_color='#2ecc71',
                    text=[f'{val:.2f}' for val in no_fraude_values],
                    textposition='outside',
                    textfont=dict(size=10)
                )
            ])
            fig_time.update_layout(
                title={
                    'text': 'Estadísticas de Time por Clase',
                    'x': 0.5,
                    'xanchor': 'center'
                },
                xaxis_title='Estadística',
                yaxis_title='Valor (segundos)',
                barmode='group',
                template='plotly_white',
                height=450,
                margin=dict(l=50, r=50, t=80, b=50),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            fig_time = json.dumps(fig_time, cls=PlotlyJSONEncoder)

        # Matriz de correlación
        correlation_matrix = metrics.get('correlation_matrix', {})
        fig_correlation = None

        if correlation_matrix:
            corr_values = correlation_matrix.get('values', [])
            corr_columns = correlation_matrix.get('columns', [])

            if corr_values and corr_columns:
                fig_correlation = go.Figure(data=go.Heatmap(
                    z=corr_values,
                    x=corr_columns,
                    y=corr_columns,
                    colorscale='RdBu_r',
                    zmid=0,
                    colorbar=dict(title="Correlación")
                ))
                fig_correlation.update_layout(
                    title={
                        'text': 'Matriz de Correlación - Todas las Variables',
                        'x': 0.5,
                        'xanchor': 'center'
                    },
                    template='plotly_white',
                    height=700,
                    margin=dict(l=50, r=50, t=80, b=50)
                )
                fig_correlation = json.dumps(fig_correlation, cls=PlotlyJSONEncoder)

        # Correlación con variable objetivo
        correlation_with_class = data_analysis.get('correlation_with_class', {})
        fig_corr_class = None

        if correlation_with_class:
            features = correlation_with_class.get('features', [])
            correlations = correlation_with_class.get('correlations', [])

            if features and correlations:
                fig_corr_class = go.Figure(data=[
                    go.Bar(
                        x=correlations,
                        y=features,
                        orientation='h',
                        marker_color='#9b59b6',
                        text=[f'{val:.4f}' for val in correlations],
                        textposition='outside',
                        textfont=dict(size=10)
                    )
                ])
                fig_corr_class.update_layout(
                    title={
                        'text': 'Correlación de Variables con Class (Top 20)',
                        'x': 0.5,
                        'xanchor': 'center'
                    },
                    xaxis_title='Correlación Absoluta',
                    yaxis_title='Variable',
                    template='plotly_white',
                    height=600,
                    margin=dict(l=80, r=80, t=80, b=50),
                    yaxis=dict(autorange='reversed')
                )
                fig_corr_class = json.dumps(fig_corr_class, cls=PlotlyJSONEncoder)

        # Balanceo de clases
        class_balance = metrics.get('class_balance', {})
        fig_balance = None

        if class_balance:
            before = class_balance.get('before', {})
            after = class_balance.get('after', {})

            if before and after:
                from plotly.subplots import make_subplots

                fig_balance = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('ANTES del Balanceo', 'DESPUÉS del Balanceo')
                )

                # ANTES
                fig_balance.add_trace(
                    go.Bar(
                        x=['No Fraude', 'Fraude'],
                        y=[before.get('no_fraude', 0), before.get('fraude', 0)],
                        marker_color=['#2ecc71', '#e74c3c'],
                        text=[f"{before.get('no_fraude', 0):,}", f"{before.get('fraude', 0):,}"],
                        textposition='outside',
                        textfont=dict(size=12),
                        showlegend=False
                    ),
                    row=1, col=1
                )

                # DESPUÉS
                fig_balance.add_trace(
                    go.Bar(
                        x=['No Fraude', 'Fraude'],
                        y=[after.get('no_fraude', 0), after.get('fraude', 0)],
                        marker_color=['#2ecc71', '#e74c3c'],
                        text=[f"{after.get('no_fraude', 0):,}", f"{after.get('fraude', 0):,}"],
                        textposition='outside',
                        textfont=dict(size=12),
                        showlegend=False
                    ),
                    row=1, col=2
                )

                fig_balance.update_layout(
                    title={
                        'text': 'Impacto del Balanceo de Clases (SMOTE + Undersampling)',
                        'x': 0.5,
                        'xanchor': 'center'
                    },
                    template='plotly_white',
                    height=450,
                    margin=dict(l=50, r=50, t=100, b=50)
                )
                fig_balance.update_yaxes(title_text="Cantidad de Muestras")
                fig_balance = json.dumps(fig_balance, cls=PlotlyJSONEncoder)

        return render_template(
            'analisis.html',
            fig_importance_list=fig_importance_list,
            fig_amount=fig_amount,
            fig_time=fig_time,
            fig_metrics_comparison=fig_metrics_comparison,
            fig_correlation=fig_correlation,
            fig_corr_class=fig_corr_class,
            fig_balance=fig_balance
        )
    except Exception as e:
        return render_template('analisis.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
