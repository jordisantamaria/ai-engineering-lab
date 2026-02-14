# Prediccion de abandono de clientes (Churn)

## Problema de negocio

La retencion de clientes es uno de los pilares mas criticos de cualquier negocio:

- **Coste de adquisicion vs. retencion**: captar un nuevo cliente cuesta entre 5x y 25x mas que retener uno existente.
- **Impacto en ingresos**: una reduccion del 5% en la tasa de churn puede incrementar los beneficios entre un 25% y un 95% (Harvard Business Review).
- **Efecto cascada**: los clientes que se van generan boca a boca negativo y arrastran a otros.
- **Reactividad**: la mayoria de empresas solo actuan cuando el cliente ya se ha ido, cuando el coste de recuperacion es maximo.

El problema no es saber *cuantos* clientes se van, sino identificar *cuales* estan en riesgo **antes** de que tomen la decision.

## Solucion propuesta

Modelo predictivo de machine learning que analiza el comportamiento historico de los clientes para asignar una probabilidad de churn a cada uno, permitiendo acciones de retencion proactivas y focalizadas.

### Enfoque tecnico

```
Datos historicos del cliente
        |
        v
  EDA (Analisis Exploratorio)
        |
        v
  Feature Engineering
  (tenure_bucket, monthly_charges_per_service,
   contract_value, engagement_score, ...)
        |
        v
  Optimizacion de hiperparametros (Optuna)
        |
        v
  XGBoost / LightGBM (ensemble)
        |
        v
  SHAP: interpretabilidad
  ("por que este cliente esta en riesgo?")
        |
        v
  API de prediccion + Dashboard
```

### Dataset

- **Telco Customer Churn**: dataset de referencia de Kaggle ([enlace](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)).
- 7.043 clientes con 21 variables (demograficas, servicios contratados, facturacion, tenure).
- Target: columna `Churn` (Yes/No).

## Resultados esperados

| Metrica | Valor |
|---------|-------|
| AUC-ROC | >0.85 |
| Precision (clase Churn) | >0.75 |
| Recall (clase Churn) | >0.80 |
| F1-Score | >0.77 |

Ademas, el modelo proporciona:
- **Top 5 factores de riesgo** para cada cliente (via SHAP).
- **Segmentacion de riesgo**: alto, medio, bajo.
- **Insights accionables**: que palancas mover para reducir el riesgo de cada segmento.

## Tecnologias

- **XGBoost / LightGBM**: modelos gradient boosting de alto rendimiento
- **SHAP**: interpretabilidad y explicabilidad del modelo
- **Optuna**: optimizacion bayesiana de hiperparametros
- **scikit-learn**: preprocesamiento y metricas
- **pandas / numpy**: manipulacion de datos
- **matplotlib / seaborn**: visualizacion
- **FastAPI**: API de prediccion en tiempo real

## Como ejecutar

### 1. Instalacion

```bash
cd portfolio/03-churn-prediction
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Descargar el dataset

Descargar el dataset de [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) y colocarlo en `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`.

### 3. Entrenar el modelo

```bash
python src/train.py \
    --data_path data/WA_Fn-UseC_-Telco-Customer-Churn.csv \
    --output_dir models/ \
    --n_trials 50
```

Esto generara:
- `models/churn_model.joblib` - modelo entrenado
- `models/roc_curve.png` - curva ROC
- `models/pr_curve.png` - curva Precision-Recall
- `models/shap_summary.png` - resumen SHAP de importancia de features
- `models/classification_report.txt` - reporte de clasificacion

### 4. Lanzar la API

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8002
```

### 5. Predecir churn de un cliente

```bash
curl -X POST "http://localhost:8002/predict" \
    -H "Content-Type: application/json" \
    -d '{
        "tenure": 12,
        "monthly_charges": 79.5,
        "total_charges": 954.0,
        "contract": "Month-to-month",
        "payment_method": "Electronic check",
        "internet_service": "Fiber optic",
        "online_security": "No",
        "tech_support": "No",
        "num_services": 4
    }'
```

Respuesta:
```json
{
    "churn_probability": 0.82,
    "risk_level": "alto",
    "top_risk_factors": [
        {"feature": "contract_Month-to-month", "impact": 0.23},
        {"feature": "tenure", "impact": -0.18},
        {"feature": "online_security_No", "impact": 0.15}
    ]
}
```

## Como presentarlo: pitch para cliente

### Propuesta de valor

> "Con este modelo pueden focalizar sus campanas de retencion en el 20% de clientes con mas riesgo, sabiendo exactamente *por que* cada cliente esta en riesgo y *que* acciones tomar para retenerlo."

### ROI estimado

**Escenario**: empresa de telecomunicaciones con 50.000 clientes, ticket medio 60 EUR/mes, tasa de churn actual 2% mensual.

| Concepto | Antes | Despues |
|----------|-------|---------|
| Clientes perdidos/mes | 1.000 | 700 (-30%) |
| Ingreso perdido/mes | 60.000 EUR | 42.000 EUR |
| Coste campana retencion | 0 EUR | 5.000 EUR (focalizada) |
| **Ahorro neto mensual** | - | **13.000 EUR** |

**Ahorro anual estimado: ~156.000 EUR**, solo en ingresos recuperados por retencion. Sin contar el ahorro en coste de adquisicion de nuevos clientes.

### Puntos clave para la presentacion

1. **Demo personalizada**: si el cliente proporciona datos anonimizados, entrenar el modelo con sus datos reales y mostrar resultados en la reunion.
2. **Interpretabilidad**: SHAP permite explicar cada prediccion en lenguaje de negocio ("este cliente tiene riesgo alto porque lleva solo 3 meses, tiene contrato mensual y no tiene soporte tecnico").
3. **Accionabilidad**: el modelo no solo dice *quien* se va, sino *por que*, lo que permite disenar intervenciones especificas.
4. **Integracion CRM**: las predicciones se pueden integrar directamente en Salesforce, HubSpot o cualquier CRM via API.
5. **Mejora continua**: el modelo se reentrena periodicamente con datos nuevos para mantener su precision.

### Preguntas frecuentes del cliente

- **"Nuestros datos son diferentes"** - El enfoque es agnnostico al sector. Se adapta a cualquier negocio con datos de clientes historicos.
- **"Cuantos datos necesitamos?"** - Un minimo de 5.000 clientes historicos con al menos 6 meses de datos. Cuantos mas, mejor.
- **"Es una caja negra?"** - No. SHAP proporciona explicaciones completas de cada prediccion. Cumple con requisitos de explicabilidad regulatoria.
- **"Cada cuanto hay que reentrenar?"** - Recomendamos reentrenamiento mensual o trimestral, segun la velocidad de cambio del negocio.
