# Deteccion automatica de defectos en manufactura

## Problema de negocio

La inspeccion visual manual en lineas de produccion presenta desafios criticos:

- **Lentitud**: un inspector humano tarda entre 5-15 segundos por pieza, limitando el throughput de la linea.
- **Inconsistencia**: la precision varia segun la fatiga, iluminacion y experiencia del inspector. Estudios muestran que la tasa de deteccion manual cae un 20-30% tras 2 horas continuas.
- **Coste elevado**: mantener equipos de inspectores 24/7 en tres turnos supone un coste salarial significativo.
- **Defectos escapados**: los defectos que llegan al cliente generan devoluciones, reclamaciones de garantia y dano reputacional.

## Solucion propuesta

Sistema de vision por computador que detecta defectos en productos en tiempo real, integrado directamente en la linea de produccion.

### Arquitectura tecnica

```
Camara industrial --> Preprocesamiento imagen --> EfficientNet-B0 (Transfer Learning)
                                                        |
                                                  Classification Head
                                                        |
                                              Defecto / No defecto
                                              (+ tipo de defecto)
                                                        |
                                                  FastAPI Server
                                                        |
                                              Dashboard / Alertas
```

- **Modelo base**: EfficientNet-B0 preentrenado en ImageNet, con la ultima capa fully-connected reemplazada para clasificacion binaria (defecto/no-defecto) o multi-clase (tipo de defecto).
- **Transfer Learning**: se congelan las primeras capas del backbone y se entrenan solo las ultimas capas + classification head, permitiendo entrenar con pocos datos (~500-1000 imagenes).
- **Data Augmentation**: rotaciones, flips, cambios de brillo/contraste, recortes aleatorios para robustecer el modelo.
- **Servicio de inferencia**: API REST con FastAPI, containerizado con Docker para despliegue sencillo.

### Dataset

- **MVTec Anomaly Detection Dataset** (MVTec AD): dataset de referencia para deteccion de defectos industriales con 15 categorias de productos y multiples tipos de defecto.
- Alternativa: dataset sintetico incluido en el repositorio para demos rapidas.

## Resultados esperados

| Metrica | Valor |
|---------|-------|
| Accuracy | >95% |
| Precision | >93% |
| Recall | >94% |
| Tiempo de inferencia | <100ms por imagen |
| Throughput | >10 imagenes/segundo |

## Tecnologias

- **PyTorch** + **torchvision**: framework de deep learning y modelos preentrenados
- **Albumentations**: data augmentation avanzada
- **FastAPI**: API REST de alto rendimiento
- **ONNX Runtime**: inferencia optimizada en produccion
- **Docker**: containerizacion para despliegue
- **OpenCV**: preprocesamiento de imagenes

## Como ejecutar

### 1. Instalacion

```bash
# Clonar el repositorio
git clone <repo-url>
cd portfolio/01-defect-detection

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Entrenar el modelo

```bash
python src/train.py \
    --data_dir data/defects \
    --epochs 30 \
    --batch_size 32 \
    --lr 0.001 \
    --output_dir models/
```

La estructura esperada del directorio de datos:
```
data/defects/
    train/
        good/
            img001.jpg
            img002.jpg
        defect/
            img101.jpg
            img102.jpg
    val/
        good/
        defect/
```

### 3. Lanzar la API

```bash
python src/api.py
# o bien:
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

### 4. Probar la prediccion

```bash
curl -X POST "http://localhost:8000/predict" \
    -F "file=@test_image.jpg"
```

### 5. Docker

```bash
docker build -t defect-detection .
docker run -p 8000:8000 defect-detection
```

## Como presentarlo: pitch para cliente

### Propuesta de valor

> "Imagine sustituir la variabilidad humana por un sistema que inspecciona cada pieza en menos de 100 milisegundos, 24 horas al dia, 7 dias a la semana, sin fatiga y con una precision superior al 95%."

### ROI estimado

**Escenario**: planta con 3 turnos, 4 inspectores por turno (12 inspectores totales).

| Concepto | Antes | Despues |
|----------|-------|---------|
| Coste inspeccion anual | ~360.000 EUR (12 inspectores) | ~80.000 EUR (2 inspectores + sistema) |
| Defectos escapados | 2-5% | <0.5% |
| Velocidad inspeccion | 5-15 seg/pieza | <0.1 seg/pieza |
| Disponibilidad | Sujeta a turnos/bajas | 24/7 continuo |

**Ahorro neto estimado: ~280.000 EUR/ano**, sin contar la reduccion en costes de garantia y devoluciones.

### Puntos clave para la presentacion

1. **Demo en vivo**: mostrar la API procesando imagenes en tiempo real.
2. **Metricas claras**: precision, recall, y velocidad de inferencia.
3. **Escalabilidad**: un modelo, multiples camaras/lineas.
4. **Integracion**: se conecta con sistemas SCADA/MES existentes via API REST.
5. **Mejora continua**: el modelo se puede reentrenar con nuevos tipos de defecto sin redisenar el sistema.

### Preguntas frecuentes del cliente

- **"Y si aparece un tipo de defecto nuevo?"** - Se recogen imagenes del nuevo defecto, se reentrena el modelo (fine-tuning rapido), y se despliega sin parar la linea.
- **"Que pasa si falla el sistema?"** - Fallback a inspeccion manual. El sistema tiene health checks y alertas automaticas.
- **"Cuanto tarda la implementacion?"** - Piloto funcional en 4-6 semanas. Integracion completa en 2-3 meses.
