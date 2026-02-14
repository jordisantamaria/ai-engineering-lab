# Busqueda semantica para documentacion enterprise

## Problema de negocio

La busqueda por palabras clave en documentacion interna presenta limitaciones criticas en entornos empresariales:

- **Vocabulario desalineado**: los usuarios buscan "como pedir vacaciones" pero el documento dice "solicitud de ausencia reglamentaria". La busqueda por keywords no encuentra el resultado.
- **Sinonimos y contexto**: buscar "cancelar pedido" no encuentra documentos sobre "anulacion de ordenes de compra" o "devolucion de productos".
- **Sobrecarga informativa**: los empleados pasan una media de 1.8 horas/dia buscando informacion interna (McKinsey). En empresas grandes, este coste es enorme.
- **Conocimiento perdido**: el 80% del conocimiento corporativo esta en documentos no estructurados (PDFs, wikis, emails, manuales). Si no se puede encontrar, no existe.

## Solucion propuesta

Sistema de busqueda semantica que entiende el **significado** de las consultas, no solo las palabras exactas. Usa embeddings de lenguaje para representar documentos y consultas en un espacio vectorial donde la proximidad indica similitud semantica.

### Arquitectura

```
Documentos corporativos
        |
        v
  Sentence-BERT (encoding)
        |
        v
  Embeddings (vectores 384/768-dim)
        |
        v
  FAISS Index (busqueda vectorial rapida)
        |
        v
  [Opcional] Cross-Encoder Reranking
        |
        v
  Resultados ordenados por relevancia semantica

  ---

  Consulta del usuario
        |
        v
  Sentence-BERT (encoding)
        |
        v
  Query embedding
        |
        v
  FAISS nearest neighbor search
        |
        v
  Top-K documentos + scores
```

### Componentes clave

1. **Sentence-BERT**: modelo de embeddings que convierte texto en vectores densos capturando el significado semantico.
2. **FAISS (Facebook AI Similarity Search)**: biblioteca de busqueda vectorial de alto rendimiento. Soporta billones de vectores con tiempos de respuesta <10ms.
3. **Cross-Encoder Reranking**: paso opcional que reordena los resultados iniciales con un modelo mas preciso para mejorar la calidad del top-5.
4. **Busqueda hibrida**: combina busqueda semantica con busqueda por keywords (BM25) para lo mejor de ambos mundos.

## Resultados esperados

| Metrica | Keyword Search | Busqueda Semantica |
|---------|---------------|-------------------|
| MRR (Mean Reciprocal Rank) | 0.35 | >0.65 |
| Recall@10 | 0.45 | >0.80 |
| Tiempo de busqueda | ~50ms | <100ms |
| Satisfaccion usuario | 55% | >85% |

## Tecnologias

- **sentence-transformers**: embeddings semanticos de ultima generacion
- **FAISS**: busqueda vectorial de alto rendimiento (Meta AI)
- **FastAPI**: API REST para integracion
- **numpy / pandas**: manipulacion de datos y vectores
- **Pydantic**: validacion de datos

## Como ejecutar

### 1. Instalacion

```bash
cd portfolio/05-semantic-search
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Lanzar la API

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8004
```

### 3. Indexar documentos

```bash
curl -X POST "http://localhost:8004/index" \
    -H "Content-Type: application/json" \
    -d '{
        "documents": [
            {"id": "doc1", "text": "Procedimiento para solicitar vacaciones anuales"},
            {"id": "doc2", "text": "Politica de devolucion de productos defectuosos"},
            {"id": "doc3", "text": "Guia de onboarding para nuevos empleados"}
        ]
    }'
```

### 4. Buscar

```bash
curl -X POST "http://localhost:8004/search" \
    -H "Content-Type: application/json" \
    -d '{"query": "como pedir dias libres", "top_k": 5}'
```

Respuesta esperada:
```json
{
    "results": [
        {
            "document_id": "doc1",
            "text": "Procedimiento para solicitar vacaciones anuales",
            "score": 0.8923,
            "rank": 1
        }
    ],
    "query": "como pedir dias libres",
    "total_results": 1,
    "search_time_ms": 12.5
}
```

### 5. Docker

```bash
docker build -t semantic-search .
docker run -p 8004:8004 semantic-search
```

## Como presentarlo: pitch para cliente

### Propuesta de valor

> "Sus empleados encuentran la informacion que necesitan 3x mas rapido. Nuestra busqueda semantica entiende lo que el usuario *quiere decir*, no solo lo que escribe. Funciona con documentos en multiples idiomas y no requiere etiquetado manual."

### ROI estimado

**Escenario**: empresa con 1.000 empleados que pasan 30 minutos/dia buscando informacion interna.

| Concepto | Antes (keywords) | Despues (semantica) |
|----------|-------------------|---------------------|
| Tiempo busqueda/dia/empleado | 30 min | 10 min |
| Horas totales/mes perdidas | 10.000 h | 3.333 h |
| Coste productividad perdida (30 EUR/h) | 300.000 EUR/mes | 100.000 EUR/mes |
| Busquedas exitosas (1er intento) | 40% | >80% |

**Ahorro estimado: ~200.000 EUR/mes** en productividad recuperada. Adicionalmente, reduccion de errores por uso de informacion desactualizada o incorrecta.

### Puntos clave para la presentacion

1. **Demo en vivo**: indexar algunos documentos del cliente y demostrar busquedas semanticas que fallan con keywords.
2. **Multi-idioma**: el modelo soporta busquedas en espanol, ingles, catalan y otros idiomas simultaneamente.
3. **Incrementalidad**: se pueden anadir documentos al indice sin reconstruirlo desde cero.
4. **Integracion**: API REST que se conecta con cualquier wiki, intranet o sistema de gestion documental.
5. **Privacidad**: todo se ejecuta on-premise, los documentos no salen de la infraestructura del cliente.

### Preguntas frecuentes del cliente

- **"Funciona con PDFs y documentos escaneados?"** - Si, se combina con OCR para documentos escaneados y con extractores de texto para PDFs nativos.
- **"Cuantos documentos puede manejar?"** - FAISS escala a millones de documentos con tiempos de respuesta <100ms. Para colecciones muy grandes se usa un indice IVF (Inverted File).
- **"Necesita GPU?"** - No para servir (CPU es suficiente para busqueda). Si es recomendable GPU para el indexado inicial si hay muchos documentos.
- **"Se puede integrar con nuestro SharePoint/Confluence?"** - Si, via conectores que extraen el contenido y lo indexan periodicamente.
