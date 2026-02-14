# Motor de recomendaciones de productos

## Problema de negocio

Las recomendaciones de productos son uno de los motores mas potentes de crecimiento en e-commerce y retail:

- **Conversion**: los usuarios que interactuan con recomendaciones tienen una tasa de conversion 2-5x mayor.
- **Ticket medio**: las recomendaciones tipo "tambien te puede interesar" y "comprados juntos" incrementan el valor medio del pedido en un 10-30%.
- **Engagement**: las recomendaciones personalizadas aumentan el tiempo de permanencia en la plataforma y la frecuencia de visita.
- **Descubrimiento**: el 35% de las ventas de Amazon provienen de su motor de recomendaciones (McKinsey).

Sin un sistema de recomendaciones, se muestra el mismo catalogo a todos los usuarios, perdiendo oportunidades de venta cruzada y personalizacion.

## Solucion propuesta

Sistema de recomendacion hibrido que combina dos enfoques complementarios:

1. **Content-Based Filtering**: recomienda productos similares basandose en las descripciones y atributos del producto (usa embeddings semanticos).
2. **Collaborative Filtering**: recomienda productos basandose en el comportamiento de usuarios similares (ALS - Alternating Least Squares).
3. **Enfoque hibrido**: combina ambos con ponderacion ajustable para obtener las mejores recomendaciones, incluyendo manejo de cold start.

### Arquitectura

```
                    +---------------------------+
                    |    API de Recomendaciones  |
                    +---------------------------+
                       /                    \
                      v                      v
    +--------------------+      +------------------------+
    | Content-Based      |      | Collaborative          |
    | (SentenceTransf.)  |      | (ALS / Implicit)       |
    +--------------------+      +------------------------+
    | - Embeddings prod.  |      | - Matriz user-item    |
    | - Cosine similarity |      | - Factorizacion       |
    | - Productos nuevos  |      | - Usuarios similares  |
    +--------------------+      +------------------------+
                      \                      /
                       v                    v
                    +---------------------------+
                    |    Hybrid Recommender      |
                    |  (weighted ensemble)       |
                    +---------------------------+
                              |
                              v
                    Recomendaciones finales
                    (con cold start handling)
```

## Resultados esperados

| Metrica | Valor |
|---------|-------|
| Precision@10 | >0.15 |
| Recall@10 | >0.10 |
| NDCG@10 | >0.20 |
| Cobertura del catalogo | >60% |
| Tiempo de respuesta | <200ms |

## Tecnologias

- **sentence-transformers**: embeddings semanticos de descripciones de productos
- **implicit**: biblioteca ALS para collaborative filtering
- **scikit-learn**: metricas de similitud y evaluacion
- **scipy**: matrices sparse para interacciones usuario-producto
- **FastAPI**: API REST de alto rendimiento
- **pandas / numpy**: manipulacion de datos

## Como ejecutar

### 1. Instalacion

```bash
cd portfolio/04-recommendation-engine
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Preparar datos

El sistema espera dos archivos CSV:

- `data/products.csv`: columnas `product_id`, `name`, `description`, `category`, `price`
- `data/interactions.csv`: columnas `user_id`, `product_id`, `rating` (o `purchase_count`)

### 3. Lanzar la API

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8003
```

### 4. Obtener recomendaciones

```bash
# Recomendaciones para un usuario
curl -X POST "http://localhost:8003/recommend" \
    -H "Content-Type: application/json" \
    -d '{"user_id": "user_123", "n": 10}'

# Productos similares
curl -X POST "http://localhost:8003/similar" \
    -H "Content-Type: application/json" \
    -d '{"product_id": "prod_456", "n": 5}'
```

## Como presentarlo: pitch para cliente

### Propuesta de valor

> "Las empresas con recomendaciones personalizadas ven un incremento del 10-30% en revenue. Nuestro sistema hibrido funciona incluso con usuarios nuevos (cold start) y productos recien anadidos, adaptandose en tiempo real al comportamiento de sus clientes."

### ROI estimado

**Escenario**: e-commerce con 100.000 usuarios activos/mes, ticket medio 45 EUR, tasa de conversion actual 2.5%.

| Concepto | Antes | Despues |
|----------|-------|---------|
| Tasa de conversion | 2.5% | 3.5% (+40%) |
| Ticket medio | 45 EUR | 52 EUR (+15%) |
| Pedidos/mes | 2.500 | 3.500 |
| Revenue mensual | 112.500 EUR | 182.000 EUR |

**Incremento estimado: ~69.500 EUR/mes** en revenue adicional gracias a las recomendaciones.

### Puntos clave para la presentacion

1. **Demo interactiva**: cargar el catalogo del cliente y mostrar recomendaciones en vivo.
2. **A/B testing**: el sistema se puede evaluar con A/B test antes de implementacion completa.
3. **Cold start resuelto**: nuevos usuarios reciben recomendaciones desde el primer momento (content-based), que mejoran a medida que interactuan (collaborative).
4. **Tiempo real**: las recomendaciones se sirven en <200ms, compatibles con cualquier frontend.
5. **Privacidad**: los datos de usuario no salen de la infraestructura del cliente.

### Preguntas frecuentes del cliente

- **"Tenemos pocos datos de usuarios"** - Empezamos con content-based (solo necesita catalogo de productos) y transicionamos a hibrido cuando hay suficientes interacciones.
- **"Ya tenemos recomendaciones basicas (mas vendidos)"** - Las recomendaciones personalizadas superan sistematicamente a las populares en engagement y conversion.
- **"Como medimos el impacto?"** - A/B test controlado: grupo A con recomendaciones, grupo B sin ellas. Medimos conversion, ticket medio y engagement.
