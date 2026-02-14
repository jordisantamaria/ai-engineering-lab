# Extraccion automatica de datos de documentos

## Problema de negocio

El procesamiento manual de documentos (facturas, albaranes, pedidos, contratos) es una tarea que consume tiempo y recursos en cualquier organizacion:

- **Tedioso**: un empleado tarda 3-5 minutos por factura para introducir datos manualmente en el ERP.
- **Propenso a errores**: la tasa de error humano en entrada de datos es del 1-4%, lo que genera discrepancias contables y problemas de auditoria.
- **No escalable**: en picos de actividad (cierre mensual, campanas) se acumulan documentos sin procesar.
- **Coste oculto**: el tiempo dedicado a tareas repetitivas impide que el personal se dedique a tareas de mayor valor.

## Solucion propuesta

Pipeline automatizado de extraccion de informacion que combina OCR (reconocimiento optico de caracteres) con NLP (procesamiento de lenguaje natural) para extraer campos clave de documentos de forma automatica.

### Arquitectura

```
Documento (imagen/PDF)
        |
        v
  Preprocesamiento
  (deskew, denoise, binarizacion)
        |
        v
  Motor OCR (EasyOCR / PaddleOCR)
        |
        v
  Texto + Bounding Boxes + Confianza
        |
        v
  Extractor de Campos (regex + heuristica)
        |
        v
  Datos Estructurados (JSON)
  - Fecha
  - Numero de factura
  - Proveedor
  - Importe total
  - Conceptos / lineas de detalle
```

### Componentes clave

1. **Preprocesamiento de imagen**: correccion de inclinacion, eliminacion de ruido y binarizacion para maximizar la precision del OCR.
2. **Motor OCR dual**: soporte para EasyOCR (mas sencillo) y PaddleOCR (mas preciso en documentos complejos).
3. **Extractor inteligente**: combinacion de patrones regex para campos estandar (fechas, importes) con heuristicas posicionales (el total suele estar en la parte inferior, el proveedor en la cabecera).

## Resultados esperados

| Metrica | Valor |
|---------|-------|
| Precision extraccion de campos | >90% |
| Tiempo por documento | <3 segundos |
| Tipos de documento soportados | Facturas, albaranes, pedidos |
| Idiomas OCR | Espanol, Ingles, Catalan |

## Tecnologias

- **EasyOCR**: motor OCR open-source con soporte multi-idioma
- **PaddleOCR**: motor OCR de alto rendimiento de Baidu
- **OpenCV**: preprocesamiento de imagenes
- **FastAPI**: API REST para integracion
- **Pydantic**: validacion de datos estructurados

## Como ejecutar

### 1. Instalacion

```bash
cd portfolio/02-document-ai
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Lanzar la API

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8001
```

### 3. Extraer datos de un documento

```bash
curl -X POST "http://localhost:8001/extract" \
    -F "file=@factura_ejemplo.jpg"
```

Respuesta esperada:
```json
{
    "invoice_number": "FAC-2024-001234",
    "date": "2024-03-15",
    "vendor_name": "Suministros Industriales S.L.",
    "total_amount": 1542.30,
    "line_items": [
        {"description": "Tornillos M8x40", "quantity": 500, "unit_price": 0.12, "amount": 60.00},
        {"description": "Tuercas M8", "quantity": 500, "unit_price": 0.08, "amount": 40.00}
    ],
    "confidence": 0.92
}
```

### 4. Docker

```bash
docker build -t document-ai .
docker run -p 8001:8001 document-ai
```

## Como presentarlo: pitch para cliente

### Propuesta de valor

> "Transforme pilas de facturas en datos estructurados en segundos, no en horas. Nuestro sistema extrae automaticamente la informacion clave de cualquier documento, reduciendo errores y liberando a su equipo para tareas estrategicas."

### ROI estimado

**Escenario**: empresa que procesa 2.000 facturas/mes con 2 empleados dedicados.

| Concepto | Antes | Despues |
|----------|-------|---------|
| Tiempo por factura | 3-5 minutos | <10 segundos |
| Horas mensuales dedicadas | ~130 horas | ~15 horas (revision) |
| Tasa de error | 2-4% | <0.5% |
| Coste mensual proceso | ~4.000 EUR | ~1.200 EUR |

**Ahorro estimado: ~33.600 EUR/ano** en coste de procesamiento, mas la eliminacion de costes por errores (discrepancias, reclamaciones, retrabajos).

### Puntos clave para la presentacion

1. **Demo con documentos reales**: pedir al cliente que traiga 2-3 facturas de su dia a dia y procesarlas en directo.
2. **Integracion con ERP**: los datos extraidos se pueden enviar directamente a SAP, Navision, Sage, etc.
3. **Aprendizaje continuo**: el sistema mejora con el feedback del usuario (correcciones).
4. **Cumplimiento**: trazabilidad completa del procesamiento para auditorias.
5. **Multi-formato**: funciona con escaneos, fotos de movil y PDFs.

### Preguntas frecuentes del cliente

- **"Funciona con nuestras facturas?"** - Si, el sistema se adapta a cualquier formato. En la fase de piloto se calibra con sus documentos especificos.
- **"Que pasa si el OCR falla?"** - Los documentos con baja confianza (<80%) se marcan para revision humana. El sistema prioriza precision sobre cobertura.
- **"Se puede integrar con nuestro ERP?"** - Si, la API devuelve JSON estructurado que se mapea a los campos de cualquier ERP via integracion estandar.
