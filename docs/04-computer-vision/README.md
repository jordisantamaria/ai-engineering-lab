# Computer Vision

## Que es Computer Vision y Por Que Importa

Computer Vision (CV) permite a las maquinas "ver" e interpretar imagenes y video. En consultoria, es una de las areas con mayor impacto tangible porque automatiza tareas visuales que antes requerían inspeccion humana.

### Aplicaciones por Industria

| Industria | Aplicacion | Tecnica CV |
|---|---|---|
| Manufactura | Deteccion de defectos en linea de produccion | Object Detection / Clasificacion |
| Retail | Conteo de personas, analisis de estantes | Object Detection, Tracking |
| Salud | Analisis de radiografias, patologia digital | Clasificacion, Segmentacion |
| Agricultura | Deteccion de plagas, conteo de frutos | Object Detection, Segmentacion |
| Logistica | Lectura de matriculas, OCR de documentos | OCR, Object Detection |
| Construccion | Monitoreo de avance de obra | Segmentacion, Clasificacion |
| Seguros | Evaluacion de danos en vehiculos | Object Detection, Clasificacion |
| Seguridad | Deteccion de intrusos, reconocimiento facial | Detection, Reconocimiento |

---

## Imagenes como Datos

### Representacion de Imagenes

Una imagen digital es una matriz de numeros. Cada numero representa la intensidad de un pixel.

```
Grayscale (1 canal):           RGB (3 canales):
                                R        G        B
[120, 130, 125]             [255,0,0] [0,255,0] [0,0,255]
[118, 135, 128]             [128,0,0] [0,128,0] [0,0,128]
[122, 131, 127]             [64,0,0]  [0,64,0]  [0,0,64]
```

**Conceptos clave:**

| Concepto | Descripcion |
|---|---|
| **Pixel** | Unidad minima de una imagen. Valor entre 0 (negro) y 255 (blanco) |
| **Canales** | Grayscale = 1 canal, RGB = 3 canales, RGBA = 4 canales |
| **Resolucion** | Ancho x Alto en pixeles (ej: 1920x1080) |
| **Profundidad de color** | Bits por canal. 8 bits = 0-255, 16 bits = 0-65535 |

### Tensores de Imagenes en PyTorch

PyTorch usa la convencion **(B, C, H, W)**:

```
B = Batch size     (cuantas imagenes procesas a la vez)
C = Channels       (3 para RGB, 1 para grayscale)
H = Height         (alto en pixeles)
W = Width          (ancho en pixeles)
```

```python
import torch
from PIL import Image
from torchvision import transforms

# Cargar imagen y convertir a tensor
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # Convierte a tensor (C, H, W) y escala a [0, 1]
])

img = Image.open("foto.jpg")
tensor = transform(img)
print(tensor.shape)  # torch.Size([3, 224, 224])

# Crear un batch
batch = tensor.unsqueeze(0)  # Agrega dimension de batch
print(batch.shape)  # torch.Size([1, 3, 224, 224])
```

> **Punto clave:** `ToTensor()` hace dos cosas: convierte la imagen a tensor y escala los valores de [0, 255] a [0.0, 1.0].

### Normalizacion de Imagenes

Los modelos preentrenados en ImageNet esperan imagenes normalizadas con estas estadisticas:

```python
# Estadisticas de ImageNet
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],  # Media por canal (R, G, B)
    std=[0.229, 0.224, 0.225]    # Desviacion estandar por canal
)

# Pipeline completo tipico
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
```

La normalizacion centra los datos alrededor de 0 con desviacion ~1, lo que ayuda a que el entrenamiento sea mas estable. Si entrenas desde cero, puedes calcular tus propias estadisticas.

---

## CNNs (Convolutional Neural Networks)

Las CNNs son la arquitectura fundamental de CV. La idea central: usar filtros pequenos que se deslizan por la imagen detectando patrones locales.

### Convolucion: Filtros y Kernels

Un filtro (kernel) es una matriz pequena que se desliza por la imagen, multiplicando y sumando valores. Cada filtro detecta un patron especifico.

```
Imagen de entrada (5x5):        Filtro 3x3 (detector de bordes verticales):

1  0  1  0  1                    1  0  -1
0  1  0  1  0                    1  0  -1
1  0  1  0  1                    1  0  -1
0  1  0  1  0
1  0  1  0  1

Operacion de convolucion (posicion superior-izquierda):

  1*1 + 0*0 + 1*(-1)     =  0
+ 0*1 + 1*0 + 0*(-1)     =  0
+ 1*1 + 0*0 + 1*(-1)     =  0
                          ----
  Resultado en esa posicion: 0

El filtro se desliza por toda la imagen generando el "feature map":

Imagen 5x5  -->  [Filtro 3x3]  -->  Feature Map 3x3
```

La intuicion es simple: el filtro "busca" un patron especifico en cada posicion de la imagen. Si el patron esta presente, produce un valor alto; si no, un valor bajo o cero.

### Stride y Padding

```
Stride = 1 (por defecto):        Stride = 2 (salta posiciones):
El filtro se mueve 1 pixel        El filtro se mueve 2 pixeles
a la vez. Salida grande.          a la vez. Salida mas pequena.

Padding = 0 (valid):             Padding = 1 (same):
La salida se encoge.              Se agregan ceros alrededor.
5x5 input + 3x3 filtro           La salida mantiene el tamano
= 3x3 output                     de la entrada.

Formula del tamano de salida:
output_size = (input_size - kernel_size + 2 * padding) / stride + 1
```

### Pooling

Reduce el tamano espacial de los feature maps, manteniendo la informacion mas relevante.

```
Max Pooling 2x2 (stride=2):

[1  3 | 2  4]        [3  4]
[5  2 | 6  1]  -->   [5  8]
[-----+-----]
[3  1 | 8  2]
[4  5 | 3  7]

Toma el maximo de cada region 2x2.
Reduce el tamano a la mitad.
```

| Tipo | Que hace | Cuando usar |
|---|---|---|
| **Max Pooling** | Toma el maximo de cada region | Mas comun. Conserva las activaciones mas fuertes |
| **Average Pooling** | Promedio de cada region | Menos comun. Usado a veces en capas finales |
| **Global Average Pooling** | Promedio de todo el feature map | Reemplaza capas FC al final de la red |

### Feature Maps: Aprendizaje Jerarquico

La magia de las CNNs es que aprenden features de forma jerarquica, de lo simple a lo complejo:

```
Capa 1 (temprana):     Capa 2-3 (media):      Capa 4+ (profunda):
Detecta bordes,        Detecta texturas,       Detecta partes de
lineas, gradientes     patrones, formas        objetos y objetos
                       simples                 completos

  |  / -- \            /--\  |||               [ojo] [nariz]
  |  \ -- /            \--/  ===               [rueda] [ala]
```

> **Punto clave:** No necesitas disenar los filtros manualmente. La red los aprende durante el entrenamiento optimizando la funcion de perdida.

### Arquitectura Tipica de una CNN

```
Input Image
    |
    v
[Conv2d] --> [BatchNorm] --> [ReLU] --> [MaxPool]     Bloque 1
    |
    v
[Conv2d] --> [BatchNorm] --> [ReLU] --> [MaxPool]     Bloque 2
    |
    v
[Conv2d] --> [BatchNorm] --> [ReLU] --> [MaxPool]     Bloque 3
    |
    v
[Global Average Pooling]                               Reduccion
    |
    v
[Fully Connected] --> [Softmax/Sigmoid]                Clasificacion
    |
    v
Output (probabilidades por clase)
```

```python
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # Bloque 1: 3 canales -> 32 filtros
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Bloque 2: 32 -> 64 filtros
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Bloque 3: 64 -> 128 filtros
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # Global Average Pooling
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x
```

---

## Arquitecturas Importantes

### Tabla Comparativa

| Arquitectura | Ano | Parametros (M) | Top-1 ImageNet (%) | Velocidad relativa | Idea clave |
|---|---|---|---|---|---|
| LeNet-5 | 1998 | 0.06 | - | Muy rapida | Primera CNN exitosa (digitos) |
| AlexNet | 2012 | 61 | 63.3 | Rapida | Inicio del deep learning en CV |
| VGG-16 | 2014 | 138 | 74.4 | Lenta | Filtros 3x3 apilados, simple pero pesada |
| ResNet-50 | 2015 | 25.6 | 79.3 | Media | Residual connections |
| EfficientNet-B0 | 2019 | 5.3 | 77.1 | Rapida | Compound scaling |
| EfficientNet-B7 | 2019 | 66 | 84.3 | Lenta | Mejor accuracy pre-ViT |
| ViT-B/16 | 2020 | 86 | 81.8 | Media | Transformers para vision |
| ConvNeXt-T | 2022 | 29 | 82.1 | Media | CNN modernizada, compite con ViT |

### ResNet: Residual Connections

El problema: redes mas profundas deberian ser mejores, pero en la practica empeoran (degradation problem). No es overfitting, sino que es dificil optimizar redes muy profundas.

La solucion de ResNet: **skip connections** (conexiones residuales).

```
Bloque estandar:                Bloque residual (ResNet):

x --> [Conv] --> [Conv] --> y   x ----> [Conv] --> [Conv] --> (+) --> y
                                 |                            ^
                                 |____________________________|
                                        skip connection

y = F(x)                       y = F(x) + x

En lugar de aprender y = F(x),  la red aprende el "residuo": F(x) = y - x
Es mas facil aprender un residuo pequeno que una transformacion completa.
```

> **Intuicion:** si la capa ideal es la identidad (no hacer nada), es mucho mas facil aprender F(x) = 0 y obtener y = x, que aprender F(x) = x directamente. Esto permite entrenar redes de 100+ capas.

### EfficientNet: Scaling Compuesto

EfficientNet propone escalar tres dimensiones a la vez en proporciones optimas:

```
Dimensiones de escalado:

Ancho (width):    Mas filtros por capa        [32] -> [64]
Profundidad:      Mas capas                   3 bloques -> 6 bloques
Resolucion:       Imagenes mas grandes        224x224 -> 380x380

EfficientNet escala las tres de forma equilibrada:
ancho = alpha^phi
profundidad = beta^phi
resolucion = gamma^phi

donde phi es el coeficiente de escalado (B0=0, B1=1, ..., B7=7)
```

### Vision Transformer (ViT)

ViT aplica la arquitectura Transformer (de NLP) directamente a imagenes:

```
Imagen 224x224
      |
      v
Dividir en parches 16x16 = 196 parches
      |
      v
Cada parche se aplana y proyecta a un embedding (como un "token")
      |
      v
Agregar positional embeddings + [CLS] token
      |
      v
Transformer Encoder (self-attention entre parches)
      |
      v
[CLS] token --> Clasificacion
```

> **Cuando usar ViT:** funciona mejor con grandes cantidades de datos. Para datasets pequenos, ResNet y EfficientNet suelen ganar. ConvNeXt es una alternativa que combina lo mejor de ambos mundos.

---

## Transfer Learning en CV

Transfer Learning es **la tecnica mas importante en CV practico**. En lugar de entrenar desde cero, usas un modelo preentrenado y lo adaptas a tu problema.

### Por Que Funciona

Los primeros layers de una CNN aprenden features generales (bordes, texturas) utiles para cualquier tarea visual. Solo los ultimos layers son especificos del problema.

### Uso con torchvision

```python
import torchvision.models as models
import torch.nn as nn

# 1. Cargar modelo preentrenado
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

# 2. Congelar todas las capas
for param in model.parameters():
    param.requires_grad = False

# 3. Reemplazar la ultima capa (classifier head)
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Linear(256, num_classes)  # Tu numero de clases
)

# 4. Solo los parametros nuevos se entrenan
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
```

### Fine-Tuning Avanzado: Learning Rates por Capa

```python
# Descongelar gradualmente y usar learning rates diferentes
# Las capas mas profundas (cerca de la salida) aprenden mas rapido
# Las capas tempranas (features generales) se ajustan menos

param_groups = [
    {"params": model.layer1.parameters(), "lr": 1e-5},   # Casi congelado
    {"params": model.layer2.parameters(), "lr": 5e-5},
    {"params": model.layer3.parameters(), "lr": 1e-4},
    {"params": model.layer4.parameters(), "lr": 5e-4},
    {"params": model.fc.parameters(), "lr": 1e-3},       # Aprende mas
]

optimizer = torch.optim.Adam(param_groups)
```

### Cuando Funciona Transfer Learning

| Escenario | Datos | Similitud con ImageNet | Estrategia |
|---|---|---|---|
| **Ideal** | Pocos datos (<1K) | Alta (fotos naturales) | Congelar todo, solo entrenar head |
| **Comun** | Datos medios (1K-10K) | Media | Fine-tune ultimas capas + head |
| **Mucho dato** | Muchos datos (>50K) | Baja (medico, satelital) | Fine-tune completo, lr bajo en primeras capas |
| **Dominio muy diferente** | Muchos datos | Muy baja | Considerar entrenar desde cero |

---

## Data Augmentation

Data augmentation genera variaciones artificiales de tus imagenes de entrenamiento, aumentando la diversidad sin necesidad de mas datos reales.

### torchvision.transforms

```python
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2),  # Cutout-like
])

# Para validacion/test: NO augmentation, solo resize y normalize
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
```

### Albumentations (Mas Potente)

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
    A.RandomResizedCrop(224, 224, scale=(0.8, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),  # Util para satelital, microscopia
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
    A.OneOf([
        A.GaussNoise(var_limit=(10, 50)),
        A.GaussianBlur(blur_limit=(3, 7)),
        A.MotionBlur(blur_limit=7),
    ], p=0.3),
    A.CLAHE(clip_limit=4.0, p=0.3),  # Mejora contraste
    A.CoarseDropout(  # Cutout
        max_holes=8, max_height=32, max_width=32,
        fill_value=0, p=0.3
    ),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])
```

> **Albumentations vs torchvision:** Albumentations es mas rapido (implementado en C/OpenCV), tiene mas augmentations, y soporta augmentations para bounding boxes y mascaras de segmentacion. **Recomendado para proyectos serios.**

### Augmentations Comunes

| Augmentation | Descripcion | Cuando usar |
|---|---|---|
| **HorizontalFlip** | Espejo horizontal | Casi siempre (no en OCR/texto) |
| **VerticalFlip** | Espejo vertical | Satelital, microscopia, aereo |
| **Rotation** | Rotar imagen | Cuando la orientacion no importa |
| **RandomCrop** | Recortar region aleatoria | Siempre util |
| **ColorJitter** | Variar brillo/contraste/saturacion | Fotos naturales |
| **Cutout/Erasing** | Borrar regiones aleatorias | Regularizacion efectiva |
| **MixUp** | Mezclar dos imagenes y sus labels | Regularizacion avanzada |
| **CutMix** | Pegar un trozo de otra imagen | Similar a MixUp, a veces mejor |

### Test Time Augmentation (TTA)

Aplicar augmentations tambien en inference y promediar predicciones. Mejora accuracy ~1-3% a costa de mas tiempo.

```python
def predict_with_tta(model, image, transforms_list, n_augments=5):
    """Predice con TTA: aplica augmentations y promedia."""
    predictions = []

    # Prediccion sin augmentation
    pred = model(original_transform(image).unsqueeze(0))
    predictions.append(pred)

    # Predicciones con augmentations
    for _ in range(n_augments):
        aug_image = tta_transform(image)
        pred = model(aug_image.unsqueeze(0))
        predictions.append(pred)

    # Promediar probabilidades
    avg_pred = torch.stack(predictions).mean(dim=0)
    return avg_pred
```

---

## Tareas de Computer Vision

### Clasificacion de Imagenes

La tarea mas basica: dado una imagen, predecir su clase.

```python
# Single-label: una clase por imagen (softmax)
# Multi-label: multiples clases por imagen (sigmoid)

# Con modelo preentrenado
from torchvision import models

# Single-label
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)
criterion = nn.CrossEntropyLoss()

# Multi-label
model.fc = nn.Linear(model.fc.in_features, num_labels)
criterion = nn.BCEWithLogitsLoss()
```

### Object Detection

Detectar Y localizar objetos en una imagen con bounding boxes.

**Conceptos Clave:**

```
Bounding Box: [x_min, y_min, x_max, y_max] o [x_center, y_center, width, height]

IoU (Intersection over Union):
                 Area de interseccion
   IoU = ---------------------------------
           Area de union de ambos boxes

   IoU = 1.0 -> Cajas identicas
   IoU = 0.0 -> Sin solapamiento
   IoU > 0.5 -> Generalmente se considera "correcto"

NMS (Non-Maximum Suppression):
   Cuando el modelo predice multiples boxes para el mismo objeto,
   NMS elimina los duplicados quedandose con el de mayor confianza.

Anchor Boxes:
   Boxes predefinidos de diferentes tamaños y proporciones
   que sirven como punto de partida para las predicciones.
```

**YOLO (You Only Look Once) - El mas Practico:**

```python
# YOLOv8 con ultralytics - la forma mas rapida de hacer detection
from ultralytics import YOLO

# Cargar modelo preentrenado
model = YOLO("yolov8n.pt")  # nano (rapido), s, m, l, x (preciso)

# Entrenar con tus datos
model.train(
    data="mi_dataset.yaml",  # Formato YOLO
    epochs=100,
    imgsz=640,
    batch=16,
    patience=20,
)

# Inference
results = model("imagen.jpg")
for result in results:
    boxes = result.boxes  # Bounding boxes
    for box in boxes:
        xyxy = box.xyxy[0]      # Coordenadas [x1, y1, x2, y2]
        conf = box.conf[0]      # Confianza
        cls = box.cls[0]        # Clase

# Exportar para produccion
model.export(format="onnx")
```

**YOLO vs Faster R-CNN:**

| Aspecto | YOLO (v8) | Faster R-CNN |
|---|---|---|
| Enfoque | One-stage (una pasada) | Two-stage (RPN + clasificacion) |
| Velocidad | Muy rapido (~30+ FPS) | Mas lento (~5-15 FPS) |
| Precision | Muy buena | Ligeramente mejor en objetos pequenos |
| Facilidad de uso | Muy facil (ultralytics) | Mas complejo (detectron2/torchvision) |
| Mejor para | Tiempo real, edge, proyectos rapidos | Maxima precision, objetos densos |

**Metricas de Detection:**

| Metrica | Descripcion |
|---|---|
| **mAP** | Mean Average Precision: promedio del AP de todas las clases |
| **AP50** | AP con umbral IoU = 0.5 (el mas usado) |
| **AP75** | AP con umbral IoU = 0.75 (mas estricto) |
| **mAP@[.5:.95]** | Promedio de AP en diferentes umbrales IoU (metrica COCO) |

### Segmentacion

Clasificar cada pixel de la imagen.

```
Clasificacion:    "Hay un gato"
Detection:        "Hay un gato AQUI" (bounding box)
Segmentacion:     "ESTOS pixeles son el gato" (mascara)
```

**Tipos de Segmentacion:**

| Tipo | Descripcion | Modelo tipico | Uso |
|---|---|---|---|
| **Semantica** | Cada pixel tiene una clase, no distingue instancias | U-Net, DeepLab | Medico, satelital |
| **De instancia** | Distingue entre instancias individuales | Mask R-CNN | Conteo, robotica |
| **Panoptica** | Combina semantica + instancia | Mask2Former | Lo mas completo |

**U-Net (Segmentacion Semantica):**

```
Arquitectura U-Net (encoder-decoder con skip connections):

Encoder (downsample)                    Decoder (upsample)
    |                                        |
[Input] --> [Conv,Conv] ------skip------> [Conv,Conv] --> [Output]
                |                              ^
            [MaxPool]                      [UpConv]
                |                              ^
            [Conv,Conv] ------skip------> [Conv,Conv]
                |                              ^
            [MaxPool]                      [UpConv]
                |                              ^
            [Conv,Conv] ----> Bottleneck --> [Conv,Conv]

Las skip connections pasan informacion de alta resolucion
del encoder al decoder, preservando detalles finos.
```

### OCR y Document AI

| Herramienta | Tipo | Velocidad | Precision | Idiomas | Mejor para |
|---|---|---|---|---|---|
| **Tesseract** | Open source clasico | Rapido | Media | 100+ | OCR basico, documentos limpios |
| **EasyOCR** | Deep learning, simple | Medio | Buena | 80+ | Texto en escenas, multiples idiomas |
| **PaddleOCR** | Deep learning, completo | Rapido | Muy buena | 80+ | Produccion, documentos complejos |
| **AWS Textract** | Cloud API | N/A | Excelente | Limitados | Formularios, tablas, integrado AWS |
| **Google Vision** | Cloud API | N/A | Excelente | 100+ | Maxima precision, OCR general |
| **Azure Doc Intelligence** | Cloud API | N/A | Excelente | Muchos | Documentos empresariales |

```python
# EasyOCR - rapido de implementar
import easyocr

reader = easyocr.Reader(['es', 'en'])  # Espanol e ingles
results = reader.readtext('documento.jpg')

for (bbox, text, confidence) in results:
    print(f"Texto: {text}, Confianza: {confidence:.2f}")

# PaddleOCR - mas robusto para produccion
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang='es')
result = ocr.ocr('documento.jpg', cls=True)
```

**Document Layout Analysis:**

Para documentos complejos (facturas, formularios), no basta con OCR. Necesitas entender la estructura:

1. **Detectar regiones:** titulos, parrafos, tablas, figuras
2. **Extraer texto** de cada region
3. **Entender relaciones** entre regiones (que campo va con que valor)

Herramientas: LayoutLM (Microsoft), Donut (sin OCR, end-to-end), DocTR.

---

## Datasets Populares para Practicar

| Dataset | Tamano | Tarea | Nivel | Notas |
|---|---|---|---|---|
| **MNIST** | 70K imagenes 28x28 | Clasificacion digitos | Principiante | El "Hello World" de CV |
| **CIFAR-10** | 60K imagenes 32x32 | Clasificacion 10 clases | Principiante | Imagenes pequenas, util para experimentar |
| **ImageNet** | 1.2M imagenes | Clasificacion 1000 clases | Referencia | El benchmark estandar |
| **COCO** | 330K imagenes | Detection + Segmentacion | Intermedio | El mas usado para detection |
| **Pascal VOC** | 11K imagenes | Detection + Segmentacion | Intermedio | Clasico, mas pequeno que COCO |
| **Open Images** | 9M imagenes | Detection + Segmentacion | Avanzado | Enorme, muchas clases |

```python
# Cargar datasets con torchvision
from torchvision import datasets

# CIFAR-10
train = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)

# Para datasets custom: ImageFolder
# Estructura: root/class_name/image.jpg
train = datasets.ImageFolder(root="./data/train", transform=transform)
```

**Roboflow** para datasets custom: plataforma que te permite buscar datasets publicos, hacer labeling, aplicar augmentations, y exportar en cualquier formato (YOLO, COCO, VOC, etc.).

---

## Labeling de Datos

### Herramientas de Labeling

| Herramienta | Tipo | Tareas | Costo | Mejor para |
|---|---|---|---|---|
| **Label Studio** | Open source / Cloud | Todas | Gratis / Pago | Versatil, todo tipo de datos |
| **CVAT** | Open source | CV (boxes, segmentacion) | Gratis | Labeling de imagenes/video |
| **Roboflow** | Cloud | CV | Freemium | Workflow completo CV |
| **V7** | Cloud | CV + Video | Pago | Equipos grandes, auto-labeling |

### Tips para Labeling Eficiente

1. **Definir guia de anotacion clara** antes de empezar (que es cada clase, bordes ambiguos)
2. **Empezar con pocas clases** e ir expandiendo
3. **Medir inter-annotator agreement** si hay multiples personas anotando
4. **Pre-anotar con un modelo** y luego corregir (semi-automatico)
5. **Iteraciones rapidas:** anotar pocas imagenes -> entrenar -> evaluar -> anotar mas

### Active Learning

Concepto: en lugar de anotar imagenes al azar, dejar que el modelo te diga cuales imagenes son mas "utiles" para anotar (las que tienen mas incertidumbre). Reduce el labeling necesario en un 30-70%.

```
Ciclo de Active Learning:
1. Entrenar modelo con datos anotados existentes
2. Predecir sobre datos sin anotar
3. Seleccionar las imagenes con mayor incertidumbre
4. Anotar esas imagenes (las mas informativas)
5. Re-entrenar el modelo
6. Repetir
```

---

## Tips Practicos para Proyectos CV en Consultoria

### Siempre Empezar con Transfer Learning

Nunca entrenes desde cero a menos que tengas una muy buena razon. Un ResNet50 preentrenado con 100 imagenes te va a dar mejores resultados que una CNN custom con 10,000.

### Cuantos Datos Necesitas (Reglas de Oro)

| Tarea | Datos minimos viables | Datos recomendados |
|---|---|---|
| Clasificacion binaria (transfer learning) | 50-100 por clase | 500+ por clase |
| Clasificacion multi-clase | 100+ por clase | 1000+ por clase |
| Object Detection | 200+ bounding boxes por clase | 1000+ por clase |
| Segmentacion semantica | 100+ mascaras | 500+ mascaras |

### Datasets Desbalanceados en CV

```python
# Opcion 1: Weighted sampler (mas comun en CV)
from torch.utils.data import WeightedRandomSampler

class_counts = [1000, 200, 50]  # Imagenes por clase
weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
sample_weights = weights[targets]  # targets = lista de labels

sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
loader = DataLoader(dataset, batch_size=32, sampler=sampler)

# Opcion 2: Class weights en la loss
weights = torch.tensor([1.0, 5.0, 20.0])  # Inverso de frecuencia
criterion = nn.CrossEntropyLoss(weight=weights)

# Opcion 3: Data augmentation agresiva en clases minoritarias
```

### Edge Deployment

Para llevar modelos a dispositivos (movil, IoT, camaras):

| Framework | Descripcion | Mejor para |
|---|---|---|
| **ONNX** | Formato estandar de intercambio | Interoperabilidad entre frameworks |
| **TensorRT** | Optimizador de NVIDIA | GPUs NVIDIA (maxima velocidad) |
| **OpenVINO** | Optimizador de Intel | CPUs Intel, edge devices |
| **Core ML** | Framework de Apple | iOS/macOS |
| **TFLite** | TensorFlow Lite | Android, microcontroladores |

```python
# Exportar a ONNX desde PyTorch
import torch.onnx

dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "modelo.onnx",
                  input_names=["input"],
                  output_names=["output"],
                  dynamic_axes={"input": {0: "batch_size"},
                               "output": {0: "batch_size"}})
```

---

## Checklist de Proyecto CV

```
[ ] Definir el problema con el cliente (clasificacion? detection? segmentacion?)
[ ] Recopilar y explorar datos (calidad, cantidad, balance)
[ ] Definir metrica de exito con el cliente
[ ] Elegir modelo base (empezar con pretrained)
[ ] Configurar data augmentation
[ ] Entrenar con transfer learning
[ ] Evaluar en validation set
[ ] Iterar: mas datos? mejor augmentation? modelo mas grande?
[ ] Interpretar resultados (confusion matrix, errores comunes)
[ ] Optimizar para deployment si es necesario (ONNX, quantizacion)
[ ] Monitorear en produccion (data drift visual)
```
