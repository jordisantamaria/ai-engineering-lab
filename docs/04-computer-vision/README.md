# Computer Vision

## What is Computer Vision and Why It Matters

Computer Vision (CV) allows machines to "see" and interpret images and video. In consulting, it is one of the areas with the greatest tangible impact because it automates visual tasks that previously required human inspection.

### Applications by Industry

| Industry | Application | CV Technique |
|---|---|---|
| Manufacturing | Defect detection on production lines | Object Detection / Classification |
| Retail | People counting, shelf analysis | Object Detection, Tracking |
| Healthcare | X-ray analysis, digital pathology | Classification, Segmentation |
| Agriculture | Pest detection, fruit counting | Object Detection, Segmentation |
| Logistics | License plate reading, document OCR | OCR, Object Detection |
| Construction | Construction progress monitoring | Segmentation, Classification |
| Insurance | Vehicle damage assessment | Object Detection, Classification |
| Security | Intrusion detection, face recognition | Detection, Recognition |

---

## Images as Data

### Image Representation

A digital image is a matrix of numbers. Each number represents the intensity of a pixel.

```
Grayscale (1 channel):           RGB (3 channels):
                                R        G        B
[120, 130, 125]             [255,0,0] [0,255,0] [0,0,255]
[118, 135, 128]             [128,0,0] [0,128,0] [0,0,128]
[122, 131, 127]             [64,0,0]  [0,64,0]  [0,0,64]
```

**Key concepts:**

| Concept | Description |
|---|---|
| **Pixel** | Smallest unit of an image. Value between 0 (black) and 255 (white) |
| **Channels** | Grayscale = 1 channel, RGB = 3 channels, RGBA = 4 channels |
| **Resolution** | Width x Height in pixels (e.g., 1920x1080) |
| **Color depth** | Bits per channel. 8 bits = 0-255, 16 bits = 0-65535 |

### Image Tensors in PyTorch

PyTorch uses the convention **(B, C, H, W)**:

```
B = Batch size     (how many images you process at once)
C = Channels       (3 for RGB, 1 for grayscale)
H = Height         (height in pixels)
W = Width          (width in pixels)
```

```python
import torch
from PIL import Image
from torchvision import transforms

# Load image and convert to tensor
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # Converts to tensor (C, H, W) and scales to [0, 1]
])

img = Image.open("photo.jpg")
tensor = transform(img)
print(tensor.shape)  # torch.Size([3, 224, 224])

# Create a batch
batch = tensor.unsqueeze(0)  # Add batch dimension
print(batch.shape)  # torch.Size([1, 3, 224, 224])
```

> **Key point:** `ToTensor()` does two things: converts the image to a tensor and scales values from [0, 255] to [0.0, 1.0].

### Image Normalization

Models pretrained on ImageNet expect images normalized with these statistics:

```python
# ImageNet statistics
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],  # Mean per channel (R, G, B)
    std=[0.229, 0.224, 0.225]    # Standard deviation per channel
)

# Typical complete pipeline
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
```

Normalization centers the data around 0 with deviation ~1, which helps make training more stable. If you train from scratch, you can compute your own statistics.

---

## CNNs (Convolutional Neural Networks)

CNNs are the fundamental architecture of CV. The central idea: use small filters that slide across the image detecting local patterns.

### Convolution: Filters and Kernels

A filter (kernel) is a small matrix that slides across the image, multiplying and summing values. Each filter detects a specific pattern.

```
Input image (5x5):              Filter 3x3 (vertical edge detector):

1  0  1  0  1                    1  0  -1
0  1  0  1  0                    1  0  -1
1  0  1  0  1                    1  0  -1
0  1  0  1  0
1  0  1  0  1

Convolution operation (top-left position):

  1*1 + 0*0 + 1*(-1)     =  0
+ 0*1 + 1*0 + 0*(-1)     =  0
+ 1*1 + 0*0 + 1*(-1)     =  0
                          ----
  Result at that position: 0

The filter slides across the entire image generating the "feature map":

Image 5x5  -->  [Filter 3x3]  -->  Feature Map 3x3
```

The intuition is simple: the filter "looks for" a specific pattern at each position of the image. If the pattern is present, it produces a high value; if not, a low value or zero.

### Stride and Padding

```
Stride = 1 (default):           Stride = 2 (skips positions):
The filter moves 1 pixel         The filter moves 2 pixels
at a time. Large output.         at a time. Smaller output.

Padding = 0 (valid):            Padding = 1 (same):
The output shrinks.              Zeros are added around.
5x5 input + 3x3 filter          The output maintains the
= 3x3 output                    size of the input.

Output size formula:
output_size = (input_size - kernel_size + 2 * padding) / stride + 1
```

### Pooling

Reduces the spatial size of feature maps, keeping the most relevant information.

```
Max Pooling 2x2 (stride=2):

[1  3 | 2  4]        [3  4]
[5  2 | 6  1]  -->   [5  8]
[-----+-----]
[3  1 | 8  2]
[4  5 | 3  7]

Takes the maximum of each 2x2 region.
Reduces the size by half.
```

| Type | What it does | When to use |
|---|---|---|
| **Max Pooling** | Takes the maximum of each region | Most common. Preserves the strongest activations |
| **Average Pooling** | Average of each region | Less common. Sometimes used in final layers |
| **Global Average Pooling** | Average of the entire feature map | Replaces FC layers at the end of the network |

### Feature Maps: Hierarchical Learning

The magic of CNNs is that they learn features hierarchically, from simple to complex:

```
Layer 1 (early):       Layer 2-3 (middle):     Layer 4+ (deep):
Detects edges,         Detects textures,       Detects object parts
lines, gradients       patterns, simple        and complete objects
                       shapes

  |  / -- \            /--\  |||               [eye] [nose]
  |  \ -- /            \--/  ===               [wheel] [wing]
```

> **Key point:** You don't need to design filters manually. The network learns them during training by optimizing the loss function.

### Typical CNN Architecture

```
Input Image
    |
    v
[Conv2d] --> [BatchNorm] --> [ReLU] --> [MaxPool]     Block 1
    |
    v
[Conv2d] --> [BatchNorm] --> [ReLU] --> [MaxPool]     Block 2
    |
    v
[Conv2d] --> [BatchNorm] --> [ReLU] --> [MaxPool]     Block 3
    |
    v
[Global Average Pooling]                               Reduction
    |
    v
[Fully Connected] --> [Softmax/Sigmoid]                Classification
    |
    v
Output (probabilities per class)
```

```python
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 3 channels -> 32 filters
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2: 32 -> 64 filters
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3: 64 -> 128 filters
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

## Important Architectures

### Comparison Table

| Architecture | Year | Parameters (M) | Top-1 ImageNet (%) | Relative Speed | Key Idea |
|---|---|---|---|---|---|
| LeNet-5 | 1998 | 0.06 | - | Very fast | First successful CNN (digits) |
| AlexNet | 2012 | 61 | 63.3 | Fast | Start of deep learning in CV |
| VGG-16 | 2014 | 138 | 74.4 | Slow | Stacked 3x3 filters, simple but heavy |
| ResNet-50 | 2015 | 25.6 | 79.3 | Medium | Residual connections |
| EfficientNet-B0 | 2019 | 5.3 | 77.1 | Fast | Compound scaling |
| EfficientNet-B7 | 2019 | 66 | 84.3 | Slow | Best accuracy pre-ViT |
| ViT-B/16 | 2020 | 86 | 81.8 | Medium | Transformers for vision |
| ConvNeXt-T | 2022 | 29 | 82.1 | Medium | Modernized CNN, competes with ViT |

### ResNet: Residual Connections

The problem: deeper networks should be better, but in practice they get worse (degradation problem). It's not overfitting, but rather it's difficult to optimize very deep networks.

ResNet's solution: **skip connections** (residual connections).

```
Standard block:                 Residual block (ResNet):

x --> [Conv] --> [Conv] --> y   x ----> [Conv] --> [Conv] --> (+) --> y
                                 |                            ^
                                 |____________________________|
                                        skip connection

y = F(x)                       y = F(x) + x

Instead of learning y = F(x),  the network learns the "residual": F(x) = y - x
It's easier to learn a small residual than a complete transformation.
```

> **Intuition:** if the ideal layer is the identity (do nothing), it's much easier to learn F(x) = 0 and get y = x, than to learn F(x) = x directly. This allows training networks with 100+ layers.

### EfficientNet: Compound Scaling

EfficientNet proposes scaling three dimensions at once in optimal proportions:

```
Scaling dimensions:

Width:        More filters per layer        [32] -> [64]
Depth:        More layers                   3 blocks -> 6 blocks
Resolution:   Larger images                 224x224 -> 380x380

EfficientNet scales all three in a balanced way:
width = alpha^phi
depth = beta^phi
resolution = gamma^phi

where phi is the scaling coefficient (B0=0, B1=1, ..., B7=7)
```

### Vision Transformer (ViT)

ViT applies the Transformer architecture (from NLP) directly to images:

```
Image 224x224
      |
      v
Split into 16x16 patches = 196 patches
      |
      v
Each patch is flattened and projected to an embedding (like a "token")
      |
      v
Add positional embeddings + [CLS] token
      |
      v
Transformer Encoder (self-attention between patches)
      |
      v
[CLS] token --> Classification
```

> **When to use ViT:** works better with large amounts of data. For small datasets, ResNet and EfficientNet usually win. ConvNeXt is an alternative that combines the best of both worlds.

---

## Transfer Learning in CV

Transfer Learning is **the most important technique in practical CV**. Instead of training from scratch, you use a pretrained model and adapt it to your problem.

### Why It Works

The first layers of a CNN learn general features (edges, textures) useful for any visual task. Only the last layers are problem-specific.

### Usage with torchvision

```python
import torchvision.models as models
import torch.nn as nn

# 1. Load pretrained model
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

# 2. Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# 3. Replace the last layer (classifier head)
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Linear(256, num_classes)  # Your number of classes
)

# 4. Only the new parameters are trained
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
```

### Advanced Fine-Tuning: Per-Layer Learning Rates

```python
# Gradually unfreeze and use different learning rates
# Deeper layers (near the output) learn faster
# Early layers (general features) are adjusted less

param_groups = [
    {"params": model.layer1.parameters(), "lr": 1e-5},   # Almost frozen
    {"params": model.layer2.parameters(), "lr": 5e-5},
    {"params": model.layer3.parameters(), "lr": 1e-4},
    {"params": model.layer4.parameters(), "lr": 5e-4},
    {"params": model.fc.parameters(), "lr": 1e-3},       # Learns more
]

optimizer = torch.optim.Adam(param_groups)
```

### When Transfer Learning Works

| Scenario | Data | Similarity to ImageNet | Strategy |
|---|---|---|---|
| **Ideal** | Few data (<1K) | High (natural photos) | Freeze everything, only train head |
| **Common** | Medium data (1K-10K) | Medium | Fine-tune last layers + head |
| **Lots of data** | Many data (>50K) | Low (medical, satellite) | Full fine-tune, low lr on first layers |
| **Very different domain** | Many data | Very low | Consider training from scratch |

---

## Data Augmentation

Data augmentation generates artificial variations of your training images, increasing diversity without needing more real data.

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

# For validation/test: NO augmentation, only resize and normalize
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
```

### Albumentations (More Powerful)

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
    A.RandomResizedCrop(224, 224, scale=(0.8, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),  # Useful for satellite, microscopy
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
    A.OneOf([
        A.GaussNoise(var_limit=(10, 50)),
        A.GaussianBlur(blur_limit=(3, 7)),
        A.MotionBlur(blur_limit=7),
    ], p=0.3),
    A.CLAHE(clip_limit=4.0, p=0.3),  # Contrast enhancement
    A.CoarseDropout(  # Cutout
        max_holes=8, max_height=32, max_width=32,
        fill_value=0, p=0.3
    ),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])
```

> **Albumentations vs torchvision:** Albumentations is faster (implemented in C/OpenCV), has more augmentations, and supports augmentations for bounding boxes and segmentation masks. **Recommended for serious projects.**

### Common Augmentations

| Augmentation | Description | When to use |
|---|---|---|
| **HorizontalFlip** | Horizontal mirror | Almost always (not for OCR/text) |
| **VerticalFlip** | Vertical mirror | Satellite, microscopy, aerial |
| **Rotation** | Rotate image | When orientation doesn't matter |
| **RandomCrop** | Crop random region | Always useful |
| **ColorJitter** | Vary brightness/contrast/saturation | Natural photos |
| **Cutout/Erasing** | Erase random regions | Effective regularization |
| **MixUp** | Blend two images and their labels | Advanced regularization |
| **CutMix** | Paste a piece of another image | Similar to MixUp, sometimes better |

### Test Time Augmentation (TTA)

Apply augmentations also during inference and average predictions. Improves accuracy ~1-3% at the cost of more time.

```python
def predict_with_tta(model, image, transforms_list, n_augments=5):
    """Predict with TTA: apply augmentations and average."""
    predictions = []

    # Prediction without augmentation
    pred = model(original_transform(image).unsqueeze(0))
    predictions.append(pred)

    # Predictions with augmentations
    for _ in range(n_augments):
        aug_image = tta_transform(image)
        pred = model(aug_image.unsqueeze(0))
        predictions.append(pred)

    # Average probabilities
    avg_pred = torch.stack(predictions).mean(dim=0)
    return avg_pred
```

---

## Computer Vision Tasks

### Image Classification

The most basic task: given an image, predict its class.

```python
# Single-label: one class per image (softmax)
# Multi-label: multiple classes per image (sigmoid)

# With pretrained model
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

Detect AND locate objects in an image with bounding boxes.

**Key Concepts:**

```
Bounding Box: [x_min, y_min, x_max, y_max] or [x_center, y_center, width, height]

IoU (Intersection over Union):
                 Intersection area
   IoU = ---------------------------------
           Union area of both boxes

   IoU = 1.0 -> Identical boxes
   IoU = 0.0 -> No overlap
   IoU > 0.5 -> Generally considered "correct"

NMS (Non-Maximum Suppression):
   When the model predicts multiple boxes for the same object,
   NMS removes duplicates keeping the one with highest confidence.

Anchor Boxes:
   Predefined boxes of different sizes and aspect ratios
   that serve as starting points for predictions.
```

**YOLO (You Only Look Once) - The Most Practical:**

```python
# YOLOv8 with ultralytics - the fastest way to do detection
from ultralytics import YOLO

# Load pretrained model
model = YOLO("yolov8n.pt")  # nano (fast), s, m, l, x (accurate)

# Train with your data
model.train(
    data="my_dataset.yaml",  # YOLO format
    epochs=100,
    imgsz=640,
    batch=16,
    patience=20,
)

# Inference
results = model("image.jpg")
for result in results:
    boxes = result.boxes  # Bounding boxes
    for box in boxes:
        xyxy = box.xyxy[0]      # Coordinates [x1, y1, x2, y2]
        conf = box.conf[0]      # Confidence
        cls = box.cls[0]        # Class

# Export for production
model.export(format="onnx")
```

**YOLO vs Faster R-CNN:**

| Aspect | YOLO (v8) | Faster R-CNN |
|---|---|---|
| Approach | One-stage (single pass) | Two-stage (RPN + classification) |
| Speed | Very fast (~30+ FPS) | Slower (~5-15 FPS) |
| Accuracy | Very good | Slightly better on small objects |
| Ease of use | Very easy (ultralytics) | More complex (detectron2/torchvision) |
| Best for | Real-time, edge, quick projects | Maximum accuracy, dense objects |

**Detection Metrics:**

| Metric | Description |
|---|---|
| **mAP** | Mean Average Precision: average AP across all classes |
| **AP50** | AP with IoU threshold = 0.5 (most commonly used) |
| **AP75** | AP with IoU threshold = 0.75 (more strict) |
| **mAP@[.5:.95]** | Average AP across different IoU thresholds (COCO metric) |

### Segmentation

Classify each pixel of the image.

```
Classification:    "There is a cat"
Detection:         "There is a cat HERE" (bounding box)
Segmentation:      "THESE pixels are the cat" (mask)
```

**Segmentation Types:**

| Type | Description | Typical Model | Use |
|---|---|---|---|
| **Semantic** | Each pixel has a class, doesn't distinguish instances | U-Net, DeepLab | Medical, satellite |
| **Instance** | Distinguishes between individual instances | Mask R-CNN | Counting, robotics |
| **Panoptic** | Combines semantic + instance | Mask2Former | Most complete |

**U-Net (Semantic Segmentation):**

```
U-Net Architecture (encoder-decoder with skip connections):

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

The skip connections pass high-resolution information
from the encoder to the decoder, preserving fine details.
```

### OCR and Document AI

| Tool | Type | Speed | Accuracy | Languages | Best for |
|---|---|---|---|---|---|
| **Tesseract** | Classic open source | Fast | Medium | 100+ | Basic OCR, clean documents |
| **EasyOCR** | Deep learning, simple | Medium | Good | 80+ | Scene text, multiple languages |
| **PaddleOCR** | Deep learning, complete | Fast | Very good | 80+ | Production, complex documents |
| **AWS Textract** | Cloud API | N/A | Excellent | Limited | Forms, tables, AWS integrated |
| **Google Vision** | Cloud API | N/A | Excellent | 100+ | Maximum accuracy, general OCR |
| **Azure Doc Intelligence** | Cloud API | N/A | Excellent | Many | Enterprise documents |

```python
# EasyOCR - quick to implement
import easyocr

reader = easyocr.Reader(['es', 'en'])  # Spanish and English
results = reader.readtext('document.jpg')

for (bbox, text, confidence) in results:
    print(f"Text: {text}, Confidence: {confidence:.2f}")

# PaddleOCR - more robust for production
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang='es')
result = ocr.ocr('document.jpg', cls=True)
```

**Document Layout Analysis:**

For complex documents (invoices, forms), OCR alone is not enough. You need to understand the structure:

1. **Detect regions:** titles, paragraphs, tables, figures
2. **Extract text** from each region
3. **Understand relationships** between regions (which field goes with which value)

Tools: LayoutLM (Microsoft), Donut (no OCR, end-to-end), DocTR.

---

## Popular Datasets for Practice

| Dataset | Size | Task | Level | Notes |
|---|---|---|---|---|
| **MNIST** | 70K images 28x28 | Digit classification | Beginner | The "Hello World" of CV |
| **CIFAR-10** | 60K images 32x32 | 10-class classification | Beginner | Small images, useful for experimenting |
| **ImageNet** | 1.2M images | 1000-class classification | Reference | The standard benchmark |
| **COCO** | 330K images | Detection + Segmentation | Intermediate | Most used for detection |
| **Pascal VOC** | 11K images | Detection + Segmentation | Intermediate | Classic, smaller than COCO |
| **Open Images** | 9M images | Detection + Segmentation | Advanced | Huge, many classes |

```python
# Load datasets with torchvision
from torchvision import datasets

# CIFAR-10
train = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)

# For custom datasets: ImageFolder
# Structure: root/class_name/image.jpg
train = datasets.ImageFolder(root="./data/train", transform=transform)
```

**Roboflow** for custom datasets: a platform that lets you search public datasets, do labeling, apply augmentations, and export in any format (YOLO, COCO, VOC, etc.).

---

## Data Labeling

### Labeling Tools

| Tool | Type | Tasks | Cost | Best for |
|---|---|---|---|---|
| **Label Studio** | Open source / Cloud | All | Free / Paid | Versatile, all data types |
| **CVAT** | Open source | CV (boxes, segmentation) | Free | Image/video labeling |
| **Roboflow** | Cloud | CV | Freemium | Complete CV workflow |
| **V7** | Cloud | CV + Video | Paid | Large teams, auto-labeling |

### Tips for Efficient Labeling

1. **Define a clear annotation guide** before starting (what each class is, ambiguous edges)
2. **Start with few classes** and expand gradually
3. **Measure inter-annotator agreement** if multiple people are annotating
4. **Pre-annotate with a model** and then correct (semi-automatic)
5. **Fast iterations:** annotate few images -> train -> evaluate -> annotate more

### Active Learning

Concept: instead of annotating images randomly, let the model tell you which images are most "useful" to annotate (those with the most uncertainty). Reduces the labeling needed by 30-70%.

```
Active Learning Cycle:
1. Train model with existing annotated data
2. Predict on unannotated data
3. Select images with highest uncertainty
4. Annotate those images (the most informative ones)
5. Retrain the model
6. Repeat
```

---

## Practical Tips for CV Projects in Consulting

### Always Start with Transfer Learning

Never train from scratch unless you have a very good reason. A pretrained ResNet50 with 100 images will give you better results than a custom CNN with 10,000.

### How Much Data You Need (Rules of Thumb)

| Task | Minimum Viable Data | Recommended Data |
|---|---|---|
| Binary classification (transfer learning) | 50-100 per class | 500+ per class |
| Multi-class classification | 100+ per class | 1000+ per class |
| Object Detection | 200+ bounding boxes per class | 1000+ per class |
| Semantic segmentation | 100+ masks | 500+ masks |

### Imbalanced Datasets in CV

```python
# Option 1: Weighted sampler (most common in CV)
from torch.utils.data import WeightedRandomSampler

class_counts = [1000, 200, 50]  # Images per class
weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
sample_weights = weights[targets]  # targets = list of labels

sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
loader = DataLoader(dataset, batch_size=32, sampler=sampler)

# Option 2: Class weights in the loss
weights = torch.tensor([1.0, 5.0, 20.0])  # Inverse of frequency
criterion = nn.CrossEntropyLoss(weight=weights)

# Option 3: Aggressive data augmentation on minority classes
```

### Edge Deployment

For deploying models to devices (mobile, IoT, cameras):

| Framework | Description | Best for |
|---|---|---|
| **ONNX** | Standard interchange format | Interoperability between frameworks |
| **TensorRT** | NVIDIA optimizer | NVIDIA GPUs (maximum speed) |
| **OpenVINO** | Intel optimizer | Intel CPUs, edge devices |
| **Core ML** | Apple framework | iOS/macOS |
| **TFLite** | TensorFlow Lite | Android, microcontrollers |

```python
# Export to ONNX from PyTorch
import torch.onnx

dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "model.onnx",
                  input_names=["input"],
                  output_names=["output"],
                  dynamic_axes={"input": {0: "batch_size"},
                               "output": {0: "batch_size"}})
```

---

## CV Project Checklist

```
[ ] Define the problem with the client (classification? detection? segmentation?)
[ ] Collect and explore data (quality, quantity, balance)
[ ] Define success metric with the client
[ ] Choose base model (start with pretrained)
[ ] Configure data augmentation
[ ] Train with transfer learning
[ ] Evaluate on validation set
[ ] Iterate: more data? better augmentation? larger model?
[ ] Interpret results (confusion matrix, common errors)
[ ] Optimize for deployment if necessary (ONNX, quantization)
[ ] Monitor in production (visual data drift)
```
