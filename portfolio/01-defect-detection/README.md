# Automated Defect Detection in Manufacturing

## Business Problem

Manual visual inspection on production lines presents critical challenges:

- **Slowness**: a human inspector takes between 5-15 seconds per part, limiting line throughput.
- **Inconsistency**: accuracy varies with fatigue, lighting, and inspector experience. Studies show that the manual detection rate drops 20-30% after 2 continuous hours.
- **High cost**: maintaining inspection teams 24/7 across three shifts represents a significant salary cost.
- **Escaped defects**: defects reaching the customer generate returns, warranty claims, and reputational damage.

## Proposed Solution

Computer vision system that detects defects in products in real time, integrated directly into the production line.

### Technical Architecture

```
Industrial camera --> Image preprocessing --> EfficientNet-B0 (Transfer Learning)
                                                        |
                                                  Classification Head
                                                        |
                                              Defect / No defect
                                              (+ defect type)
                                                        |
                                                  FastAPI Server
                                                        |
                                              Dashboard / Alerts
```

- **Base model**: EfficientNet-B0 pretrained on ImageNet, with the last fully-connected layer replaced for binary classification (defect/no-defect) or multi-class (defect type).
- **Transfer Learning**: the first layers of the backbone are frozen and only the last layers + classification head are trained, allowing training with few data (~500-1000 images).
- **Data Augmentation**: rotations, flips, brightness/contrast changes, random crops to make the model more robust.
- **Inference service**: REST API with FastAPI, containerized with Docker for easy deployment.

### Dataset

- **MVTec Anomaly Detection Dataset** (MVTec AD): reference dataset for industrial defect detection with 15 product categories and multiple defect types.
- Alternative: synthetic dataset included in the repository for quick demos.

## Expected Results

| Metric | Value |
|---------|-------|
| Accuracy | >95% |
| Precision | >93% |
| Recall | >94% |
| Inference time | <100ms per image |
| Throughput | >10 images/second |

## Technologies

- **PyTorch** + **torchvision**: deep learning framework and pretrained models
- **Albumentations**: advanced data augmentation
- **FastAPI**: high-performance REST API
- **ONNX Runtime**: optimized inference in production
- **Docker**: containerization for deployment
- **OpenCV**: image preprocessing

## How to Run

### 1. Installation

```bash
# Clone the repository
git clone <repo-url>
cd portfolio/01-defect-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python src/train.py \
    --data_dir data/defects \
    --epochs 30 \
    --batch_size 32 \
    --lr 0.001 \
    --output_dir models/
```

Expected data directory structure:
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

### 3. Launch the API

```bash
python src/api.py
# or:
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

### 4. Test Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
    -F "file=@test_image.jpg"
```

### 5. Docker

```bash
docker build -t defect-detection .
docker run -p 8000:8000 defect-detection
```

## How to Present It: Client Pitch

### Value Proposition

> "Imagine replacing human variability with a system that inspects every part in less than 100 milliseconds, 24 hours a day, 7 days a week, without fatigue and with accuracy above 95%."

### Estimated ROI

**Scenario**: plant with 3 shifts, 4 inspectors per shift (12 inspectors total).

| Item | Before | After |
|----------|-------|---------|
| Annual inspection cost | ~360,000 EUR (12 inspectors) | ~80,000 EUR (2 inspectors + system) |
| Escaped defects | 2-5% | <0.5% |
| Inspection speed | 5-15 sec/part | <0.1 sec/part |
| Availability | Subject to shifts/absences | 24/7 continuous |

**Estimated net savings: ~280,000 EUR/year**, not counting the reduction in warranty and return costs.

### Key Points for the Presentation

1. **Live demo**: show the API processing images in real time.
2. **Clear metrics**: precision, recall, and inference speed.
3. **Scalability**: one model, multiple cameras/lines.
4. **Integration**: connects to existing SCADA/MES systems via REST API.
5. **Continuous improvement**: the model can be retrained with new defect types without redesigning the system.

### Frequently Asked Client Questions

- **"What if a new type of defect appears?"** - Images of the new defect are collected, the model is retrained (quick fine-tuning), and deployed without stopping the line.
- **"What happens if the system fails?"** - Fallback to manual inspection. The system has health checks and automatic alerts.
- **"How long does implementation take?"** - Functional pilot in 4-6 weeks. Full integration in 2-3 months.
