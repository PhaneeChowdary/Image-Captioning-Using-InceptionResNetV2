# Image Captioning using InceptionResNetV2

Deep learning model that generates descriptions for images using InceptionResNetV2 encoder and attention-based GRU decoder.

## Setup

### Windows
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### macOS/Linux
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running the Model

1. Ensure required model files are in `Epochs10/` directory:
   - encoder.weights.h5
   - decoder.weights.h5
   - model_config.json
   - tokenizer_vocab.json
   - You can access these files here [LINK](https://drive.google.com/drive/folders/13jtMWh_qFOrVcUll43nYE7qMvxGheLd0?usp=sharing)

2. Run inference:
```bash
python Inference.py
```

3. Enter image path when prompted or 'q' to quit.

## Thank you
