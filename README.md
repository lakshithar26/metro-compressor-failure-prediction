# Metro Train Air Compressor - Failure Prediction & Anomaly Detection

Course: CS5998 | Student ID: 268500F

## Overview

End-to-end system for predicting failures and detecting anomalies in metro train air compressors using the MetroPT-3 dataset from UCI Machine Learning Repository.

**Key Results:**
- 4/4 failure events detected
- 94.2% detection rate within failure periods
- 2.18% false alarm rate

## Project Structure

```
metro-compressor-failure-prediction/
├── notebooks/                  # Exploration & modeling
│   ├── 01_eda.ipynb           # Exploratory data analysis
│   ├── 02_preprocessing.ipynb # Data preprocessing & feature engineering
│   ├── 03_baseline_models.ipynb # Statistical & ML baselines
│   └── 04_advanced_models.ipynb # Anomaly detection models
│
├── src/                        # Production code
│   ├── features.py            # Feature engineering utilities
│   ├── streaming/
│   │   └── simulator.py       # Data replay simulator
│   └── api/
│       └── scoring_service.py # FastAPI prediction service
│
├── app/
│   └── dashboard.py           # Streamlit monitoring dashboard
│
├── models/                     # Trained model artifacts
├── dataset/                    # Data files (not in repo)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Dataset

MetroPT-3 Dataset from UCI ML Repository:
- 1,516,948 records at 1 Hz sampling rate
- 15 sensor signals (7 analogue, 8 digital)
- 4 documented air leak failures
- Period: February - August 2020

## Installation

```bash
# Clone repository
git clone <repo-url>
cd metro-compressor-failure-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download dataset
# Place MetroPT3(AirCompressor).csv in dataset/metropt+3+dataset/
```

## Usage

### Run Notebooks
```bash
source venv/bin/activate
jupyter lab
```

### Run API Server
```bash
source venv/bin/activate
python -m src.api.scoring_service
# API available at http://localhost:8000
```

### Run Data Simulator
```bash
source venv/bin/activate
python -m src.streaming.simulator --failure 4 --speed 100
```

### Run Dashboard
```bash
source venv/bin/activate
streamlit run app/dashboard.py
# Dashboard at http://localhost:8501
```

### Docker Deployment
```bash
docker-compose up scoring-api
```

## Models

### Approach
Due to distribution shift between failure events (each failure has unique characteristics), supervised models failed to generalize. Solution: unsupervised anomaly detection with leave-one-failure-out cross-validation.

### Results

| Model | Failures Detected | Detection Rate | False Alarm Rate |
|-------|-------------------|----------------|------------------|
| Simple (Oil_temp threshold) | 4/4 | 94.2% | 2.18% |
| Isolation Forest | 4/4 | 86.3% | 4.99% |
| Autoencoder | 3/4 | 29.6% | 5.00% |

### Key Finding
Oil_temperature is the most robust indicator - elevated in all 4 failure events. A simple threshold detector (mean + 2*std) achieves best overall performance.

## API Endpoints

- `GET /health` - Health check
- `POST /predict` - Submit sensor reading, get anomaly prediction
- `GET /model/info` - Model information
- `POST /reset` - Reset feature buffer

## Limitations

- Single compressor from one metro system
- Only 4 failure events (all air leaks)
- Model may not generalize to other failure types
- Requires 60-second warmup for rolling features

## References

- Dataset: https://archive.ics.uci.edu/dataset/791/metropt+3+dataset
- DOI: https://doi.org/10.24432/C5VW3R
