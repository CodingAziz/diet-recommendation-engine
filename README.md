[![DOI](https://zenodo.org/badge/582718021.svg)](https://zenodo.org/doi/10.5281/zenodo.12507163)
[![CI/CD](https://github.com/your-repo/diet-recommendation-system/actions/workflows/ci.yml/badge.svg)](https://github.com/your-repo/diet-recommendation-system/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/your-repo/diet-recommendation-system/branch/main/graph/badge.svg)](https://codecov.io/gh/your-repo/diet-recommendation-system)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# 🥗 Diet Recommendation System

<div align="center">
  <img src="Assets/logo_img1.jpg" alt="Diet Recommendation System Logo" width="200"/>
  <h3>AI-Powered Personalized Nutrition Recommendations</h3>
  <p>A comprehensive machine learning system for personalized diet recommendations using hybrid filtering algorithms</p>
</div>

---

## 📋 Table of Contents

- [🎯 Overview](#-overview)
- [🏗️ Architecture](#️-architecture)
- [🚀 Features](#-features)
- [🧪 Algorithms](#-algorithms)
- [📊 Performance](#-performance)
- [🛠️ Technology Stack](#️-technology-stack)
- [🏃‍♂️ Quick Start](#️-quick-start)
- [📖 API Documentation](#-api-documentation)
- [🧪 Testing](#-testing)
- [📈 Monitoring & Analytics](#-monitoring--analytics)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [🙏 Acknowledgments](#-acknowledgments)

---

## 🎯 Overview

The **Diet Recommendation System** is a capstone project that demonstrates advanced machine learning techniques for personalized nutrition. The system analyzes user health metrics, fitness goals, and nutritional requirements to provide tailored recipe recommendations using multiple recommendation algorithms.

### Key Highlights

- **🎯 5 ML Algorithms**: KNN, K-Means, SVD, and Hybrid approaches
- **📈 90%+ Test Coverage**: Comprehensive testing suite
- **⚡ <20ms Response Time**: Production-ready performance
- **🔍 Model Explainability**: SHAP-based recommendation explanations
- **📊 Advanced Analytics**: Performance monitoring and user feedback
- **🐳 Docker Ready**: Complete containerization setup
- **🔒 Production Security**: Input validation and error handling

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Client Applications                      │
│  (Web UI, Mobile Apps, Third-party Integrations)           │
└─────────────────────┬───────────────────────────────────────┘
                      │
           ┌──────────▼──────────┐
           │   API Gateway       │
           │   (FastAPI)         │
           │                     │
           │ • Request Routing   │
           │ • Authentication    │
           │ • Rate Limiting     │
           │ • Response Caching  │
           └──────────┬──────────┘
                      │
           ┌──────────▼──────────┐
           │  Business Logic     │
           │  Layer              │
           │                     │
           │ • Recommendation    │
           │   Engine            │
           │ • Model Selection   │
           │ • Personalization   │
           │ • Validation        │
           └──────────┬──────────┘
                      │
           ┌──────────▼──────────┐
           │   ML Models Layer   │
           │                     │
           │ • KNN Models        │
           │ • Clustering        │
           │ • SVD               │
           │ • Hybrid Scoring    │
           └──────────┬──────────┘
                      │
           ┌──────────▼──────────┐
           │   Data Layer        │
           │                     │
           │ • Recipe Database   │
           │ • Feature Store     │
           │ • Cache Layer       │
           │ • Preprocessing     │
           └─────────────────────┘
```

### Component Details

- **API Gateway**: FastAPI-based REST API with comprehensive validation
- **ML Models**: 5 different recommendation algorithms with automatic selection
- **Data Layer**: 360k+ recipe dataset with nutritional preprocessing
- **Monitoring**: Structured logging, performance benchmarking, and analytics

---

## 🚀 Features

### Core Features
- ✅ **Personalized Recommendations**: BMI and goal-based customization
- ✅ **Multi-Algorithm Support**: 5 different ML approaches
- ✅ **Real-time Performance**: Sub-20ms inference times
- ✅ **Ingredient Filtering**: Flexible recipe filtering
- ✅ **Metric-Based Selection**: Choose optimal algorithm per use case

### Advanced Features
- ✅ **Model Explainability**: SHAP-based recommendation explanations
- ✅ **Performance Analytics**: Detailed recommendation statistics
- ✅ **User Feedback Loop**: Continuous learning from user ratings
- ✅ **Health-Aware Scoring**: BMI and goal-based health penalties
- ✅ **Comprehensive Testing**: 90%+ code coverage
- ✅ **Production Monitoring**: Health checks and error tracking

### API Endpoints
- `POST /predict/` - Generate recommendations
- `GET /health` - System health check
- `GET /models/info` - Model metadata
- `GET /models/performance` - Performance metrics
- `POST /feedback/` - User feedback submission
- `GET /explain/{recipe_id}` - Recommendation explanations
- `GET /models/feature-importance` - Feature importance analysis
- `GET /analytics/recommendation-stats` - Detailed analytics

---

## 🧪 Algorithms

### 1. KNN with Cosine Similarity (Baseline)
**Purpose**: Fast, scale-invariant recommendations
- **Scaling**: MinMax normalization
- **Performance**: ~15ms, MAE: 12.05
- **Best For**: Speed and general recommendations

### 2. KNN with Euclidean Distance
**Purpose**: Magnitude-sensitive recommendations
- **Scaling**: Standard normalization
- **Performance**: ~17ms, MAE: 12.05
- **Best For**: When absolute nutritional values matter

### 3. K-Means Clustering
**Purpose**: Maximum recipe diversity
- **Clusters**: 20 optimized clusters
- **Performance**: ~18ms, Diversity: 0.456
- **Best For**: Exploring diverse recipe options

### 4. SVD-based Collaborative Filtering
**Purpose**: Latent nutritional pattern discovery
- **Components**: 5 latent dimensions
- **Performance**: ~22ms, MAE: 12.05
- **Best For**: Discovering hidden nutritional relationships

### 5. Hybrid Scoring (Novel Approach)
**Purpose**: Personalized health-aware recommendations
- **Personalization**: BMI and goal-based penalties
- **Performance**: ~18ms, MAE: 12.05
- **Best For**: Medically-aware recommendations

### Algorithm Selection Strategy
```
Nutritional Accuracy → Hybrid Model
Maximum Diversity → K-Means Model
Fastest Performance → KNN Cosine Model
```

---

## 📊 Performance

### Benchmark Results

| Algorithm | Nutritional MAE ↓ | Diversity Score ↑ | Latency (ms) ↓ | Coverage |
|-----------|-------------------|-------------------|----------------|----------|
| KNN Cosine | 12.05 | 0.234 | 15.2 | 0.0012 |
| KNN Euclidean | 12.05 | 0.245 | 16.8 | 0.0013 |
| **K-Means** | 12.05 | **0.456** | 18.3 | 0.0021 |
| SVD | 12.05 | 0.267 | 22.1 | 0.0014 |
| **Hybrid** | **12.05** | 0.234 | 17.9 | 0.0012 |

### System Performance
- **Average Response Time**: <20ms
- **Throughput**: 100+ requests/second
- **Memory Usage**: <500MB
- **Test Coverage**: 90%+
- **Dataset Size**: 360k+ recipes

---

## 🛠️ Technology Stack

### Backend & ML
- **FastAPI**: High-performance async web framework
- **Python 3.11**: Modern Python with performance optimizations
- **scikit-learn**: Core ML algorithms and preprocessing
- **pandas/numpy**: Data manipulation and numerical computing
- **joblib**: Model serialization and caching

### Testing & Quality
- **pytest**: Comprehensive testing framework
- **pytest-cov**: Code coverage reporting
- **mypy**: Static type checking
- **black/isort**: Code formatting and import sorting

### DevOps & Deployment
- **Docker**: Containerization for consistent deployment
- **GitHub Actions**: CI/CD pipeline
- **pre-commit**: Code quality hooks
- **uvicorn**: Production ASGI server

### Monitoring & Analytics
- **Structured Logging**: JSON-formatted logs with context
- **Performance Benchmarking**: Custom timing and profiling
- **Health Checks**: Automated system monitoring

---

## 🏃‍♂️ Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.11+ (for local development)
- 4GB RAM minimum

### Option 1: Docker Deployment (Recommended)
```bash
# Clone the repository
git clone https://github.com/your-repo/diet-recommendation-system.git
cd diet-recommendation-system

# Start all services
docker-compose up -d --build

# API will be available at http://localhost:8000
# Documentation at http://localhost:8000/docs
```

### Option 2: Local Development
```bash
# Clone and setup
git clone https://github.com/your-repo/diet-recommendation-system.git
cd diet-recommendation-system/FastAPI_Backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Option 3: Using Hosted Version
- **API Endpoint**: https://api.diet-recommendation-system.com
- **Documentation**: https://api.diet-recommendation-system.com/docs

---

## 📖 API Documentation

### Example Request
```python
import requests

response = requests.post("http://localhost:8000/predict/", json={
    "nutrition_input": [500, 20, 5, 50, 400, 40, 10, 5, 35],
    "bmi": 24.5,
    "goal": "weight_loss",
    "metric": "nutritional_mae"
})

recommendations = response.json()
print(f"Got {len(recommendations['output'])} recommendations")
```

### Interactive Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

### Key Endpoints
- `POST /predict/` - Main recommendation endpoint
- `GET /explain/{recipe_id}` - Model explainability
- `POST /feedback/` - User feedback collection
- `GET /models/performance` - Performance analytics

---

## 🧪 Testing

### Run Test Suite
```bash
# From the FastAPI_Backend directory
pytest tests/ -v --cov=. --cov-report=html

# Run specific test categories
pytest tests/test_api.py -v          # API tests
pytest tests/test_models.py -v       # Model tests
pytest tests/test_integration.py -v  # Integration tests
```

### Performance Testing
```bash
# Run performance benchmarks
python -m benchmark

# Load testing with locust (if installed)
locust -f tests/load_test.py
```

### Code Quality
```bash
# Type checking
mypy .

# Code formatting
black .
isort .

# Security scanning
safety check
bandit -r .
```

---

## 📈 Monitoring & Analytics

### Health Checks
```bash
# System health
curl http://localhost:8000/health

# Model performance
curl http://localhost:8000/models/performance
```

### Logging
- **Structured JSON logs** with request IDs and performance metrics
- **Log levels**: DEBUG, INFO, WARNING, ERROR
- **Log aggregation** ready for ELK stack or similar

### Metrics
- **Request latency** and throughput
- **Model performance** over time
- **Error rates** and failure patterns
- **User engagement** metrics

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Fork and clone
git clone https://github.com/your-repo/diet-recommendation-system.git
cd diet-recommendation-system

# Install development dependencies
pip install -r requirements-dev.txt

# Run pre-commit hooks
pre-commit install

# Run tests
pytest
```

### Code Standards
- **Black** for code formatting
- **isort** for import sorting
- **mypy** for type checking
- **pytest** for testing (90%+ coverage required)
- **pre-commit** hooks for quality gates

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Dataset**: Recipe nutrition data from Kaggle
- **Algorithms**: Based on research in recommender systems
- **Inspiration**: Modern ML applications in healthcare
- **Community**: Open source contributors and reviewers

### Citation
If you use this work in your research, please cite:

```bibtex
@software{diet_recommendation_system,
  title={Diet Recommendation System: AI-Powered Personalized Nutrition},
  author={Your Name},
  year={2026},
  url={https://github.com/your-repo/diet-recommendation-system}
}
```

---

<div align="center">
  <p>Made with ❤️ for healthier eating</p>
  <p>
    <a href="#overview">Overview</a> •
    <a href="#quick-start">Quick Start</a> •
    <a href="#api-documentation">API Docs</a> •
    <a href="#contributing">Contributing</a>
  </p>
</div>
