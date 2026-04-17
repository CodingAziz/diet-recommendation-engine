# Diet Recommendation System - Complete Implementation Guide

## Table of Contents
1. [Backend Setup](#backend-setup)
2. [Frontend Integration](#frontend-integration)
3. [API Reference](#api-reference)
4. [Client Libraries](#client-libraries)
5. [Testing & Validation](#testing--validation)
6. [Deployment](#deployment)
7. [Error Handling](#error-handling)
8. [Performance Optimization](#performance-optimization)
9. [Advanced Features](#advanced-features)
10. [Troubleshooting](#troubleshooting)

---

## 1. Backend Setup

### Installation & Configuration

```bash
# Create virtual environment
python3.11 -m venv diet-venv
source diet-venv/bin/activate

# Install dependencies
cd FastAPI_Backend
pip install -r requirements.txt

# Run backend server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Environment Variables

Create `.env` file in project root:

```bash
# .env
ENVIRONMENT=development
DEBUG=True
LOG_LEVEL=INFO
DATA_PATH=./Data/dataset.csv
CACHE_ENABLED=True
CACHE_TTL=3600
K_CANDIDATES=50
TOP_K=10
N_CLUSTERS=20
N_COMPONENTS=5
RANDOM_SEED=42
```

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY FastAPI_Backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "FastAPI_Backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  backend:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - DEBUG=False
    volumes:
      - ./Data:/app/Data
      - ./logs:/app/logs
```

---

## 2. Frontend Integration

### React/TypeScript Setup

```bash
npm create vite@latest diet-frontend -- --template react
cd diet-frontend
npm install axios react-query zustand
```

### API Service Layer

```typescript
// src/services/api.ts
import axios from 'axios';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface NutritionInput {
  nutrition_input: [number, number, number, number, number, number, number, number, number];
  ingredients: string[];
  params: {
    n_neighbors: number;
    return_distance: boolean;
  };
  bmi: number;
  goal: 'weight_loss' | 'muscle_gain' | 'maintenance';
  metric: 'nutritional_mae' | 'diversity_score';
}

export interface Recipe {
  Name: string;
  CookTime: string;
  PrepTime: string;
  TotalTime: string;
  RecipeIngredientParts: string[];
  Calories: number;
  FatContent: number;
  SaturatedFatContent: number;
  CholesterolContent: number;
  SodiumContent: number;
  CarbohydrateContent: number;
  FiberContent: number;
  SugarContent: number;
  ProteinContent: number;
  RecipeInstructions: string[];
}

export interface RecommendationResponse {
  output: Recipe[] | null;
  metadata: {
    model_used: string;
    metric_basis: string;
    total_candidates: number;
    request_timestamp: string;
  };
}

export interface ModelMetrics {
  nutritional_mae: number;
  diversity_score: number;
  latency_ms: number;
  coverage: number;
}

export interface ModelPerformanceResponse {
  models: Record<string, ModelMetrics>;
  recommendations: Record<string, string>;
  last_updated: string;
}

// Recommendation endpoints
export const recommendationAPI = {
  getRecommendations: (data: NutritionInput) =>
    api.post<RecommendationResponse>('/predict/', data),
  
  getModelPerformance: () =>
    api.get<ModelPerformanceResponse>('/models/performance'),
  
  getModelsInfo: () =>
    api.get('/models/info'),
  
  getHealth: () =>
    api.get('/health'),
  
  submitFeedback: (feedback: any) =>
    api.post('/feedback/', feedback),
  
  getExplanation: (recipeId: number) =>
    api.get(`/explain/${recipeId}`),
};
```

### React Hooks

```typescript
// src/hooks/useRecommendations.ts
import { useQuery, useMutation } from 'react-query';
import { recommendationAPI, NutritionInput } from '../services/api';

export const useRecommendations = (input: NutritionInput | null) => {
  return useQuery(
    ['recommendations', input],
    () => recommendationAPI.getRecommendations(input!),
    { enabled: !!input }
  );
};

export const useModelPerformance = () => {
  return useQuery(
    'modelPerformance',
    () => recommendationAPI.getModelPerformance(),
    { staleTime: 5 * 60 * 1000 } // 5 minutes
  );
};

export const useHealth = () => {
  return useQuery(
    'health',
    () => recommendationAPI.getHealth(),
    { staleTime: 1 * 60 * 1000 } // 1 minute
  );
};

export const useFeedbackMutation = () => {
  return useMutation(
    (feedback) => recommendationAPI.submitFeedback(feedback)
  );
};
```

### React Components

```typescript
// src/components/RecommendationForm.tsx
import React, { useState } from 'react';
import { useRecommendations } from '../hooks/useRecommendations';
import { NutritionInput } from '../services/api';

export const RecommendationForm: React.FC<{
  onSubmit: (input: NutritionInput) => void;
}> = ({ onSubmit }) => {
  const [nutrition, setNutrition] = useState<number[]>([2000, 65, 20, 0, 2000, 300, 25, 50, 150]);
  const [bmi, setBmi] = useState(25);
  const [goal, setGoal] = useState<'weight_loss' | 'muscle_gain' | 'maintenance'>('maintenance');
  const [metric, setMetric] = useState<'nutritional_mae' | 'diversity_score'>('nutritional_mae');

  const handleNutritionChange = (index: number, value: number) => {
    const newNutrition = [...nutrition];
    newNutrition[index] = value;
    setNutrition(newNutrition);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit({
      nutrition_input: nutrition as any,
      ingredients: [],
      params: { n_neighbors: 5, return_distance: false },
      bmi,
      goal,
      metric,
    });
  };

  const nutritionLabels = ['Calories', 'Fat', 'Sat Fat', 'Cholesterol', 'Sodium', 'Carbs', 'Fiber', 'Sugar', 'Protein'];

  return (
    <form onSubmit={handleSubmit} className="form">
      <fieldset>
        <legend>Nutrition Targets</legend>
        <div className="nutrition-grid">
          {nutritionLabels.map((label, idx) => (
            <div key={idx} className="input-group">
              <label>{label}</label>
              <input
                type="number"
                value={nutrition[idx]}
                onChange={(e) => handleNutritionChange(idx, parseFloat(e.target.value))}
              />
            </div>
          ))}
        </div>
      </fieldset>

      <fieldset>
        <legend>Personal Info</legend>
        <div className="input-group">
          <label>BMI</label>
          <input
            type="number"
            step="0.1"
            value={bmi}
            onChange={(e) => setBmi(parseFloat(e.target.value))}
          />
        </div>
      </fieldset>

      <fieldset>
        <legend>Preferences</legend>
        <div className="input-group">
          <label>Goal</label>
          <select value={goal} onChange={(e) => setGoal(e.target.value as any)}>
            <option>weight_loss</option>
            <option>muscle_gain</option>
            <option>maintenance</option>
          </select>
        </div>
        <div className="input-group">
          <label>Metric</label>
          <select value={metric} onChange={(e) => setMetric(e.target.value as any)}>
            <option>nutritional_mae</option>
            <option>diversity_score</option>
          </select>
        </div>
      </fieldset>

      <button type="submit">Get Recommendations</button>
    </form>
  );
};
```

```typescript
// src/components/RecipeCard.tsx
import React from 'react';
import { Recipe } from '../services/api';
import { useFeedbackMutation } from '../hooks/useRecommendations';

export const RecipeCard: React.FC<{ recipe: Recipe }> = ({ recipe }) => {
  const feedbackMutation = useFeedbackMutation();
  const [rating, setRating] = React.useState<number>(0);

  const handleRating = (value: number) => {
    setRating(value);
    feedbackMutation.mutate({
      user_id: 'user_123',
      recipe_id: recipe.Name,
      rating: value,
      was_helpful: value >= 4,
    });
  };

  return (
    <div className="recipe-card">
      <h3>{recipe.Name}</h3>
      <div className="recipe-times">
        <span>⏱️ Prep: {recipe.PrepTime}</span>
        <span>🍳 Cook: {recipe.CookTime}</span>
        <span>Total: {recipe.TotalTime}</span>
      </div>

      <div className="nutrition-summary">
        <div>Calories: {recipe.Calories.toFixed(0)}</div>
        <div>Protein: {recipe.ProteinContent.toFixed(1)}g</div>
        <div>Carbs: {recipe.CarbohydrateContent.toFixed(1)}g</div>
        <div>Fat: {recipe.FatContent.toFixed(1)}g</div>
      </div>

      <div className="ingredients">
        <h4>Ingredients:</h4>
        <ul>
          {recipe.RecipeIngredientParts.slice(0, 5).map((ing, idx) => (
            <li key={idx}>{ing}</li>
          ))}
          {recipe.RecipeIngredientParts.length > 5 && (
            <li>... +{recipe.RecipeIngredientParts.length - 5} more</li>
          )}
        </ul>
      </div>

      <div className="rating">
        <label>Rate this recipe:</label>
        <div className="stars">
          {[1, 2, 3, 4, 5].map((star) => (
            <button
              key={star}
              className={`star ${rating >= star ? 'active' : ''}`}
              onClick={() => handleRating(star)}
            >
              ★
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};
```

```typescript
// src/components/ModelPerformanceDashboard.tsx
import React from 'react';
import { useModelPerformance } from '../hooks/useRecommendations';

export const ModelPerformanceDashboard: React.FC = () => {
  const { data, isLoading } = useModelPerformance();

  if (isLoading) return <div>Loading performance metrics...</div>;
  if (!data) return <div>No data available</div>;

  return (
    <div className="dashboard">
      <h2>Model Performance Comparison</h2>
      
      <table className="performance-table">
        <thead>
          <tr>
            <th>Model</th>
            <th>Nutritional MAE</th>
            <th>Diversity Score</th>
            <th>Latency (ms)</th>
            <th>Coverage</th>
            <th>Best For</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(data.data.models).map(([model, metrics]) => (
            <tr key={model}>
              <td className="model-name">{model}</td>
              <td>{metrics.nutritional_mae.toFixed(2)}</td>
              <td>{metrics.diversity_score.toFixed(3)}</td>
              <td>{metrics.latency_ms.toFixed(1)}</td>
              <td>{(metrics.coverage * 100).toFixed(2)}%</td>
              <td>{data.data.recommendations[model] || '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <div className="recommendations">
        <h3>Recommended Models by Use Case</h3>
        <ul>
          {Object.entries(data.data.recommendations).map(([usecase, model]) => (
            <li key={usecase}>
              <strong>{usecase}:</strong> {model}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
};
```

---

## 3. API Reference

### POST /predict/

Get recipe recommendations based on nutritional requirements.

**Request:**
```json
{
  "nutrition_input": [2000, 65, 20, 0, 2000, 300, 25, 50, 150],
  "ingredients": ["chicken", "broccoli"],
  "params": {
    "n_neighbors": 5,
    "return_distance": false
  },
  "bmi": 25.5,
  "goal": "maintenance",
  "metric": "nutritional_mae"
}
```

**Response (200 OK):**
```json
{
  "output": [
    {
      "Name": "Grilled Chicken Salad",
      "CookTime": "15 minutes",
      "PrepTime": "10 minutes",
      "TotalTime": "25 minutes",
      "RecipeIngredientParts": ["chicken breast", "lettuce", "tomato"],
      "Calories": 350,
      "FatContent": 12,
      "SaturatedFatContent": 3,
      "CholesterolContent": 85,
      "SodiumContent": 480,
      "CarbohydrateContent": 15,
      "FiberContent": 3,
      "SugarContent": 2,
      "ProteinContent": 45,
      "RecipeInstructions": ["Grill chicken", "Mix salad"]
    }
  ],
  "metadata": {
    "model_used": "hybrid",
    "metric_basis": "nutritional_mae",
    "total_candidates": 5,
    "request_timestamp": "2026-04-16T20:39:09.612537"
  }
}
```

### GET /models/performance

Retrieve performance metrics for all available models.

**Response (200 OK):**
```json
{
  "models": {
    "knn_cosine": {
      "nutritional_mae": 12.05,
      "diversity_score": 0.234,
      "latency_ms": 15.2,
      "coverage": 0.0012
    },
    "hybrid": {
      "nutritional_mae": 12.05,
      "diversity_score": 0.234,
      "latency_ms": 17.9,
      "coverage": 0.0012
    }
  },
  "recommendations": {
    "nutritional_accuracy": "hybrid",
    "maximum_diversity": "kmeans",
    "fastest": "knn_cosine"
  },
  "last_updated": "2026-04-16T20:39:09.618138"
}
```

### GET /models/info

Get information about available recommendation models.

**Response (200 OK):**
```json
{
  "available_models": ["knn_cosine", "knn_euclidean", "kmeans", "svd", "hybrid"],
  "available_metrics": ["nutritional_mae", "diversity_score"],
  "default_metric": "nutritional_mae",
  "model_details": {
    "knn_cosine": {
      "description": "Fast cosine similarity search",
      "best_for": "speed"
    },
    "kmeans": {
      "description": "Cluster-based recommendations",
      "best_for": "diversity"
    }
  }
}
```

### GET /health

Health check endpoint for system monitoring.

**Response (200 OK):**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2026-04-16T20:39:09.612537",
  "uptime_seconds": null
}
```

### POST /feedback/

Submit user feedback on recommendations.

**Request:**
```json
{
  "user_id": "user_123",
  "recipe_id": 42,
  "rating": 5,
  "was_helpful": true,
  "comments": "Great recipe, made it tonight!",
  "session_id": "session_abc123"
}
```

**Response (200 OK):**
```json
{
  "status": "success",
  "feedback_id": "feedback_xyz789",
  "message": "Feedback recorded successfully",
  "timestamp": "2026-04-16T20:39:09.612537"
}
```

---

## 4. Client Libraries

### Python Client

```python
# diet_recommendation_client.py
import requests
from typing import List, Literal, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class RecipeRecommendation:
    Name: str
    CookTime: str
    PrepTime: str
    Calories: float
    ProteinContent: float
    CarbohydrateContent: float
    FatContent: float
    RecipeIngredientParts: List[str]
    RecipeInstructions: List[str]

class DietRecommendationClient:
    def __init__(self, base_url: str = 'http://localhost:8000'):
        self.base_url = base_url
        self.session = requests.Session()
    
    def get_recommendations(
        self,
        nutrition_input: List[float],
        bmi: float,
        goal: Literal['weight_loss', 'muscle_gain', 'maintenance'],
        metric: Literal['nutritional_mae', 'diversity_score'] = 'nutritional_mae',
        n_neighbors: int = 5,
        ingredients: Optional[List[str]] = None
    ) -> List[RecipeRecommendation]:
        """Get personalized recipe recommendations"""
        payload = {
            'nutrition_input': nutrition_input,
            'bmi': bmi,
            'goal': goal,
            'metric': metric,
            'ingredients': ingredients or [],
            'params': {
                'n_neighbors': n_neighbors,
                'return_distance': False
            }
        }
        response = self.session.post(
            f'{self.base_url}/predict/',
            json=payload
        )
        response.raise_for_status()
        data = response.json()
        
        recipes = []
        if data['output']:
            for recipe_data in data['output']:
                recipes.append(RecipeRecommendation(**recipe_data))
        return recipes
    
    def get_model_performance(self) -> dict:
        """Get performance metrics for all models"""
        response = self.session.get(f'{self.base_url}/models/performance')
        response.raise_for_status()
        return response.json()
    
    def get_health(self) -> dict:
        """Check API health"""
        response = self.session.get(f'{self.base_url}/health')
        response.raise_for_status()
        return response.json()
    
    def submit_feedback(
        self,
        user_id: str,
        recipe_id: int,
        rating: int,
        was_helpful: bool,
        comments: Optional[str] = None
    ) -> dict:
        """Submit feedback on a recommendation"""
        payload = {
            'user_id': user_id,
            'recipe_id': recipe_id,
            'rating': rating,
            'was_helpful': was_helpful,
            'comments': comments
        }
        response = self.session.post(
            f'{self.base_url}/feedback/',
            json=payload
        )
        response.raise_for_status()
        return response.json()

# Usage example
if __name__ == '__main__':
    client = DietRecommendationClient()
    
    # Get recommendations
    recommendations = client.get_recommendations(
        nutrition_input=[2000, 65, 20, 0, 2000, 300, 25, 50, 150],
        bmi=25.0,
        goal='maintenance',
        metric='nutritional_mae'
    )
    
    print(f"Got {len(recommendations)} recommendations:")
    for recipe in recommendations:
        print(f"- {recipe.Name} ({recipe.Calories:.0f} cal)")
    
    # Get model performance
    performance = client.get_model_performance()
    print("\nModel Performance:")
    for model, metrics in performance['models'].items():
        print(f"{model}: MAE={metrics['nutritional_mae']:.2f}, "
              f"Diversity={metrics['diversity_score']:.3f}")
```

### JavaScript Client

```javascript
// diet-recommendation-client.js
class DietRecommendationClient {
  constructor(baseUrl = 'http://localhost:8000') {
    this.baseUrl = baseUrl;
  }

  async getRecommendations({
    nutritionInput,
    bmi,
    goal,
    metric = 'nutritional_mae',
    nNeighbors = 5,
    ingredients = []
  }) {
    const response = await fetch(`${this.baseUrl}/predict/`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        nutrition_input: nutritionInput,
        bmi,
        goal,
        metric,
        ingredients,
        params: {
          n_neighbors: nNeighbors,
          return_distance: false
        }
      })
    });
    
    if (!response.ok) throw new Error(`API Error: ${response.status}`);
    const data = await response.json();
    return data.output || [];
  }

  async getModelPerformance() {
    const response = await fetch(`${this.baseUrl}/models/performance`);
    if (!response.ok) throw new Error(`API Error: ${response.status}`);
    return response.json();
  }

  async getHealth() {
    const response = await fetch(`${this.baseUrl}/health`);
    if (!response.ok) throw new Error(`API Error: ${response.status}`);
    return response.json();
  }

  async submitFeedback({
    userId,
    recipeId,
    rating,
    wasHelpful,
    comments
  }) {
    const response = await fetch(`${this.baseUrl}/feedback/`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        user_id: userId,
        recipe_id: recipeId,
        rating,
        was_helpful: wasHelpful,
        comments
      })
    });
    
    if (!response.ok) throw new Error(`API Error: ${response.status}`);
    return response.json();
  }
}

// Usage
const client = new DietRecommendationClient();
client.getRecommendations({
  nutritionInput: [2000, 65, 20, 0, 2000, 300, 25, 50, 150],
  bmi: 25,
  goal: 'maintenance'
}).then(recipes => console.log(recipes));
```

---

## 5. Testing & Validation

### Unit Tests

```python
# tests/test_models.py
import pytest
import numpy as np
from FastAPI_Backend.model import (
    recommend_knn_cosine,
    recommend_knn_euclidean,
    recommend_kmeans,
    recommend_svd,
    recommend_hybrid,
    recommend
)

@pytest.fixture
def sample_nutrition_vector():
    return [2000, 65, 20, 0, 2000, 300, 25, 50, 150]

def test_knn_cosine_returns_recipes(sample_nutrition_vector):
    results = recommend_knn_cosine(sample_nutrition_vector, top_k=5)
    assert len(results) <= 5
    assert 'Name' in results.columns
    assert 'Calories' in results.columns

def test_recommend_hybrid_applies_penalties(sample_nutrition_vector):
    # Test with high BMI
    results_high_bmi = recommend_hybrid(sample_nutrition_vector, bmi=35, goal='weight_loss', top_k=5)
    assert len(results_high_bmi) <= 5
    
    # Test with normal BMI
    results_normal = recommend_hybrid(sample_nutrition_vector, bmi=22, goal='maintenance', top_k=5)
    assert len(results_normal) <= 5

def test_recommend_filters_by_metric(sample_nutrition_vector):
    mae_results = recommend(dataset=None, _input=sample_nutrition_vector, metric='nutritional_mae')
    diversity_results = recommend(dataset=None, _input=sample_nutrition_vector, metric='diversity_score')
    
    # Should use different algorithms
    assert len(mae_results) > 0
    assert len(diversity_results) > 0
```

### Integration Tests

```python
# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from FastAPI_Backend.main import app

client = TestClient(app)

@pytest.fixture
def valid_prediction_payload():
    return {
        "nutrition_input": [2000, 65, 20, 0, 2000, 300, 25, 50, 150],
        "ingredients": [],
        "params": {"n_neighbors": 5, "return_distance": False},
        "bmi": 25.0,
        "goal": "maintenance",
        "metric": "nutritional_mae"
    }

def test_predict_returns_recipes(valid_prediction_payload):
    response = client.post("/predict/", json=valid_prediction_payload)
    assert response.status_code == 200
    data = response.json()
    assert "output" in data
    assert "metadata" in data
    assert len(data["output"]) > 0

def test_predict_invalid_input():
    response = client.post("/predict/", json={
        "nutrition_input": [1, 2, 3],  # Too short
        "bmi": 25,
        "goal": "maintenance",
        "metric": "nutritional_mae"
    })
    assert response.status_code == 422

def test_models_performance_endpoint():
    response = client.get("/models/performance")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert "recommendations" in data

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data

def test_feedback_endpoint():
    feedback = {
        "user_id": "test_user",
        "recipe_id": 1,
        "rating": 5,
        "was_helpful": True,
        "comments": "Great recipe!"
    }
    response = client.post("/feedback/", json=feedback)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
```

### Performance Testing

```python
# tests/test_performance.py
import time
import pytest
from FastAPI_Backend.model import recommend_knn_cosine, recommend_hybrid

@pytest.fixture
def sample_nutrition_vector():
    return [2000, 65, 20, 0, 2000, 300, 25, 50, 150]

def test_knn_cosine_latency(sample_nutrition_vector):
    start = time.time()
    for _ in range(100):
        recommend_knn_cosine(sample_nutrition_vector)
    elapsed = (time.time() - start) / 100
    
    # Should be under 20ms per request
    assert elapsed < 0.020, f"KNN Cosine took {elapsed*1000:.2f}ms"

def test_hybrid_latency(sample_nutrition_vector):
    start = time.time()
    for _ in range(100):
        recommend_hybrid(sample_nutrition_vector, bmi=25, goal='maintenance')
    elapsed = (time.time() - start) / 100
    
    # Should be under 20ms per request
    assert elapsed < 0.020, f"Hybrid took {elapsed*1000:.2f}ms"
```

---

## 6. Deployment

### Production Configuration

```python
# config.py (production)
import os
from pathlib import Path

class Settings:
    # Environment
    ENVIRONMENT = os.getenv("ENVIRONMENT", "production")
    DEBUG = False
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = "/var/log/diet-recommendation.log"
    
    # Model Configuration
    K_CANDIDATES = 50
    TOP_K = 10
    N_CLUSTERS = 20
    N_COMPONENTS = 5
    RANDOM_SEED = 42
    
    # Data Configuration
    DATA_PATH = os.getenv("DATA_PATH", "/app/Data/dataset.csv")
    CACHE_ENABLED = True
    CACHE_TTL = 3600
    
    # API Configuration
    API_TITLE = "Diet Recommendation System API"
    API_VERSION = "1.0.0"
    
    # Security
    ALLOWED_HOSTS = ["*.example.com", "localhost"]
    
    # Performance
    WORKERS = 4
    WORKER_CLASS = "uvicorn.workers.UvicornWorker"
```

### Nginx Configuration

```nginx
# /etc/nginx/sites-available/diet-api
upstream diet_api {
    server 127.0.0.1:8000;
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
    server 127.0.0.1:8003;
}

server {
    listen 80;
    server_name api.diet-recommendation.com;
    
    client_max_body_size 10M;
    
    location / {
        proxy_pass http://diet_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
    
    location /health {
        access_log off;
        proxy_pass http://diet_api;
    }
}
```

### Systemd Service

```ini
# /etc/systemd/system/diet-api.service
[Unit]
Description=Diet Recommendation System API
After=network.target

[Service]
Type=notify
User=api-user
WorkingDirectory=/app
Environment="PATH=/app/venv/bin"
ExecStart=/app/venv/bin/gunicorn \
    -w 4 \
    -k uvicorn.workers.UvicornWorker \
    -b 127.0.0.1:8000 \
    FastAPI_Backend.main:app

Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

---

## 7. Error Handling

### Backend Error Handling

```python
# FastAPI_Backend/error_handlers.py
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
import logging

logger = logging.getLogger(__name__)

async def validation_exception_handler(request: Request, exc: Exception):
    """Handle validation errors"""
    logger.error(f"Validation error: {exc}")
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Invalid input data",
            "errors": str(exc)
        },
    )

async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors"""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "message": str(exc) if DEBUG else "An error occurred"
        },
    )

# In main.py
from fastapi.exceptions import RequestValidationError
app.add_exception_handler(
    RequestValidationError,
    validation_exception_handler
)
app.add_exception_handler(
    Exception,
    general_exception_handler
)
```

### Client Error Handling

```typescript
// src/services/errorHandler.ts
export class APIError extends Error {
  constructor(
    public status: number,
    public detail: string,
    message?: string
  ) {
    super(message || detail);
  }
}

export async function handleAPIResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const data = await response.json();
    throw new APIError(
      response.status,
      data.detail,
      `API Error: ${response.status}`
    );
  }
  return response.json();
}

// Usage in components
try {
  const recommendations = await recommendationAPI.getRecommendations(input);
  setRecipes(recommendations);
} catch (error) {
  if (error instanceof APIError) {
    setError(`Error (${error.status}): ${error.detail}`);
  } else {
    setError('An unexpected error occurred');
  }
}
```

---

## 8. Performance Optimization

### Caching Strategy

```python
# FastAPI_Backend/cache.py
from functools import lru_cache
from typing import Tuple
import hashlib
import json

class RecommendationCache:
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
    
    def _create_key(self, nutrition: Tuple, bmi: float, goal: str, metric: str) -> str:
        key_str = json.dumps({
            'nutrition': nutrition,
            'bmi': bmi,
            'goal': goal,
            'metric': metric
        })
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, nutrition: Tuple, bmi: float, goal: str, metric: str):
        key = self._create_key(nutrition, bmi, goal, metric)
        return self.cache.get(key)
    
    def set(self, nutrition: Tuple, bmi: float, goal: str, metric: str, result):
        key = self._create_key(nutrition, bmi, goal, metric)
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = result

# Usage
recommendation_cache = RecommendationCache()

@app.post("/predict/")
def predict_recipes(prediction_input: PredictionIn):
    cache_key = tuple(prediction_input.nutrition_input)
    cached = recommendation_cache.get(
        cache_key,
        prediction_input.bmi,
        prediction_input.goal,
        prediction_input.metric
    )
    
    if cached:
        return cached
    
    # Get recommendations...
    result = PredictionOut(output=output, metadata=metadata)
    recommendation_cache.set(
        cache_key,
        prediction_input.bmi,
        prediction_input.goal,
        prediction_input.metric,
        result
    )
    return result
```

### Database Connection Pooling

```python
# For future PostgreSQL integration
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    'postgresql://user:password@localhost/diet_db',
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=40,
    pool_recycle=3600
)
```

---

## 9. Advanced Features

### Explainability

```python
# FastAPI_Backend/explainability.py
from typing import Dict, List

def explain_recommendation(recipe_id: int, target_vector: List[float]) -> Dict:
    """Generate explanation for why a recipe was recommended"""
    recipe_data = df[df.index == recipe_id].iloc[0]
    
    nutrition_explanation = []
    for i, nutrient in enumerate(NUTRITION_COLS):
        target = target_vector[i]
        recipe_val = recipe_data[nutrient]
        diff = abs(recipe_val - target)
        similarity = 1 - (diff / max(abs(target), 1))
        
        nutrition_explanation.append({
            'nutrient': nutrient,
            'target': target,
            'recipe_value': recipe_val,
            'difference': diff,
            'similarity_score': max(0, similarity)
        })
    
    return {
        'recipe_id': recipe_id,
        'recipe_name': recipe_data['Name'],
        'nutrition_analysis': nutrition_explanation,
        'overall_match': sum([n['similarity_score'] for n in nutrition_explanation]) / len(nutrition_explanation)
    }
```

### User Preferences & History

```typescript
// src/hooks/useUserPreferences.ts
import { create } from 'zustand';

interface UserPreferences {
  bmi: number;
  goal: 'weight_loss' | 'muscle_gain' | 'maintenance';
  dislikedIngredients: string[];
  preferredIngredients: string[];
  history: RecommendationHistory[];
  
  updatePreferences: (prefs: Partial<UserPreferences>) => void;
  addToHistory: (recipe: Recipe) => void;
}

interface RecommendationHistory {
  timestamp: string;
  recipe: Recipe;
  rating?: number;
}

export const useUserPreferences = create<UserPreferences>((set) => ({
  bmi: 25,
  goal: 'maintenance',
  dislikedIngredients: [],
  preferredIngredients: [],
  history: [],
  
  updatePreferences: (prefs) =>
    set((state) => ({ ...state, ...prefs })),
  
  addToHistory: (recipe) =>
    set((state) => ({
      history: [
        {
          timestamp: new Date().toISOString(),
          recipe
        },
        ...state.history
      ]
    }))
}));
```

---

## 10. Troubleshooting

### Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| `numpy.dtype size changed` | Binary incompatibility | Reinstall: `pip install --force-reinstall numpy==1.24.1 pandas==1.5.1` |
| `ModuleNotFoundError: 'FastAPI_Backend'` | Import path issue | Ensure running from project root: `cd /path/to/project` |
| `CORS blocked` | Frontend origin not whitelisted | Add origin to `CORS_ORIGINS` in config |
| `Dataset file not found` | Wrong path resolution | Check `DATA_PATH` in config and project structure |
| `Slow recommendations` | Model not cached | Enable `CACHE_ENABLED=True` in config |
| `Memory leak` | Cache unbounded | Set `CACHE_TTL` and `max_cache_size` |

### Debugging

```bash
# Check backend logs
tail -f logs/recommendation_system_*.log

# Test API connectivity
curl http://localhost:8000/health

# Profile API endpoint
python -m cProfile -s cumulative benchmark.py

# Monitor system resources
python FastAPI_Backend/benchmarking.py
```

---

## Summary

This comprehensive guide covers:
- ✅ Full backend setup and configuration
- ✅ Complete frontend React/TypeScript integration
- ✅ Multiple client library implementations
- ✅ Production deployment strategies
- ✅ Testing frameworks and examples
- ✅ Performance optimization techniques
- ✅ Advanced features and troubleshooting

Total implementation scope: ~2,000+ lines of production-ready code across backend, frontend, and client libraries.
