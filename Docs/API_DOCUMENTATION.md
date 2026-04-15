# Diet Recommendation System API Documentation

## Overview

The Diet Recommendation System API provides AI-powered recipe recommendations based on nutritional requirements, user health metrics, and fitness goals. The system uses multiple machine learning algorithms to ensure optimal recommendations.

## Base URL
```
http://localhost:8000
```

## Authentication
Currently, no authentication is required. All endpoints are publicly accessible.

## Rate Limiting
- 100 requests per minute per IP address
- 1000 requests per hour per IP address

## Endpoints

### 1. Health Check
Get the health status of the API.

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2026-04-15T10:30:00Z"
}
```

### 2. Model Information
Get information about available models and metrics.

**Endpoint:** `GET /models/info`

**Response:**
```json
{
  "available_models": [
    "knn_cosine",
    "knn_euclidean",
    "kmeans",
    "svd",
    "hybrid"
  ],
  "available_metrics": [
    "nutritional_mae",
    "diversity_score"
  ],
  "default_metric": "nutritional_mae"
}
```

### 3. Model Performance
Get performance metrics for all models.

**Endpoint:** `GET /models/performance`

**Response:**
```json
{
  "models": {
    "knn_cosine": {
      "nutritional_mae": 12.05,
      "diversity_score": 0.234,
      "latency_ms": 15.2
    },
    "knn_euclidean": {
      "nutritional_mae": 12.05,
      "diversity_score": 0.245,
      "latency_ms": 16.8
    },
    "kmeans": {
      "nutritional_mae": 12.05,
      "diversity_score": 0.456,
      "latency_ms": 18.3
    },
    "svd": {
      "nutritional_mae": 12.05,
      "diversity_score": 0.267,
      "latency_ms": 22.1
    },
    "hybrid": {
      "nutritional_mae": 12.05,
      "diversity_score": 0.234,
      "latency_ms": 17.9
    }
  },
  "recommendations": {
    "nutritional_accuracy": "hybrid",
    "maximum_diversity": "kmeans",
    "fastest": "knn_cosine"
  }
}
```

### 4. Recipe Recommendations
Generate personalized recipe recommendations.

**Endpoint:** `POST /predict/`

**Request Body:**
```json
{
  "nutrition_input": [
    500.0,    // calories
    20.0,     // fat_content
    5.0,      // saturated_fat_content
    50.0,     // cholesterol_content
    400.0,    // sodium_content
    40.0,     // carbohydrate_content
    10.0,     // fiber_content
    5.0,      // sugar_content
    35.0      // protein_content
  ],
  "ingredients": ["chicken", "rice", "broccoli"],
  "params": {
    "n_neighbors": 5,
    "return_distance": false
  },
  "bmi": 24.5,
  "goal": "weight_loss",
  "metric": "nutritional_mae"
}
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `nutrition_input` | array[float] | Yes | Target nutritional values (9 values) |
| `ingredients` | array[string] | No | Preferred ingredients to filter by |
| `params.n_neighbors` | integer | No | Number of recommendations (1-100) |
| `bmi` | float | Yes | Body Mass Index for personalization |
| `goal` | string | Yes | Fitness goal: "weight_loss", "muscle_gain", "maintenance" |
| `metric` | string | Yes | Selection metric: "nutritional_mae", "diversity_score" |

**Response:**
```json
{
  "output": [
    {
      "Name": "Grilled Chicken with Rice",
      "CookTime": "30 min",
      "PrepTime": "15 min",
      "TotalTime": "45 min",
      "RecipeIngredientParts": [
        "chicken breast",
        "brown rice",
        "broccoli",
        "olive oil",
        "garlic"
      ],
      "Calories": 485.0,
      "FatContent": 18.5,
      "SaturatedFatContent": 4.2,
      "CholesterolContent": 45.0,
      "SodiumContent": 380.0,
      "CarbohydrateContent": 42.0,
      "FiberContent": 8.5,
      "SugarContent": 3.2,
      "ProteinContent": 38.0,
      "RecipeInstructions": [
        "Season chicken with salt and pepper",
        "Grill chicken for 15 minutes",
        "Cook rice according to package",
        "Steam broccoli until tender",
        "Serve together"
      ]
    }
  ],
  "metadata": {
    "model_used": "hybrid",
    "metric_basis": "nutritional_mae",
    "total_candidates": 50,
    "filtered_results": 5,
    "processing_time_ms": 17.9
  }
}
```

### 5. User Feedback
Submit feedback on recommendations for system improvement.

**Endpoint:** `POST /feedback/`

**Request Body:**
```json
{
  "user_id": "user123",
  "recipe_id": 12345,
  "rating": 4,
  "was_helpful": true,
  "comments": "Great recipe, but a bit too spicy",
  "session_id": "session_abc123"
}
```

**Response:**
```json
{
  "status": "feedback_received",
  "feedback_id": "fb_123456",
  "message": "Thank you for your feedback!"
}
```

### 6. Model Explainability
Get explanation for why a specific recipe was recommended.

**Endpoint:** `GET /explain/{recipe_id}`

**Query Parameters:**
- `nutrition_input`: Target nutrition values (comma-separated)
- `bmi`: Body Mass Index
- `goal`: Fitness goal
- `metric`: Selection metric

**Example:** `GET /explain/12345?nutrition_input=500,20,5,50,400,40,10,5,35&bmi=24.5&goal=weight_loss&metric=nutritional_mae`

**Response:**
```json
{
  "recipe_id": 12345,
  "recipe_name": "Grilled Chicken with Rice",
  "explanation": {
    "primary_factors": [
      {
        "nutrient": "ProteinContent",
        "target": 35.0,
        "recipe_value": 38.0,
        "contribution": 0.85,
        "description": "High protein content matches your target"
      },
      {
        "nutrient": "Calories",
        "target": 500.0,
        "recipe_value": 485.0,
        "contribution": 0.72,
        "description": "Calorie content very close to target"
      }
    ],
    "health_considerations": [
      {
        "factor": "sugar_content",
        "penalty": -0.15,
        "reason": "Reduced sugar aligns with weight loss goal"
      }
    ],
    "similarity_score": 0.89,
    "health_penalty": 0.12,
    "final_score": 0.92
  },
  "model_used": "hybrid",
  "confidence": 0.92
}
```

## Data Models

### Nutrition Input Format
The `nutrition_input` array must contain exactly 9 float values in this order:
1. Calories (kcal)
2. Fat Content (g)
3. Saturated Fat Content (g)
4. Cholesterol Content (mg)
5. Sodium Content (mg)
6. Carbohydrate Content (g)
7. Fiber Content (g)
8. Sugar Content (g)
9. Protein Content (g)

### Recipe Response Format
Each recipe in the response includes:
- Basic information (name, times)
- Complete ingredient list
- Full nutritional breakdown
- Step-by-step instructions

### Error Responses

#### Validation Error (400)
```json
{
  "detail": [
    {
      "loc": ["body", "nutrition_input"],
      "msg": "ensure this value has at least 9 items",
      "type": "value_error.const"
    }
  ]
}
```

#### Invalid Metric (400)
```json
{
  "detail": "Invalid metric. Must be one of: nutritional_mae, diversity_score"
}
```

#### Server Error (500)
```json
{
  "detail": "Internal server error. Please try again later."
}
```

## Usage Examples

### Python Client
```python
import requests

# Get recommendations
response = requests.post("http://localhost:8000/predict/", json={
    "nutrition_input": [500, 20, 5, 50, 400, 40, 10, 5, 35],
    "bmi": 24.5,
    "goal": "weight_loss",
    "metric": "nutritional_mae"
})

recommendations = response.json()
print(f"Got {len(recommendations['output'])} recommendations")
```

### cURL Example
```bash
curl -X POST "http://localhost:8000/predict/" \
  -H "Content-Type: application/json" \
  -d '{
    "nutrition_input": [500, 20, 5, 50, 400, 40, 10, 5, 35],
    "bmi": 24.5,
    "goal": "weight_loss",
    "metric": "nutritional_mae"
  }'
```

## Performance Characteristics

- **Average Response Time**: <20ms for recommendations
- **Throughput**: 100+ requests/second
- **Memory Usage**: <500MB
- **Dataset Size**: 360k+ recipes
- **Model Count**: 5 different algorithms

## Error Handling

The API implements comprehensive error handling:
- Input validation with detailed error messages
- Graceful degradation for model failures
- Structured logging for debugging
- Rate limiting to prevent abuse

## Version History

- **v1.0.0** (April 2026): Initial release with 5 ML algorithms
  - Hybrid scoring implementation
  - Metric-based model selection
  - Comprehensive testing suite
  - Performance benchmarking

## Support

For API support or bug reports:
- Email: support@diet-recommendation-system.com
- Documentation: https://docs.diet-recommendation-system.com
- GitHub Issues: https://github.com/your-repo/issues

## License

This API is provided under the MIT License. See LICENSE file for details.