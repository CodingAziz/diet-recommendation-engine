# Diet Recommendation System - Prompts for Diagrams & Documentation

## 1. System Architecture Diagram

### Prompt for Generating Mermaid/Visual Diagram:

```
Generate a comprehensive system architecture diagram for a Diet Recommendation System with the following components and flow:

**Layers:**
1. **Client Layer**: Web UI (React/Vite frontend), Mobile Apps, Third-party Integrations
2. **API Gateway Layer**: FastAPI with CORS middleware, input validation, authentication, rate limiting, and health checks
3. **Business Logic Layer**: 
   - Recommendation Engine (orchestrates ML models)
   - Personalization Engine (applies health metrics: BMI, fitness goal)
   - Validation Engine (input sanitization)
   - Formatting Engine (response structuring)
4. **ML Models Layer**: Contains 5 algorithms:
   - KNN Cosine Similarity (MinMaxScaler, ~15ms)
   - KNN Euclidean Distance (StandardScaler, ~17ms)
   - K-Means Clustering (20 clusters, ~18ms)
   - SVD-based Collaborative Filtering (5 components, ~22ms)
   - Hybrid Scoring (KNN + BMI/goal penalties, ~18ms)
5. **Data Layer**: 
   - Recipe Database (360k+ recipes)
   - Feature Store (pre-computed scalers and model indices)
   - Caching Layer (for frequent queries)
   - Preprocessing Pipeline

**Data Flow:**
- User Input: [Calories, Fat, Sat Fat, Cholesterol, Sodium, Carbs, Fiber, Sugar, Protein] + Ingredients + BMI + Goal
- Processing: Feature scaling → Model selection based on metric → Ranking/filtering
- Output: Top-K recommendations with metadata

**Key Metrics Tracked:**
- Response Latency (<20ms)
- Nutritional MAE (Mean Absolute Error)
- Diversity Score
- Model Coverage

Show all components, their relationships, and data flow with clear connections between layers.
```

---

## 2. Frontend Prompt for Integration with Backend Routes

### Prompt for Frontend Developer:

```
Design a React/Vite frontend for the Diet Recommendation System with the following backend API integration requirements:

**Backend API Endpoints to Integrate:**
1. POST /predict/ - Main recommendation endpoint
   - Input: nutrition values (9 floats), ingredients (list), BMI (float), goal (weight_loss|muscle_gain|maintenance), metric (nutritional_mae|diversity_score), n_neighbors (1-100)
   - Output: List of Recipe objects with Name, CookTime, PrepTime, Calories, nutritional content, ingredients, instructions

2. GET /health - Health check
   - Displays system status, version, timestamp

3. GET /models/info - Get available models
   - Returns: knn_cosine, knn_euclidean, kmeans, svd, hybrid

4. GET /models/performance - Performance metrics
   - Returns: nutritional_mae, diversity_score, latency_ms, coverage for each model

5. POST /feedback/ - Collect user feedback
   - Input: user_id, recipe_id, rating (1-5), was_helpful (bool), comments, session_id
   - Output: feedback_id, status confirmation

6. GET /explain/{recipe_id} - Model explainability
   - Returns: explanation dict, model used, confidence score

7. POST /models/feature-importance - Feature importance analysis
   - Returns: feature importance list, methodology

**Frontend Components to Build:**
- **Input Component**: 
  - Nutrition value sliders/inputs (for 9 nutrition values)
  - Ingredient multi-select filter
  - BMI input field
  - Fitness goal selector (radio/dropdown)
  - Metric selector (diversity vs accuracy)
  - Number of recommendations slider

- **Results Display Component**:
  - Recipe cards showing name, image, prep/cook time, nutritional values
  - Ingredient list with expandable view
  - Cooking instructions (collapsible)
  - Rating/feedback button

- **Model Performance Dashboard**:
  - Comparison table of all 5 models
  - Performance metrics visualization
  - Recommendations for each metric type

- **Feedback Component**:
  - Star rating system
  - Helpful/not helpful toggle
  - Comments textarea
  - Submit feedback button

- **Explainability Component**:
  - Display model explanation for recommended recipes
  - Show which features influenced the recommendation
  - Confidence score visualization

**Functional Requirements:**
- Real-time form validation
- Loading state management
- Error handling with user-friendly messages
- Responsive design (mobile/tablet/desktop)
- Caching of recent recommendations
- Session tracking for analytics
- Dark/light mode support

**Performance Requirements:**
- API response time: <20ms
- Frontend load time: <3s
- Smooth animations and transitions
- Optimized re-renders
```

---

## 3. Data Flow Diagram

### Prompt for Generating Data Flow Diagram:

```
Create a detailed Data Flow Diagram (DFD) for the Diet Recommendation System showing data movement between components:

**Level 0 (System Context):**
- User (external entity) → [Diet Recommendation System] → Recommended Recipes
- System ↔ Recipe Database (external data store)

**Level 1 (Major Processes):**

Process 1.0: Accept User Input
- Input: User Profile (BMI, goal, nutrition requirements, preferred ingredients)
- Validate inputs using Validators module
- Output: Validated user request

Process 2.0: Select & Execute Recommendation Model
- Input: Validated request + Recipe Database
- Processes: 
  - Feature Scaling (MinMaxScaler or StandardScaler)
  - Model Selection based on metric (nutritional_mae vs diversity_score)
  - Execute chosen algorithm (KNN Cosine/Euclidean, K-Means, SVD, or Hybrid)
- Output: Initial candidate recipes with similarity scores

Process 3.0: Apply Personalization
- Input: Candidate recipes + User health profile (BMI, goal)
- Apply penalties based on:
  - BMI ≥ 30: Reduce calorie and fat content scores
  - Weight loss goal: Penalize sugar content
  - Maintenance: No additional penalties
- Output: Re-ranked recipes

Process 4.0: Filter & Format Results
- Input: Re-ranked recipes + Ingredient preferences
- Filter by selected ingredients
- Format output with metadata
- Output: Final recommendations with timestamps

Process 5.0: Generate Explanations (Optional)
- Input: Selected recipe + Model used
- Use SHAP-based explainability
- Output: Feature importance, confidence scores

**Data Stores:**
- D1: Recipe Database (360k+ recipes with 15+ nutritional attributes)
- D2: Feature Store (pre-computed scalers, model indices)
- D3: Cache (frequently accessed results, TTL-based invalidation)
- D4: User Feedback (ratings, helpful flags, comments)
- D5: Performance Logs (latency, accuracy metrics, error tracking)

**Data Elements:**
- Nutrition Vector: [Calories, Fat, SatFat, Cholesterol, Sodium, Carbs, Fiber, Sugar, Protein]
- Recipe Record: Name, Instructions, Ingredients, All nutrition values, Cook/Prep times
- User Profile: BMI, Fitness goal, Preferences, Session ID
- Recommendation Record: Recipe + Score + Confidence + Model used + Timestamp

Show all flows with clear labels and data structure annotations.
```

---

## 4. Use Case Diagram

### Prompt for Generating Use Case Diagram:

```
Create a comprehensive Use Case Diagram for the Diet Recommendation System with the following actors and use cases:

**Primary Actors:**
1. Regular User (end consumer seeking recipes)
2. Admin/System Operator (system monitoring and management)
3. Third-party Application (API consumer)

**Use Cases:**

**For Regular User:**
UC1: Get Recipe Recommendations
  - Precondition: User has account/session
  - Flow: Enter nutrition targets, BMI, goal → Select metric → Receive recommendations
  - Extensions: Filter by ingredients, adjust parameters

UC2: View Recipe Details
  - View full recipe with instructions, ingredients, nutritional breakdown
  - See cooking times, difficulty level

UC3: Rate Recommendation
  - Rate recipe (1-5 stars)
  - Mark as helpful/not helpful
  - Add comments
  - System uses feedback for model improvement

UC4: View Model Explanations
  - Understand why a recipe was recommended
  - See feature importance and model confidence
  - UC4a: View nutritional impact analysis

UC5: Save Favorites
  - Save recommended recipes to personal list
  - Track recipe history

**For Admin/Operator:**
UC6: Monitor System Health
  - View API status, uptime, error rates
  - Check performance metrics for all models
  - View latency and response time trends

UC7: Manage Models
  - Update model parameters
  - Retrain models with new data
  - Switch between models based on performance

UC8: View Analytics Dashboard
  - User engagement metrics
  - Most recommended recipes
  - Model performance comparison
  - Feature importance trends

UC9: Manage Data
  - Update recipe database
  - Archive old feedback
  - Purge cache

**For Third-party Application:**
UC10: Call Recommendation API
  - Send request with nutrition parameters
  - Receive recommendations in JSON format
  - Include error handling

UC11: Access Performance Metrics
  - Get model performance data
  - Track API usage

UC12: Provide User Feedback
  - Submit ratings and comments via API
  - Update recommendation accuracy

**System Processes (Internal):**
UC13: Calculate Nutritional Match
  - Compute distances between user requirements and recipes

UC14: Apply Personalization Rules
  - Apply health-based penalties based on BMI and goal

UC15: Generate Explanations
  - Create SHAP-based explanations for recommendations

UC16: Log Performance Metrics
  - Track latency, accuracy, coverage

Show all actors with relationships (associations, extends, includes) to use cases.
```

---

## 5. Class Diagram

### Prompt for Generating Class Diagram:

```
Create a comprehensive Class Diagram for the Diet Recommendation System with the following classes:

**API Models/Schemas:**

Class: PredictionIn
- nutrition_input: List[float] (length 9)
- ingredients: List[str]
- params: ParamConfig
- bmi: float
- goal: string (enum: weight_loss, muscle_gain, maintenance)
- metric: string (enum: nutritional_mae, diversity_score)

Class: ParamConfig
- n_neighbors: int (1-100, default: 5)
- return_distance: bool

Class: Recipe
- Name: string
- CookTime: string
- PrepTime: string
- TotalTime: string
- RecipeIngredientParts: List[string]
- Calories: float
- FatContent: float
- SaturatedFatContent: float
- CholesterolContent: float
- SodiumContent: float
- CarbohydrateContent: float
- FiberContent: float
- SugarContent: float
- ProteinContent: float
- RecipeInstructions: List[string]

Class: PredictionOut
- output: Optional[List[Recipe]]
- metadata: Dict[string, Any]

Class: FeedbackIn
- user_id: string
- recipe_id: int
- rating: int (1-5)
- was_helpful: bool
- comments: Optional[string]
- session_id: Optional[string]

Class: ModelPerformanceMetrics
- nutritional_mae: float
- diversity_score: float
- latency_ms: float
- coverage: float

Class: ExplanationResponse
- recipe_id: int
- recipe_name: string
- explanation: Dict[string, Any]
- model_used: string
- confidence: float
- timestamp: datetime

**Business Logic Classes:**

Class: RecommendationEngine
- dataset: DataFrame
- models: Dict[string, Model]
- scalers: Dict[string, Scaler]
+ recommend(nutrition_input, metric): List[Recipe]
+ select_model(metric): Model
+ rank_results(candidates, bmi, goal): List[Recipe]

Class: PersonalizationEngine
- health_profiles: Dict[string, HealthProfile]
+ apply_health_penalty(recipes, bmi, goal): List[Recipe]
+ calculate_bmi_penalty(calories, fat, bmi): float
+ calculate_goal_penalty(sugar, goal): float

Class: ValidationEngine
- rules: Dict[string, Rule]
+ validate_nutrition_input(values): bool
+ validate_ingredients(ingredients): bool
+ validate_bmi(bmi): bool
+ validate_goal(goal): bool

Class: FormattingEngine
+ format_recipe_output(recipe_df): Recipe
+ add_metadata(recipes, model_used): Dict
+ serialize_response(output, metadata): JSON

**ML Model Classes:**

Interface: RecommendationModel
+ recommend(target_vector, k): List[int]

Class: KNNCosineModel implements RecommendationModel
- knn: NearestNeighbors
- scaler: MinMaxScaler
+ recommend(target_vector, k): List[int]
+ fit(X): void

Class: KNNEuclideanModel implements RecommendationModel
- knn: NearestNeighbors
- scaler: StandardScaler
+ recommend(target_vector, k): List[int]
+ fit(X): void

Class: KMeansModel implements RecommendationModel
- kmeans: KMeans
- scaler: StandardScaler
- n_clusters: int = 20
+ recommend(target_vector, k): List[int]
+ fit(X): void

Class: SVDModel implements RecommendationModel
- svd: TruncatedSVD
- knn: NearestNeighbors
- scaler: StandardScaler
+ recommend(target_vector, k): List[int]
+ fit(X): void

Class: HybridModel
- base_model: RecommendationModel
- personalization_engine: PersonalizationEngine
+ recommend(target_vector, k, bmi, goal): List[int]
+ apply_hybrid_scoring(candidates, bmi, goal): List[int]

**Data Classes:**

Class: HealthProfile
- user_id: string
- bmi: float
- goal: string (enum)
- dietary_restrictions: List[string]
- preferred_ingredients: List[string]

Class: NutritionVector
- calories: float
- fat: float
- saturated_fat: float
- cholesterol: float
- sodium: float
- carbohydrates: float
- fiber: float
- sugar: float
- protein: float
+ to_array(): List[float]

**Configuration Classes:**

Class: Settings
- app_version: string = "1.0.0"
- app_environment: string
- cors_origins: List[string]
- rate_limit_rpm: int = 100
- log_level: string

Show all classes with attributes, methods, inheritance, and relationships (associations, dependencies).
```

---

## 6. Sequence Diagram

### Prompt for Generating Sequence Diagram:

```
Create a detailed Sequence Diagram for the Diet Recommendation System showing the interaction flow for the main "Get Recommendations" use case:

**Main Flow: User Requests Recipe Recommendations**

Actors/Components: User, Frontend, FastAPI Gateway, ValidationEngine, ModelSelector, KNNCosineModel, PersonalizationEngine, FormattingEngine, Database, User Response

**Sequence of Interactions:**

1. User → Frontend: Enter nutrition values, BMI, goal, metric, ingredients
   
2. Frontend → FastAPI (/predict/): POST request with PredictionIn payload
   
3. FastAPI → ValidationEngine: validate_input(nutrition_input, bmi, goal, metric)
   - Alt: Invalid input
     - ValidationEngine → FastAPI: Raise HTTPException 422
     - FastAPI → Frontend: Error response with validation details
     - Frontend → User: Display error message
   
4. FastAPI → ModelSelector: select_model(metric="nutritional_mae")
   - ModelSelector → ModelSelector: Check metric type
   - ModelSelector → FastAPI: Return KNNCosineModel
   
5. FastAPI → KNNCosineModel: recommend(nutrition_input, n_neighbors=50)
   - KNNCosineModel → MinMaxScaler: Transform nutrition_input
   - MinMaxScaler → KNNCosineModel: Scaled vector
   - KNNCosineModel → NearestNeighbors: Find K nearest neighbors
   - NearestNeighbors → KNNCosineModel: Return 50 candidate recipes (indices)
   - KNNCosineModel → Database: Fetch recipe details for indices
   - Database → KNNCosineModel: Recipe objects
   - KNNCosineModel → FastAPI: Return 50 candidates with scores

6. FastAPI → PersonalizationEngine: apply_personalization(candidates, bmi=25.5, goal="maintenance")
   - Alt: BMI ≥ 30
     - PersonalizationEngine → PersonalizationEngine: Calculate calorie & fat penalties
   - Alt: goal == "weight_loss"
     - PersonalizationEngine → PersonalizationEngine: Calculate sugar penalties
   - PersonalizationEngine → FastAPI: Return re-ranked recipes
   
7. FastAPI → FilterEngine: filter_by_ingredients(recipes, ["chicken", "broccoli"])
   - FilterEngine → FastAPI: Return filtered recipes (top 10)
   
8. FastAPI → FormattingEngine: format_output(filtered_recipes)
   - FormattingEngine → FormattingEngine: Create Recipe objects
   - FormattingEngine → FormattingEngine: Add metadata (model_used, metric_basis, timestamp)
   - FormattingEngine → FastAPI: Return PredictionOut object

9. FastAPI → Logger: Log successful prediction
   - Logger → Logs: Record timestamp, model used, metrics

10. FastAPI → Frontend: Return PredictionOut JSON response
    - Response includes: [List of 10 Recipe objects], metadata

11. Frontend → Frontend: Parse response and render recipe cards
    
12. Frontend → User: Display recommendations with:
    - Recipe name, images, prep/cook times
    - Nutritional breakdown
    - Ingredients list
    - Rating buttons

**Alternative Flow: Model Not Available**
- ModelSelector → FastAPI: Model not found exception
- FastAPI → Frontend: 500 error with fallback suggestion
- Frontend → User: Display error and retry option

**Alternative Flow: No Recommendations Found**
- KNNCosineModel → FastAPI: Empty results
- FastAPI → Frontend: 204 No Content or response with empty output
- Frontend → User: "No recipes match your criteria" message

Show timing markers, lifelines for each component, and clear message labels.
```

---

## 7. System Accuracy Metrics & Performance Evaluation Prompts

### Comprehensive Prompt for Metrics Documentation:

```
Create a comprehensive System Accuracy Metrics and Performance Evaluation document for the Diet Recommendation System:

## 1. ACCURACY METRICS

### Nutritional Accuracy (Primary Metric)
- **Metric Name**: Nutritional Mean Absolute Error (MAE)
- **Definition**: Average absolute difference between recommended recipe nutrition and user's target nutrition across all 9 dimensions
- **Formula**: MAE = (1/N) × Σ|predicted_nutrition_i - target_nutrition_i|
- **Where N** = number of recommendations (typically 10)
- **Components Measured**:
  - Calories MAE (±15 cal target)
  - Fat MAE (±2g target)
  - Saturated Fat MAE (±1g target)
  - Cholesterol MAE (±10mg target)
  - Sodium MAE (±200mg target)
  - Carbs MAE (±5g target)
  - Fiber MAE (±2g target)
  - Sugar MAE (±3g target)
  - Protein MAE (±3g target)
- **Target Performance**: < 12.05 MAE across all models
- **Model Performance Breakdown**:
  - KNN Cosine: 12.05 MAE
  - KNN Euclidean: 12.05 MAE
  - K-Means: 12.05 MAE
  - SVD: 12.05 MAE
  - Hybrid: 12.05 MAE

### Diversity Score (Secondary Metric)
- **Metric Name**: Recipe Diversity Score
- **Definition**: Measure of variety in recommended recipes (cuisine type, ingredients, cooking methods)
- **Formula**: Diversity = (1 - avg_pairwise_similarity) × 100
- **Calculation**: Average pairwise cosine similarity of top-K recommendations
- **Interpretation**:
  - High diversity (0.35-0.50): Varied recipes, different cuisines/methods
  - Medium diversity (0.20-0.35): Some variation with focus on best matches
  - Low diversity (0.00-0.20): Very similar recipes, repetitive
- **Model Performance Breakdown**:
  - KNN Cosine: 0.234 diversity score
  - KNN Euclidean: 0.245 diversity score
  - K-Means: 0.456 diversity score (highest diversity)
  - SVD: 0.267 diversity score
  - Hybrid: 0.234 diversity score
- **Target Performance**: > 0.25 diversity score

## 2. PERFORMANCE METRICS

### Response Latency
- **Metric Name**: API Response Time
- **Definition**: Time from request receipt to response transmission
- **Measurement**: milliseconds (ms)
- **Target**: < 20ms
- **Breakdown by Model**:
  - KNN Cosine: ~15ms (fastest)
  - KNN Euclidean: ~17ms
  - Hybrid: ~18ms
  - K-Means: ~18ms
  - SVD: ~22ms (slowest, but best for pattern discovery)
- **Components**:
  - Input validation: ~1-2ms
  - Feature scaling: ~2-3ms
  - Model inference: ~8-15ms
  - Personalization: ~1-2ms
  - Formatting & serialization: ~1-2ms

### Throughput
- **Metric Name**: Requests Per Second (RPS)
- **Definition**: Number of recommendations generated per second
- **Target**: 50+ RPS
- **Formula**: RPS = Total requests / Time period
- **Peak Load Target**: Handle 1000 concurrent users

### Coverage
- **Metric Name**: Recipe Coverage
- **Definition**: Percentage of recipe database that can be recommended
- **Calculation**: Recipes recommended / Total recipes × 100
- **Target**: > 85% coverage
- **Factors**:
  - Ingredient availability
  - Nutritional constraints
  - User preferences

## 3. MODEL COMPARISON METRICS

| Metric | KNN Cosine | KNN Euclidean | K-Means | SVD | Hybrid |
|--------|-----------|---------------|---------|-----|--------|
| Nutritional MAE | 12.05 | 12.05 | 12.05 | 12.05 | 12.05 |
| Diversity Score | 0.234 | 0.245 | 0.456 | 0.267 | 0.234 |
| Latency (ms) | 15 | 17 | 18 | 22 | 18 |
| Best For | Speed | Magnitude | Diversity | Patterns | Accuracy |
| Scalability | Excellent | Excellent | Good | Good | Excellent |

## 4. STATISTICAL VALIDATION METRICS

### Precision@K
- **Definition**: Proportion of top-K recommendations that meet user criteria
- **Formula**: Precision@10 = (Relevant items in top-10) / 10
- **Target**: > 0.85 (85% of recommendations should be relevant)

### Recall@K
- **Definition**: Proportion of all relevant recipes in top-K
- **Formula**: Recall@10 = (Relevant items in top-10) / Total relevant items
- **Target**: > 0.75

### Normalized Discounted Cumulative Gain (NDCG@K)
- **Definition**: Ranking quality metric considering position
- **Formula**: NDCG@10 = DCG@10 / IDCG@10
- **Target**: > 0.82

### Mean Reciprocal Rank (MRR)
- **Definition**: Average inverse rank of first relevant result
- **Formula**: MRR = (1/N) × Σ(1/rank_i)
- **Target**: > 0.90

## 5. USER SATISFACTION METRICS

### Rating Distribution
- **Average User Rating**: Collect 1-5 star ratings
- **Target**: 4.0+ average rating
- **Distribution Target**:
  - 5 stars: 50%+
  - 4 stars: 30%+
  - 3 stars: 10%-
  - 1-2 stars: <5%

### Helpfulness Rate
- **Definition**: Percentage of recommendations marked as "helpful"
- **Formula**: Helpful rate = (Helpful ratings) / (Total ratings) × 100
- **Target**: > 80% helpful rate

### Click-Through Rate (CTR)
- **Definition**: Percentage of recommended recipes users interact with
- **Formula**: CTR = (Clicks) / (Impressions) × 100
- **Target**: > 35% CTR

### Conversion Rate
- **Definition**: Percentage of recommendations that users cook/prepare
- **Formula**: Conversion = (Cooked recipes) / (Recommended recipes) × 100
- **Target**: > 20% conversion rate

## 6. SYSTEM RELIABILITY METRICS

### Uptime
- **Target**: 99.95% (11.6 minutes downtime/month)
- **Measured**: (Operational time / Total time) × 100

### Error Rate
- **Definition**: Percentage of requests that result in errors
- **Formula**: Error rate = (Failed requests) / (Total requests) × 100
- **Target**: < 0.1% error rate
- **Breakdown**:
  - 4xx errors (client errors): < 0.05%
  - 5xx errors (server errors): < 0.01%

### Mean Time Between Failures (MTBF)
- **Target**: > 720 hours (30 days)

### Mean Time to Recovery (MTTR)
- **Target**: < 5 minutes

## 7. SCALABILITY METRICS

### Query Response Time under Load
- **Scenario**: 100 concurrent users
- **Target**: < 50ms response time
- **Scenario**: 1000 concurrent users
- **Target**: < 200ms response time

### Memory Usage
- **Target**: < 2GB for production deployment
- **Model storage**: ~500MB
- **Cache layer**: ~500MB
- **Operational memory**: < 1GB

### Database Query Time
- **Target**: < 5ms for recipe lookups
- **Cache hit rate**: > 80%

## 8. MONITORING & ALERTING THRESHOLDS

### Critical Alerts
- Response time > 50ms: CRITICAL
- Error rate > 1%: CRITICAL
- System downtime: CRITICAL
- Memory usage > 80%: WARNING

### Standard Metrics to Track
- Request count per minute
- Average response time per model
- Error count by type
- Cache hit rate
- Database query performance
- User feedback sentiment
- Model accuracy drift over time

## 9. REPORTING & DASHBOARDS

### Real-time Dashboard
- Current system health
- Active requests
- Error rates
- Response time distribution
- Model performance comparison

### Weekly Reports
- System uptime percentage
- Average metrics for each model
- Top performing recommendations
- User satisfaction trends
- Performance anomalies

### Monthly Analysis
- Accuracy trends
- User engagement metrics
- Model retraining recommendations
- Scalability assessment
- Cost-benefit analysis

## 10. CONTINUOUS IMPROVEMENT

### Metric Targets (Year 1)
- Nutritional MAE: Reduce to 10.0
- Diversity Score: Improve to 0.35+
- Latency: Reduce to 12ms
- User satisfaction: Maintain 4.2+ rating
- Error rate: Maintain < 0.05%

### Metric Targets (Year 2)
- Nutritional MAE: Reduce to 8.5
- Diversity Score: Improve to 0.40+
- Latency: Reduce to 8ms
- User satisfaction: Achieve 4.5+ rating
- Error rate: Maintain < 0.02%
```

---

## Summary of All Prompts

| # | Diagram/Document | Use Case |
|---|---|---|
| 1 | System Architecture | Visual overview of system layers and components |
| 2 | Frontend Integration | Guide for frontend development with all API endpoints |
| 3 | Data Flow Diagram | Track data movement through the system |
| 4 | Use Case Diagram | Define user interactions and system behavior |
| 5 | Class Diagram | Detail object-oriented structure |
| 6 | Sequence Diagram | Show interaction flow for key scenarios |
| 7 | Accuracy Metrics | Comprehensive performance evaluation framework |

## How to Use These Prompts

1. **For Diagrams**: Copy each prompt and paste into:
   - Mermaid Live Editor (https://mermaid.live)
   - Lucidchart
   - Draw.io
   - Claude/ChatGPT (with vision capabilities)
   - PlantUML tools

2. **For Documentation**: Use prompts with:
   - Claude (long-form analysis)
   - ChatGPT
   - Code-aware AI tools
   - Documentation generators

3. **For Frontend Development**: Share Prompt #2 with your frontend team

4. **For Monitoring**: Use Prompt #7 metrics to set up monitoring dashboards
