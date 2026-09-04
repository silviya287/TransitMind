# TransitMind 🚌

### AI-Based Adaptive Bus Scheduling & Route Optimization System

TransitMind is a machine-learning-based decision-support system designed to help public transport operators understand passenger demand, analyze route patterns, predict operational requirements, and recommend adaptive bus scheduling actions.

The system combines supervised learning, unsupervised learning, ensemble methods, and reinforcement learning to transform transport data into actionable scheduling recommendations.

---

## 🎯 Problem Statement

Public bus systems operate with changing passenger demand throughout the day. Fixed schedules can lead to:

- Overcrowded buses during peak periods
- Underutilized buses during low-demand periods
- Longer passenger waiting times
- Inefficient allocation of available buses
- Difficulty responding to changing operational conditions

TransitMind aims to provide a data-driven approach to these problems.

---

## 💡 Proposed Solution

TransitMind analyzes historical and operational transport data to:

1. Analyze existing bus routes and schedules
2. Identify demand patterns across routes and time periods
3. Predict passenger/operational demand
4. Classify demand conditions
5. Discover similar route and demand patterns
6. Compare multiple machine-learning models
7. Use Q-Learning to recommend adaptive scheduling actions

The system is designed as a **decision-support prototype**, rather than a system that directly controls real-world buses.

---

## 🧠 Machine Learning Components

### Regression

Used for predicting continuous demand-related values.

Models explored include:

- Linear Regression
- Multiple Linear Regression
- Polynomial Regression
- Decision Tree Regression
- Random Forest Regression
- Support Vector Regression
- Ridge Regression
- Lasso Regression

Evaluation metrics:

- MAE
- MSE
- RMSE
- R² Score

---

### Classification

Used to categorize transport demand or operational conditions.

Models explored include:

- K-Nearest Neighbors
- Linear SVM
- Soft Margin SVM
- RBF Kernel SVM
- Polynomial Kernel SVM
- Sigmoid Kernel SVM

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

### Clustering

Used to discover naturally occurring patterns among routes and demand conditions.

Algorithms explored include:

- K-Means
- Hierarchical Clustering
- K-Medoids
- DBSCAN
- Gaussian Mixture Models

Evaluation:

- Elbow Method
- Silhouette Score

---

### Ensemble Learning

Multiple models can be combined to improve prediction robustness.

Techniques explored include:

- Voting
- Averaging
- Weighted Averaging
- Bagging
- Random Forest
- AdaBoost
- Gradient Boosting
- Stacking

---

### Reinforcement Learning

The final decision-support layer uses **Q-Learning**.

The environment represents transport scheduling conditions.

#### State

A state can contain:

- Route
- Passenger demand
- Available buses
- Time period
- Traffic/operational condition

#### Actions

Possible scheduling actions include:

- Keep current allocation
- Add a bus
- Remove a bus
- Increase frequency
- Decrease frequency

#### Reward

The reward function is designed to encourage:

- Lower passenger waiting time
- Reduced overcrowding
- Better bus utilization
- Fewer operational delays

---

## 📊 Current Data

TransitMind currently uses real-world **PMPML (Pune Mahanagar Parivahan Mahamandal Ltd.)** transit schedule data.

The GTFS data provides information such as:

- Routes
- Stops
- Trips
- Scheduled arrival/departure times
- Route shapes
- Service information

Additional operational/revenue data is being evaluated to strengthen the demand and scheduling analysis.

Raw source datasets are intentionally excluded from GitHub when they are large.

---

## 🏗️ System Architecture

```text
Transport Data
      │
      ▼
Data Preprocessing
      │
      ▼
Feature Engineering
      │
      ├───────────────┐
      ▼               ▼
 Regression      Classification
      │               │
      └───────┬───────┘
              ▼
          Clustering
              │
              ▼
       Ensemble Models
              │
              ▼
        Q-Learning Agent
              │
              ▼
   Adaptive Scheduling Action
              │
              ▼
        Streamlit Dashboard
    ```
    