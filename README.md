# 🧠 ml_learning_weekly

This repository documents my **continuous learning journey in Machine Learning**. It serves both as a personal knowledge base and a public demonstration of my growing proficiency in core ML domains.

## 🎯 Goal

To **systematically deepen my understanding** of key Machine Learning concepts through hands-on implementations, code-first experimentation, and weekly documentation.

The focus areas include:

- 📊 **Statistical & Mathematical Theory**  
  Foundations such as linear algebra, calculus, probability, optimization, and information theory.

- 🧠 **Algorithms & Modeling**  
  Core ML topics including regression, classification, clustering, deep learning, ensemble methods, and more.

- ⚙️ **MLOps & Dev Pipelines**  
  Tools and practices for reproducible ML: model tracking, data versioning, deployment, Docker, CI/CD, etc.

- 🧩 **Miscellaneous & Applied Topics**  
  Real-world projects, Kaggle notebooks, research paper reproductions, and exploratory ideas.

---

## 🗂️ Repository Structure

```bash
ml_learning_weekly/
│
├── deep_learning/               # Neural nets, CNNs, backpropagation from scratch
├── mathematical_foundations/    # Linear algebra, calculus, optimization
├── misc/                        # Anything else!
├── ml_algorithms/               # SVM, Decision Trees, kNN, etc.
├── mlops_pipeline/              # Experiment tracking, data pipelines, deployment
├── projects/                    # End-to-end mini-projects & Kaggle-style problems
├── research_summaries/          # Key ML papers and implementations
└── statistical_theory/          # Probability, distributions, Bayes, entropy, etc.


## 📅 Phase I Learning Plan – Core ML Topics (Ongoing)

This is a prioritized, balanced sequence of topics that I will work through weekly.  
✔️ Checkboxes = completed topics. Each topic should include:

- Concept review + notes
- A small notebook or script implementation
- Optionally, a blog summary on Medium

### 📘 Weeks 1–2: Fundamentals & Warmups
- [✔️] Linear Regression (theory + from scratch)
- [ ] Logistic Regression (sigmoid, loss function)
- [ ] Basic Probability & Bayes’ Theorem (refresher)
- [ ] Visualization of Decision Boundaries (with scikit-learn)

### 🧠 Weeks 3–4: Classical ML Algorithms I
- [ ] k-Nearest Neighbors (from scratch + sklearn)
- [ ] Naive Bayes (with real datasets)
- [ ] Decision Trees (entropy/gini calculation)
- [ ] Overfitting & Bias/Variance Tradeoff

### 🧠 Weeks 5–6: Classical ML Algorithms II
- [ ] Random Forests & Bagging (ensemble intuition)
- [ ] Support Vector Machines (linear & kernel trick)
- [ ] Dimensionality Reduction: PCA
- [ ] Model Evaluation: precision, recall, AUC, cross-val

### 🧬 Weeks 7–9: Deep Learning Foundations
- [ ] Neural Networks: Forward Propagation
- [ ] Backpropagation: Derivation + Code (NumPy)
- [ ] Activation Functions & Loss Functions
- [ ] Build 1-layer MLP from scratch
- [ ] PyTorch Basics: Tensors, Autograd
- [ ] Train a simple classifier with PyTorch (MNIST/FashionMNIST)

### 📊 Weeks 10–11: Statistics in Practice
- [ ] Likelihood & MLE
- [ ] Hypothesis Testing: Z-test, t-test
- [ ] Confidence Intervals & Error
- [ ] Distributions in ML: Normal, Bernoulli, etc.

### ⚙️ Weeks 12–13: MLOps Foundations
- [ ] Introduction to DVC or MLflow
- [ ] Pandas + Sklearn pipeline pattern (clean -> model -> evaluate)
- [ ] Deploy a model using FastAPI
- [ ] Save + load models and data versions

### 🧪 Weeks 14–16: Projects & Extensions
- [ ] Small ML project: tabular data problem (classification or regression)
- [ ] Kaggle mini-challenge or notebook replication
- [ ] Medium blog summary of one full pipeline/project
- [ ] First research paper summary (e.g., dropout, attention)

---

## 🧠 Phase II Learning Plan – Intermediate/Advanced ML Topics

This phase focuses on *end-to-end ownership* of machine learning systems. It extends foundational skills into production-grade implementation, experimentation, and architecture awareness—positioning me as a **full-stack ML engineer**.

📅 **Timeframe:** ~6 months (24–28 weeks)  
📦 **Focus:** Depth + Projectization + Production-Readiness  
🎯 **Goal:** Move beyond hireability → *negotiation-ready fluency*

---

### 🧩 Track 1: Advanced Modeling & Optimization (Weeks 1–5)
- [ ] Regularization (L1, L2, ElasticNet, early stopping)
- [ ] Advanced Ensembling (XGBoost, LightGBM, CatBoost)
- [ ] Bayesian Optimization for Hyperparameters (Optuna)
- [ ] Probabilistic Programming (PyMC3 or TensorFlow Probability)
- [ ] Calibration & Uncertainty (Brier Score, Confidence Curves)

🛠️ *Project Idea:*  
**“Model Garden”** – Build and compare a set of tuned models on a medium-sized dataset (e.g., housing prices, churn), using Optuna + SHAP for both tuning and explainability.

---

### ⚙️ Track 2: Advanced MLOps & Scalable Pipelines (Weeks 6–11)
- [ ] CI/CD for ML (GitHub Actions + Pytest)
- [ ] Dockerized Model Services + FastAPI
- [ ] MLflow Model Registry / DVC Pipelines
- [ ] Airflow or Prefect: Directed ML pipelines
- [ ] Data drift and model monitoring (EvidentlyAI or custom logging)

🛠️ *Project Idea:*  
**“Full-Stack ML Pipeline”** – Build a fully dockerized ML workflow with tracked data, model versioning, and a REST API endpoint. Include model monitoring (e.g., input schema validation, data drift).

---

### 📊 Track 3: Experimental Design & Causal Inference (Weeks 12–15)
- [ ] A/B/n Testing at scale (power, p-hacking mitigation)
- [ ] Multi-armed Bandits (Thompson Sampling, UCB)
- [ ] Uplift Modeling
- [ ] DAGs & Do-Calculus (Causal Inference Intro)

🛠️ *Project Idea:*  
**“Experiment Manager”** – Simulate multiple A/B scenarios with dashboards showing how false conclusions can arise. Deploy as a Streamlit app.

---

### 🧠 Track 4: Interpretability, Fairness & Ethics (Weeks 16–19)
- [ ] SHAP & LIME advanced usage
- [ ] Counterfactual explanations
- [ ] Model auditing for fairness (Fairlearn or AIF360)
- [ ] Explainability in NLP or vision

🛠️ *Project Idea:*  
**“ML Ethics Dashboard”** – Audit a model for bias and create a report or visual tool showing disparate impact, explanation breakdowns, and suggestions for mitigation.

---

### 🧰 Track 5: Specialized Tools & Modern Architectures (Weeks 20–24)
- [ ] Transformers (intro, Hugging Face use-case)
- [ ] Feature stores (Feast)
- [ ] Streaming data pipelines (Kafka + Spark intro)
- [ ] Time-series modeling (Prophet, Darts)
- [ ] Graph ML (node2vec, GNNs intro)

🛠️ *Project Idea:*  
**“Special Topics Sandbox”** – Choose 2 of the above (e.g., time-series forecasting + transformers for NLP) and build applied micro-projects with Jupyter-based demos.

---

## 🏗️ Project Portfolio Goals (by end of Phase II)

- [ ] ✅ 1–2 Real-world Full-Stack ML Projects (pipeline → deploy → monitor)
- [ ] ✅ 1 Dashboard or App (e.g., experiment explorer, fairness auditor)
- [ ] ✅ 1 Research Replication or Paper Summary
- [ ] ✅ Medium Posts / Notebooks for ~5–8 advanced topics
- [ ] ✅ Use of Docker, CI/CD, MLflow, and monitoring in at least one project

---

🚀 *Completion of this phase demonstrates mastery in modeling, system design, experimentation, and ethical deployment of ML solutions in a production context.*

