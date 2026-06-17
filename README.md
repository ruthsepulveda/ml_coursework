# ML Coursework — Notebooks from DMLA & Desafío Latam

Applied machine learning notebooks developed during two concurrent programmes:
the **Diplomado en Machine Learning Aplicado (DMLA)** at PUC Chile and the
**Data Science certification** at Desafío Latam.

Each notebook addresses a real classification problem with full EDA, 
preprocessing, modelling, hyperparameter tuning, and result analysis.

**Author:** Ruth Sepúlveda  
**Focus:** Domain Data Scientist — Urban, Environmental, and Social Systems

---

## Notebooks

### 1. Coffee Quality Classification
**File:** [coffee_quality_ml.ipynb](coffee_quality_ml.ipynb)

**Question:** Can sensory scores and physical attributes predict specialty 
coffee quality class (Average / Good / Excellent)?

**Dataset:** Coffee Quality Institute (CQI May-2023) — 207 Arabica samples, 
41 features including sensory scores, altitude, processing method, and origin.

**Methods:** Exploratory data analysis, feature engineering, t-SNE and PCA 
visualisation, scikit-learn Pipelines with ColumnTransformer, Random Forest, 
Decision Tree, DummyClassifier baseline.

**Key results:**
- Random Forest: **95.2% accuracy, 0.91 balanced accuracy**
- Decision Tree: higher balanced accuracy (0.94) but less consistent across 
  classes
- Sensory scores (Aroma, Flavor, Acidity) are the strongest predictors; 
  physical attributes add moderate signal

**Skills demonstrated:** multiclass classification, imbalanced dataset handling, 
Pipeline design to prevent data leakage, feature selection with domain reasoning, 
t-SNE visualisation

---

### 2. Image & Tabular Classification: Fashion MNIST and Forest Cover Type
**File:** [fashion_mnist_forest_cover_classification.ipynb](fashion_mnist_forest_cover_classification.ipynb)

**Question:** How well can k-NN and ensemble models classify image and tabular 
data, and which features drive those decisions?

**Datasets:**
- Fashion MNIST — 70,000 greyscale images (28×28) across 10 clothing categories
- Forest Cover Type — cartographic variables predicting 7 tree species classes, 
  sourced from Scikit-learn (581,012 samples, stratified to 14,000)

**Methods:** k-NN with systematic hyperparameter search across distance metrics 
(Euclidean, Manhattan, Cosine), Logistic Regression (multinomial, cross-validated), 
Random Forest and Gradient Boosting with GridSearchCV and Stratified 5-Fold CV, 
PCA and t-SNE, feature importance via MDI and Permutation Importance.

**Key results:**
- Fashion MNIST: best k-NN (Manhattan, k=5, uniform) — **83.1% test accuracy**, 
  +1.9 points over baseline through metric tuning alone
- Forest Cover Type: Random Forest — **87.0% accuracy, macro-F1 0.868**, 
  outperforming Gradient Boosting (86.2%), k-NN (80.8%), and Logistic 
  Regression (70.0%)
- Elevation was the dominant feature across all three Forest Cover models 
  under both MDI and Permutation Importance, confirming the ecological 
  hypothesis formed during EDA

**Skills demonstrated:** multiclass classification, hyperparameter tuning, 
cross-validation methodology, feature importance interpretation, 
train/validation/test discipline, mixed-feature preprocessing with 
ColumnTransformer, dimensionality reduction for image data

---

## How to Run

```bash
git clone https://github.com/ruthsepulveda/ml_coursework
cd ml_coursework
pip install -r requirements.txt
jupyter lab
```

---

## References

- Blackard, J. A., & Dean, D. J. (1999). Comparative accuracies of artificial 
  neural networks and discriminant analysis in predicting forest cover types 
  from cartographic variables. *Computers and Electronics in Agriculture*, 
  24(3), 131–151.
- Coffee Quality Institute. (2023). *Coffee Quality Data (CQI May-2023)* 
  [Dataset]. Kaggle. https://www.kaggle.com/datasets/fatihb/coffee-quality-data-cqi
- Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. 
  *Journal of Machine Learning Research*, 12, 2825–2830.
- Xiao, H., Rasul, K., & Vollgraf, R. (2017). Fashion-MNIST: A novel image 
  dataset for benchmarking machine learning algorithms. 
  *arXiv preprint arXiv:1708.07747*.


---

## Stack

Python · pandas · scikit-learn · matplotlib · seaborn · Jupyter · kagglehub
