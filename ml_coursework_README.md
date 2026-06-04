# ML Coursework — Desafío Latam & PUC Chile (2025–2026)

Exercises and notebooks from my Data Science training:
- **Data Science Bootcamp** — Desafío Latam (2025–2026)
- **Diplomado en Machine Learning Aplicado (DMLA)** — Pontificia Universidad Católica de Chile (2025–2026)

These notebooks document my learning progression and the reasoning behind each analytical decision. They are coursework exercises — not portfolio projects.

---

## Notebooks

| Notebook | Course | Topic | Key techniques | Result |
|----------|--------|-------|----------------|--------|
| [`coffee_quality_ml.ipynb`](coffee_quality_ml.ipynb) | DMLA · PUC Chile | Multiclass coffee quality classification | EDA · PCA + t-SNE · scikit-learn Pipelines · Random Forest · Decision Tree · overfitting analysis | RF balanced accuracy 0.91 · accuracy 0.95 |

### What this notebook covers

**Dataset:** Coffee Quality Institute (CQI) — 207 Arabica samples, 41 features (sensory scores, physical attributes, origin metadata).  
**Problem:** Classify coffee quality as Average / Good / Excellent from sensory scores and physical characteristics.

The notebook works through three phases:

**Phase 1 — EDA:** Full column audit with type and null analysis · five analytical charts including bivariate sensory score distributions per quality class and a correlation heatmap · altitude feature parsed from mixed-format text and reclassified into three elevation bands.

**Phase 2 — Feature engineering:** Data-leakage-safe column selection (Total Cup Points excluded as it directly generates the target) · dimensionality reduction comparison using PCA 2D vs PCA + t-SNE, showing that quality classes form partially separable clusters with Good/Excellent overlap.

**Phase 3 — Modelling:** Stratified 80/20 split · scikit-learn `Pipeline` with `ColumnTransformer` (separate numeric and categorical sub-pipelines, fitted only on train) · Random Forest and Decision Tree compared against a `DummyClassifier` baseline · confusion matrices · overfitting curve varying `max_depth` 1–30, identifying optimal generalisation at depth 4.

**Key finding:** Random Forest (n=100, class_weight='balanced') achieves balanced accuracy 0.91 and accuracy 0.95. Balanced accuracy is used as the primary metric because the dummy classifier achieves 0.64 accuracy simply by predicting the majority class — an important illustration of metric choice with imbalanced data.

---

## Main portfolio

My four main portfolio projects — built independently on Santiago urban data — are in separate pinned repositories:

| Project | Description | Repo |
|---------|-------------|------|
| Verde Desigual | Urban green space inequality EDA + Claude narrative | _coming soon_ |
| Aire Santiago | Air quality Streamlit dashboard + NL query interface | _coming soon_ |
| Huerta Urbana | Urban agriculture ML model + SHAP explainability | _coming soon_ |
| RAG Pago.cl | TF-IDF RAG pipeline + Claude API | _coming soon_ |

---

## Stack

Python · pandas · scikit-learn · matplotlib · seaborn · Jupyter · kagglehub
