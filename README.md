# SmartML — Profile-Driven AutoML with Meta-Learning

SmartML is a preprocessing-aware AutoML framework that uses dataset profiling to drive intelligent pipeline decisions. Instead of brute-forcing every combination, SmartML analyzes each dataset's characteristics (size, imbalance, missing values, skewness, outliers, multicollinearity) and selects appropriate preprocessing, models, and tuning strategies accordingly.

The project includes a **meta-learning extension (Meta-SmartML)** that trains a GradientBoosting meta-learner on 71 OpenML dataset profiles to predict the winning algorithm family for unseen datasets, reducing the search space before full tuning — achieving the same win rate as the full pipeline at ~3.4× faster runtime.

---

## Key Contributions

1. **CV-Validated Permutation Importance Feature Gating** — Features are ranked via permutation importance on a held-out fold, and low-importance features are pruned before model selection.

2. **IsolationForest + LOF Ensemble Outlier Detection** — Two detectors must agree before a point is flagged. Includes a minority-class protection mechanism that falls back to IQR clipping if removal would destroy >10% of the minority class.

3. **Wilcoxon Signed-Rank Model Elimination** — After quick evaluation, models that are statistically indistinguishable from the worst performer (via Wilcoxon test, α=0.05) are pruned, saving tuning budget for promising candidates.

4. **Disagreement-Based Ensemble Diversity** — Ensemble members are selected by maximizing prediction disagreement (via `cross_val_predict`), not just individual accuracy.

5. **Budget-Aware Hyperparameter Tuning** — A fixed 120-second budget is allocated proportionally by model rank, so top-performing models get more tuning time.

---

## Pipeline Architecture

```
Dataset → Profiler → Preprocessing → Feature Engineering → Model Selection
              ↓              ↓                ↓                   ↓
         Report Card    Imputation,      Permutation         13 Models
         (size, task,   Outlier IF+LOF,  Importance,         (sklearn +
          imbalance,    Encoding,        Scaler Probe,       LightGBM,
          missing%,     Duplicate        MI Selection,       XGBoost,
          skewness...)  Removal          PCA/LDA             CatBoost)
                                                              ↓
                                                    Quick Eval (5-fold CV)
                                                              ↓
                                                    Wilcoxon Elimination
                                                              ↓
                                                    Budget-Aware Tuning
                                                              ↓
                                                    Disagreement Ensemble
                                                              ↓
                                                         Final Score
```

---

## Notebook Structure

| Cell | Phase | Description |
|------|-------|-------------|
| 1 | — | Imports and global config |
| 2 | Phase 0 | Dataset Profiler — generates report card |
| 3 | Phase 1A | Smart Preprocessing (imputation, IF+LOF outliers, encoding) |
| 4 | Phase 1B | Feature Engineering (permutation importance, scaler probe, MI, PCA/LDA) |
| 5 | Phase 2 | Model Selection — 13 models (10 sklearn + 3 boosting) |
| 6 | Phase 3 | Evaluation + Wilcoxon elimination + budget-aware tuning |
| 7 | Phase 4 | Smart Ensemble with disagreement-based diversity |
| 8 | — | `run_smartml()` one-call wrapper |
| 9 | — | Dataset definitions + PyCaret benchmark runner |
| 10 | — | Ablation study (contribution toggle) |
| 11 | — | SHAP explainability |
| 12 | — | Wilcoxon significance test (SmartML vs PyCaret) |
| 13 | — | Energy tracking with CodeCarbon |
| 14A | — | Meta-learning dataset collection (71 OpenML datasets) |
| 14B | — | Meta-learner training + evaluation |
| 15 | — | Meta-SmartML vs Full SmartML vs PyCaret comparison |
| 16 | — | Full 20-dataset benchmark |

---

## Results Summary

**Benchmark**: 12 classification + 8 regression datasets (20 total)

| Comparison | Win Rate |
|------------|----------|
| Meta-SmartML vs PyCaret | 9/12 (75%) |
| Meta-SmartML vs Full SmartML | Non-degrading (identical accuracy) |
| Average speedup (Meta vs Full) | ~3.4× |

Meta-SmartML matches Full SmartML's accuracy while skipping irrelevant model families, yielding significant runtime savings.

---

## Tech Stack

- **Python 3.11+**
- **Core ML**: scikit-learn, imbalanced-learn, scipy
- **Boosting**: LightGBM, XGBoost, CatBoost
- **Explainability**: SHAP
- **Energy**: CodeCarbon
- **AutoML Baseline**: PyCaret (requires separate venv)
- **Meta-Learning Data**: OpenML

---

## Quick Start

### 1. Install Dependencies

```bash
pip install scikit-learn pandas numpy scipy matplotlib seaborn
pip install lightgbm xgboost catboost
pip install imbalanced-learn shap codecarbon openml
```

### 2. Run SmartML on Any Dataset

```python
# Load the notebook cells 1-8, then:
import pandas as pd

df = pd.read_csv("your_dataset.csv")
result = run_smartml(df, target_column="target")

print(f"Best score: {result['best_score']:.4f}")
print(f"Best model: {result['best_model_name']}")
```

### 3. PyCaret Baseline (Optional)

PyCaret requires a separate virtual environment due to dependency conflicts:

```bash
python -m venv pycaret_env
source pycaret_env/bin/activate  # Linux/Mac
pip install pycaret
```

Run the PyCaret benchmark script separately and save results to `pycaret_results.json`.

---

## Datasets Used

**Classification (12):** Breast Cancer, Iris, Wine, Diabetes (Pima), Heart Disease, Titanic, Credit Card Fraud, Customer Churn, Mushroom, Glass, Ionosphere, Sonar

**Regression (8):** Boston Housing, California Housing, Diabetes (sklearn), Auto MPG, Concrete Strength, Energy Efficiency, Airfoil Self-Noise, Wine Quality

---

## Authors

- **Aakash** — Design, implementation, benchmarking
- **Aman Behera** — Project collaborator

Manipal University Jaipur — CSE3201 Machine Learning

---

## License

This project is for academic and research purposes.
