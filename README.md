# Customer Personality Analysis

Supervised and unsupervised machine learning on the [Customer Personality Analysis](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis) marketing dataset (2,240 customers). The notebook covers the full workflow: preprocessing, feature engineering, regression, classification, and customer segmentation.

## What the notebook does

1. **Preprocessing**: median imputation for missing income, removal of unrealistic records, ordinal and one-hot encoding
2. **Feature engineering**: age, total spending, and other derived customer features
3. **Regression** on total spending: Linear Regression, Ridge, Decision Tree
4. **Classification** of campaign response: Logistic Regression, K-Nearest Neighbours, Random Forest
5. **Clustering**: K-Means with the elbow method, PCA visualisation, and cluster profiling

## Results

### Regression (target: total spending)

| Model | RMSE | R² |
| --- | --- | --- |
| Linear Regression | 265.06 | 0.815 |
| Ridge Regression (alpha = 1) | 265.05 | 0.815 |
| **Decision Tree (max depth 5)** | **244.43** | **0.843** |

### Classification (target: campaign response)

| Model | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| **Logistic Regression** | 0.808 | 0.424 | 0.791 | **0.552** |
| K-Nearest Neighbours (k = 5) | 0.877 | 0.688 | 0.328 | 0.444 |
| Random Forest (100 trees) | 0.871 | 0.680 | 0.254 | 0.370 |

The target is imbalanced (about 15% responders), so F1 is the deciding metric rather than accuracy. Logistic Regression finds far more of the actual responders than the other two models.

## Run it

```bash
git clone https://github.com/eng-Ahmed2004/customer-personality-analysis.git
cd customer-personality-analysis
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook customer_personality_analysis.ipynb
```

The dataset (`marketing_campaign.csv`) is included in the repository. Run all cells from top to bottom.

## Tech stack

Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Jupyter Notebook

## Author

Ahmed Srour — [@eng-Ahmed2004](https://github.com/eng-Ahmed2004)
