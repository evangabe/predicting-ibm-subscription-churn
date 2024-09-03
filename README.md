# Predicting IBM Subscription Churn

Data source: [Kaggle "Telco" Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

### Phase 1: Exploratory Data Analysis

- [x] Descriptive Analysis: duplicates, data types, possible conversions
- [X] Data Wrangling: missing values, binning, other conversions
- [X] Univariate Analysis: feature-by-feature statistics, frequency table, pie charts
- [T] Bivariate Analysis: mosaic plots (catxcat), spearman (numxnum), kendall (numxcat), biserial (boolxnum), phi (boolxbool), collinearity, stacked bar plots, distribution by target, box plots
- [T] Multivariate Analysis: frequency distribution, churn count distribution
- [T] Encode target and save cleaned dataset

### Phase 2: Models, Hyperparameters and Deployments

- [T] Prepare training and test datasets
- [T] Encode categorical features
- [T] Scale numerical features
- [T] Create model fit function with K-Fold on ROC-AUC scoring
    - CatBoost Model
    - XGBoost Model
    - LGBM Model
    - Stacking Ensemble Model
- [T] Plot feature importance

### Phase 3: Conclusions and Demo

- [ ] Train and save final model with best ROC-AUC score
- [ ] Identify strategies for reducing churn

- [T] Setup Streamlit application
- [ ] Create random customer generator (get image from [thispersondoesnotexist.com](https://thispersondoesnotexist.com))
- [ ] Show results on random customer by model
- [ ] Show ROC-AUC plot by model 