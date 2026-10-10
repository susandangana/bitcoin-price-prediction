# Bitcoin Price Trend Prediction Using Machine Learning

## Project Overview

This project investigates the use of machine learning to classify Bitcoin price trends using historical cryptocurrency and financial market data. It combines exploratory data analysis (EDA), feature engineering, classification modelling, and model evaluation to identify patterns associated with upward price movements.

The project follows two main stages. First, multiple classification algorithms are compared using Bitcoin market data alone. Second, additional experiments investigate whether incorporating closing prices from Ethereum, gold, and the S&P 500 improves predictive performance.

The primary objective is to predict **price direction rather than the exact future closing price**. The project also examines how model selection and the inclusion of external market variables affect classification performance.

## Project Objectives

- Analyse historical Bitcoin prices and trading volumes.
- Explore relationships and patterns within financial market data.
- Engineer predictive features representing price movements, market trends, and volatility.
- Compare Logistic Regression and ensemble classification algorithms.
- Evaluate model performance using precision, accuracy, confusion matrices, and cross-validation.
- Investigate the additional predictive value of Ethereum, gold, and S&P 500 closing prices.
- Identify the models and feature sets that perform best under the evaluation procedures used.

## Tools and Technologies

- **Python** – Data analysis and machine learning.
- **Pandas** – Data manipulation and preprocessing.
- **NumPy** – Numerical computations.
- **Matplotlib** – Data visualisation.
- **Seaborn** – Statistical visualisation.
- **Scikit-learn** – Feature scaling, classification, cross-validation, and model evaluation.
- **Jupyter Notebook** – Interactive analysis and experimentation.
- **yfinance** – Financial market data retrieval where applicable.

## Project Workflow

### 1. Data Loading and Preprocessing

Historical Bitcoin market data is loaded from CSV files and organised chronologically. The preprocessing workflow includes converting date columns into datetime format, handling missing observations, and preparing the data for exploratory analysis and model development.

Additional financial datasets are incorporated into separate experiments to examine the relationships between Bitcoin and other assets.

### 2. Exploratory Data Analysis (EDA)

Exploratory analysis is used to understand Bitcoin's historical market behaviour and identify potentially useful predictive variables.

The analysis includes:

- **Price Trend Analysis:** Line charts showing historical Bitcoin closing prices.
- **Trading Volume Analysis:** Pie charts illustrating the annual distribution of trading volume.
- **Annual Price Comparison:** Bar charts comparing average closing prices across years.
- **Correlation Analysis:** Examination of relationships between predictive features and the target variable.
- **Cross-Asset Analysis:** Scatter plots and regression lines exploring relationships between Bitcoin prices and other financial assets.

These visualisations provide an initial understanding of market patterns and inform subsequent feature engineering and modelling decisions.

### 3. Feature Engineering

Additional features are created from historical Bitcoin market data to represent price movements, longer-term trends, and volatility.

The engineered features include:

| Feature | Description |
|---|---|
| `PriceChange` | Difference between the closing and opening prices. |
| `50days_MA` | 50-day moving average of the closing price. |
| `200days_MA` | 200-day moving average of the closing price. |
| `Price_vs_50days_MA` | Difference between the closing price and its 50-day moving average. |
| `Price_vs_200days_MA` | Difference between the closing price and its 200-day moving average. |
| `RollingStd` | Rolling standard deviation used to represent recent price volatility. |
| `UpperBand` | Upper volatility band constructed from a moving average and rolling standard deviation. |
| `LowerBand` | Lower volatility band constructed from a moving average and rolling standard deviation. |

The target variable, `PriceTrend`, represents a binary classification outcome indicating whether the following day's closing price is higher than the current day's opening price.

This formulation converts the problem into a classification task in which the models predict a directional outcome rather than a numerical price.

### 4. Feature Scaling and Selection

`StandardScaler` is used to standardise predictive features. A Random Forest classifier is initially trained to estimate feature importance and support the selection of a smaller set of variables for subsequent experiments.

The selected Bitcoin features include:

- Opening price (`Open`)
- Price change (`PriceChange`)
- 200-day moving average (`200days_MA`)
- Price relative to the 50-day moving average (`Price_vs_50days_MA`)
- Rolling standard deviation (`RollingStd`)

Additional experiments expand the feature set with closing prices from other financial assets to investigate whether external market information contributes to Bitcoin trend classification.

### 5. Machine Learning Models

Four principal classification approaches are evaluated.

**Logistic Regression**

Used as a baseline model to establish a reference point for comparing more complex classification approaches.

**Random Forest Classifier**

An ensemble of decision trees used to identify patterns in market data, estimate feature importance, and classify price trends.

**Bagging Classifier**

An ensemble approach used to investigate whether combining predictions from multiple estimators improves classification performance and stability.

**Gradient Boosting Classifier**

A boosting algorithm that builds an ensemble sequentially, with each stage attempting to improve on the preceding model's predictions.

The models are trained and evaluated using training and testing datasets. Cross-validation is also used to assess performance across multiple data splits.

### 6. Model Validation and Evaluation

The project uses several metrics to compare classification performance:

- **Accuracy:** The proportion of all predictions that are correct.
- **Precision:** The proportion of positive predictions that are correct.
- **Confusion Matrix:** A summary of correct and incorrect classifications across the target classes.
- **Cross-Validation Accuracy:** Accuracy measured across multiple validation folds.
- **Mean Cross-Validation Score:** The average performance across the validation folds.
- **Standard Deviation of Cross-Validation Scores:** A measure of variation in performance across the folds.

Precision is particularly relevant when evaluating predictions of upward price trends because it indicates how often positive classifications are correct. Accuracy provides a broader view of overall classification performance.

Using multiple metrics helps avoid relying on a single measure when comparing models.

## Model Comparison and Findings

### 7. Bitcoin-Only Model Comparison

The first experiment compares Logistic Regression, Random Forest, Bagging, and Gradient Boosting using Bitcoin market data alone.

The reported findings indicate that ensemble methods improve precision more noticeably than overall accuracy when compared with Logistic Regression.

The main observations are:

- **Gradient Boosting achieves the highest precision** among the compared models.
- Random Forest and Bagging produce relatively similar precision results.
- Random Forest has a slight advantage in accuracy.
- Precision improves by almost 9% relative to Logistic Regression in the reported comparison.

These results suggest that ensemble methods may be particularly useful when the objective is to improve the reliability of positive classifications, although overall accuracy gains are more modest.

The results are specific to the dataset and evaluation procedure used. They do not establish that the models will achieve the same performance on future market data.

### 8. Multi-Asset Market Analysis

The second experiment investigates whether incorporating information from other financial assets improves Bitcoin price trend classification.

The additional assets examined are:

- **Ethereum:** Another cryptocurrency with its own market dynamics.
- **Gold:** A traditional asset included to explore its relationship with Bitcoin.
- **S&P 500:** A broad equity-market indicator used to introduce wider financial market context.

The experiments compare feature sets incorporating these assets and use Gradient Boosting to evaluate their predictive contribution.

#### Ethereum

Ethereum closing prices provide the largest marginal improvement among the additional assets examined in the reported experiments.

This suggests that Ethereum market information may contain useful information for classifying Bitcoin price trends. The shared trading availability of cryptocurrency markets may be relevant to this result, although further testing is needed to establish why the improvement occurs.

#### Gold

Adding gold closing prices does not improve predictive performance in the reported experiment.

Within the tested feature set and evaluation procedure, gold therefore does not provide a measurable predictive benefit.

#### S&P 500

Adding the S&P 500 closing price produces a small additional improvement in the reported results.

This suggests that broader equity-market information may contribute some useful context, although the observed gain is limited.

#### Multi-Asset Analysis Summary

| Additional asset | Reported finding |
|---|---|
| Ethereum | Largest marginal improvement among the additional assets examined. |
| Gold | No improvement in the reported experiment. |
| S&P 500 | Small additional improvement. |

These findings demonstrate that adding more financial variables does not automatically improve a model. The usefulness of an additional feature depends on its relationship with the target, the data preparation process, and the evaluation methodology.

## Key Findings

The main findings from the reported experiments are:

1. Ensemble methods improve precision more noticeably than accuracy compared with Logistic Regression.
2. Gradient Boosting achieves the highest precision in the Bitcoin-only comparison.
3. Random Forest has a slight advantage in accuracy.
4. Precision improves by almost 9% relative to Logistic Regression in the reported comparison.
5. Ethereum closing prices provide the largest marginal improvement among the additional assets examined.
6. Gold closing prices do not improve predictive performance in the reported experiment.
7. S&P 500 closing prices provide a small additional improvement.
8. Comparing feature sets is important because additional financial data does not necessarily provide additional predictive value.

## Skills Demonstrated

- Exploratory Data Analysis (EDA)
- Financial Market Data Analysis
- Data Cleaning and Preprocessing
- Feature Engineering
- Moving Average and Volatility Indicators
- Machine Learning Classification
- Ensemble Learning
- Feature Importance Analysis
- Feature Scaling
- Cross-Validation
- Model Evaluation and Comparison
- Correlation Analysis
- Data Visualisation
- Python and Scikit-learn

## Project Requirements

The project uses Python and the libraries listed in `requirements.txt`.

Install the dependencies using:

```bash
python -m pip install -r requirements.txt
```

The analysis is developed in Jupyter Notebook. Ensure that the required CSV datasets are