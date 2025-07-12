Here's a README file for your credit card fraud detection project:

-----

# Credit Card Fraud Detection

This project aims to detect fraudulent credit card transactions using various machine learning models. The dataset used contains highly imbalanced transaction data, where fraudulent transactions are a small minority.

-----

## Table of Contents

  - [Dataset](https://www.google.com/search?q=%23dataset)
  - [Exploratory Data Analysis (EDA)](https://www.google.com/search?q=%23exploratory-data-analysis-eda)
  - [Feature Selection](https://www.google.com/search?q=%23feature-selection)
  - [Machine Learning Models](https://www.google.com/search?q=%23machine-learning-models)
  - [Results](https://www.google.com/search?q=%23results)
  - [Dependencies](https://www.google.com/search?q=%23dependencies)
  - [Usage](https://www.google.com/search?q=%23usage)

-----

## Dataset

The dataset `creditcard.csv` contains credit card transaction data. Due to confidentiality, the original features (V1, V2, ..., V28) are transformed using PCA. The `Time` and `Amount` features are not transformed. The `Class` column is the target variable, indicating whether a transaction is fraudulent (1) or genuine (0).

  - **Features:** `Time`, `V1` - `V28` (PCA transformed features), `Amount`, `hour`, `day`
  - **Target:** `Class` (0 for normal, 1 for fraud)

The dataset exhibits a significant class imbalance:

  - **Normal Transactions:** 99.83%
  - **Fraud Transactions:** 0.17%

-----

## Exploratory Data Analysis (EDA)

The EDA section includes:

  - **Null Value Check:** A custom function `null_values(df)` is used to check for and display any missing values in the dataset. The dataset is complete with no null values.
  - **Dataset Overview:**
      - Displays the shape (rows and columns) of the dataset.
      - Provides a concise summary of the DataFrame, including data types and non-null values using `df.info()`.
      - Shows descriptive statistics for numerical columns using `df.describe()`.
  - **Feature Engineering:**
      - The `Time` feature is converted from seconds to hours and days to provide more interpretable temporal features.
  - **Class Distribution:**
      - Visualizations (pie chart and bar plot) show the severe imbalance between normal and fraudulent transactions.
      - The total amount transacted for both normal and fraud cases is calculated and displayed.
  - **Data Distribution Plots:**
      - **Histograms:** Visualizes the distribution of all features.
      - **Histograms by Class:** Shows the distribution of features separated by transaction `Class` (fraud vs. normal) to observe differences.
      - **Distplots:** Provides density plots for all features.
      - **Box Plots:** Identifies outliers in each numerical feature.
  - **Correlation Analysis:**
      - **Box Plots vs. Class:** Specific box plots show the correlation of features like `V17`, `V14`, `V12`, `V10` (negative correlation) and `V11`, `V4`, `V2`, `V19` (positive correlation) with the `Class` variable.
      - **Transaction Type Distribution:** Distplots for `Amount` and `Time` show their overall distribution.
      - **Fraud vs. Genuine Amount Distribution:** Separate distplots illustrate the `Amount` distribution for fraudulent and genuine transactions.
      - **Fraud vs. Genuine Time-Amount Scatter Plots:** Scatter plots of `Time` vs. `Amount` are shown for both fraud and genuine transactions to observe patterns.
      - **Hour vs. Amount (Fraud vs. Normal):** Line plots illustrate the average amount of fraud and normal transactions over different hours of the day.
      - **Heatmaps:** A correlation heatmap of all features is generated to show relationships.
      - **Target Correlation:** Displays features sorted by their absolute correlation with the `Class` variable.

-----

## Feature Selection

Based on the correlation analysis, the top 20 features most relevant to the `Class` variable (excluding `Class` itself and features with very low correlation) are selected for model training.

-----

## Machine Learning Models

Due to the highly imbalanced nature of the dataset, **SMOTE (Synthetic Minority Over-sampling Technique)** is applied to the training data to balance the classes.

The following classification models are trained and evaluated:

  - **Logistic Regression**
  - **K-Nearest Neighbors (KNN)**
  - **Support Vector Classifier (SVC)**
  - **Decision Tree Classifier**
  - **Random Forest Classifier**
  - **AdaBoost Classifier**

Each model's performance is assessed using a **classification report** (precision, recall, f1-score) and a **confusion matrix**.

-----

## Results

Among the evaluated models, **Random Forest Classifier** generally showed better performance. To further optimize it, **RandomizedSearchCV** was used for hyperparameter tuning.

### Initial Model Performance Comparison (F1-score for Fraud Class):

  - **Logistic Regression:** F1-score of 0.11
  - **K-Nearest Neighbors (KNN):** F1-score of 0.58
  - **Support Vector Classifier (SVC):** F1-score of 0.15
  - **Decision Tree Classifier:** F1-score of 0.50
  - **Random Forest Classifier:** F1-score of **0.88**
  - **AdaBoost Classifier:** F1-score of 0.11

### Optimized Random Forest Classifier:

Hyperparameter tuning with `RandomizedSearchCV` on the Random Forest model aimed to find the best combination of parameters. The optimized Random Forest Classifier achieved an F1-score of **0.88** for the fraud class.

-----

## Dependencies

The project relies on the following Python libraries:

  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `math`
  - `sklearn` (for various machine learning models, metrics, and data splitting)
  - `imblearn` (for SMOTE)

You can install these dependencies using pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn
```

-----

## Usage

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```
2.  **Ensure you have the dataset:** Place `creditcard.csv` in the appropriate input directory (e.g., `/kaggle/input/fraud-detection/` if running in a Kaggle environment, or adjust the path in the code).
3.  **Run the script:** Execute the Python script containing the code.
    ```bash
    python your_script_name.py
    ```

The script will perform data loading, EDA, feature engineering, model training, and evaluation, displaying the results in the console and as plots.
