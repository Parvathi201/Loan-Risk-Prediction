# Phase 2: Exploratory Data Analysis (EDA)

In this phase, we dive deep into the raw data to uncover initial insights, patterns, and relationships. This pre-modeling EDA helps us understand the basic characteristics of the data and informs our feature engineering strategies. The primary output of this phase is a report and a set of visualizations.

## Key Files and Scripts

-   `scripts/eda.py`: This is the main script for the exploratory data analysis phase. It operates on the raw data from `data/raw/` to provide a first look at the dataset. Its key functions include:
    -   **Basic Information:** Generating a summary of the dataset, including data types, shape, and descriptive statistics.
    -   **Missing Value Analysis:** Identifying and reporting any missing values.
    -   **Distribution Analysis:** Creating histograms and box plots for numerical features to understand their distribution and identify potential outliers.
    -   **Categorical Analysis:** Generating bar charts to visualize the frequency of categorical features.
    -   **Correlation Analysis:** Calculating and visualizing the correlation between numerical features and the target variable (`default`).
    -   **Demographic Analysis:** Analyzing default rates across different demographic groups.

-   `reports/eda_report.md`: The script generates a markdown report summarizing its findings.
-   `reports/figures/`: Visualizations created by the script, such as demographic default rate plots, are saved in this directory.
