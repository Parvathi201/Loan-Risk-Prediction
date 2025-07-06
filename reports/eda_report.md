# Exploratory Data Analysis Report
Generated on: 2025-07-06 10:45:42

## Dataset Overview
- Number of observations: 5000
- Number of features: 25

## Missing Values
The following columns contain missing values:
|           |   Missing Values |   Percentage |
|:----------|-----------------:|-------------:|
| age_group |              261 |         5.22 |

## Target Variable Analysis
### Class Distribution
|   default |   proportion |
|----------:|-------------:|
|         1 |        67.22 |
|         0 |        32.78 |

### Correlation with Target
Top 10 features most correlated with the target variable:
|                            |      default |
|:---------------------------|-------------:|
| default                    |  1           |
| delinquencies_2yrs         |  0.0247689   |
| application_date_month     |  0.0100837   |
| loan_amount_term           |  0.00968891  |
| dependents                 |  0.00459384  |
| open_accounts              | -0.000338438 |
| debt_to_income_ratio       | -0.00724251  |
| application_date_day       | -0.011776    |
| total_accounts             | -0.013618    |
| application_date_dayofweek | -0.0142372   |