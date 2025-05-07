# Predicting Insurance Claim Amount

This project is a complete machine learning pipeline to predict individual insurance claim amounts using demographic and lifestyle features such as age, sex, BMI, smoking status, and region. The model leverages regression algorithms to estimate the `charges` column from the dataset.

# Dataset

The dataset used is `insurance.csv`, which contains the following columns:

- `age`: Age of the individual
- `sex`: Gender (`male`/`female`)
- `bmi`: Body Mass Index
- `children`: Number of children
- `smoker`: Smoking status (`yes`/`no`)
- `region`: Residential region in the US
- `charges`: Medical insurance cost (target variable)

# Project Workflow

The project follows these steps:

1. **Import Libraries** – Load necessary Python libraries for data processing, visualization, and machine learning.
2. **Load Dataset** – Read the `insurance.csv` file.
3. **Data Inspection** – Understand the dataset structure using `.info()` and `.describe()`.
4. **Missing Value Check** – Verify the absence of missing values.
5. **Data Encoding** – Convert categorical variables (`sex`, `smoker`, `region`) into numeric format using Label Encoding.
6. **Data Visualization** – Explore the data with pie charts, count plots, and heatmaps for correlation:
   - Distribution of `sex`, `smoker`, and `region`
   - Correlation matrix
7. **Train-Test Split** – Split the dataset into training and testing subsets.
8. **Feature Scaling** – Apply standardization to numerical features.
9. **Model Training** – Train a **Linear Regression** model.
10. **Model Evaluation** – Evaluate the model using Mean Squared Error and R² Score.
11. **Prediction Visualization** – Plot actual vs predicted charges.

# Algorithms Used

- Linear Regression

# Libraries Used

- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `sklearn` (scikit-learn)

