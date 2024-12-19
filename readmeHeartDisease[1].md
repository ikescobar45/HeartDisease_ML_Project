# Predicting Heart Disease Risk Using Machine Learning

## Isaac Escobar

### Executive Summary

#### Overview and Goals
This project focuses on using machine learning to predict heart disease based on medical data. By training and evaluating four models (Logistic Regression, Support Vector Machine, Random Forest, and Gradient Boosting), the goal is to identify which model performs the best in classifying individuals as having heart disease or not. Additionally, the project analyzes feature importance to determine the factors that contribute most to accurate predictions.

#### Why It Matters (Rationale)
Heart disease is a leading cause of illness and death worldwide. Early detection can save lives by enabling timely treatment and lifestyle changes. This project demonstrates how machine learning can assist in identifying individuals at risk, offering a tool that complements traditional diagnostic methods.

#### Research Questions
Which machine learning model is most effective at predicting heart disease?
What are the most important features (e.g., age, blood pressure) in making accurate predictions?

#### Data 
Dataset:
The data was sourced from a publicly available medical dataset containing patient information such as age, blood pressure, cholesterol levels, and exercise-related heart metrics. The link is below and the data will be in this repository.

https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction/data

Contains information on Age, Gender, Resting Blood Pressure (RestingBP)
Cholesterol, Maximum Heart Rate (MaxHR)
Exercise Angina, Oldpeak (ST depression), and ST Slope

The target variable, HeartDisease, indicates the presence (1) or absence (0) of heart disease.

#### Methodology
1. Data Cleaning and Preprocessing:

-Handle missing or redundant columns (if necessary).
-Encode categorical variables such as Chest Pain Type and ST Slope using techniques like one-hot encoding or label encoding.
-Normalize numerical features such as RestingBP, Cholesterol, and MaxHR for better model performance.

2. Exploratory Data Analysis (EDA):

-Generate summary statistics to understand feature distributions.
-Create visualizations (e.g., histograms, boxplots, and correlation heatmaps) to explore relationships between features and the target variable.
-Identify key trends and patterns in the dataset.

3. Feature Engineering:

-Investigate and retain the most relevant features for predicting heart disease.


4. Model Development:

-Train multiple machine learning models, including Logistic Regression, Support Vector Machines, Random Forest, and Gradient Boosting.
-Use GridSearchCV for hyperparameter tuning to optimize each model.

5. Evaluation Metrics:

-Evaluate models using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.

-Use confusion matrices to analyze false positive and false negative rates.


#### Data and Cleaning Results

No missing values were found initially which was amazing. However, further analysis showed that restingBP, cholesterol, and oldpeak had unrealistic values. 
Zero values in RestingBP and Cholesterol were replaced with NaN since they were deemed invalid.
Missing values in numerical columns (RestingBP, Cholesterol, Oldpeak) were imputed using the mean of each column.
Outliers were removed based on domain-specific thresholds:
RestingBP: Valid range set between 50 and 200 mmHg.
Cholesterol: Retained values between 100 and 500 mg/dL.
Oldpeak: Negative values were removed, as they are invalid for this feature.
The categorical columns had no missing values and one was one- hot encoded to convert them into numeric machine learning format

FastingBS was dropped due to lack of variance. Every entry contained the value 1. ST_Slope_str and ChestPainType_str were dropped because they ended up being redudant on graphs after these features were one- hot encoded.

#### Exploratory Data Analysis Results


```python
from IPython.display import Image, display

# Display the saved correlation heatmap
display(Image(filename='correlation_heatmapHeartDisease.png'))
```


    
![png](output_11_0.png)
    


1. Correlation Analysis
Key predictors of heart disease include ST_Slope (-0.56), ExerciseAngina (0.50), and MaxHR (-0.39). Age (0.29) and Oldpeak (0.27) show weaker but notable relationships. Cholesterol and RestingBP display minimal correlation, making them less important for prediction.

2. Numerical Feature Distributions
Most individuals in the dataset are aged 50-60, aligning with the higher risk of heart disease in older populations. Resting blood pressure is typically within 120-140 mm Hg, and cholesterol peaks around 250 mg/dL but shows little connection to heart disease. Maximum heart rates are generally 130-160 bpm, with lower rates linked to higher risk. Oldpeak values, representing ST depression during exercise, cluster around 0-2, with higher values associated with heart problems.

3. Boxplots for Numerical Features
Boxplots confirm that people with heart disease tend to be older, while cholesterol and resting blood pressure have similar distributions across both groups, suggesting they are weaker predictors.

4. Proportions and Bar Plots for Categorical Variables
Asymptomatic chest pain (Type 0) and flat ST slopes are associated with the highest proportion and count of heart disease cases. Proportion plots highlight relative risks, while bar plots show total cases, together providing a complete understanding of categorical features.

### Model Developement

### Evaluation Metrics

### Model FInding and Results

### Future Works and Reccomendations

### Final Summary

Per my learning facilitator Jessica Cervi, this initial research part of the capstone is only graded based on EDA and data cleaning. My model results will be in my final report.

# Heart Disease Project

You can download the following files related to this proct: 



[Download HeartDisease_datacleaning.ipynb](./HeartDisease_datacleaning.ipynb)
 pynb)
- [Download HeartDisease_EDA.ipynb](./HeartDisease_EDA.ipynb)





```python

```
