🚢 Titanic Survival Prediction (Logistic Regression)
📘 Project Overview

This project analyzes the Titanic dataset to predict passenger survival using Logistic Regression — one of the most fundamental classification algorithms in machine learning.
It combines data exploration, visualization, and predictive modeling to understand which factors influenced survival during the tragic event of the RMS Titanic.

🎯 Objectives

Explore the Titanic dataset to understand key patterns and relationships.

Perform Exploratory Data Analysis (EDA) using Seaborn and Matplotlib.

Preprocess data (handle missing values, encode categorical features).

Build and evaluate a Logistic Regression model for binary classification (Survived / Not Survived).

Interpret model coefficients to understand feature importance.

📊 Exploratory Data Analysis (EDA)

EDA was performed to visualize and interpret the data:

Survival Count: Overall survivors vs. non-survivors.

Sex vs. Survival: Women had a much higher survival rate.

Class vs. Survival: First-class passengers were more likely to survive.

Age Distribution: Younger passengers had slightly higher survival chances.

Embarked vs. Survival: Passengers boarding from Cherbourg had better odds.

Fare Distribution: Higher fares correlated with higher survival probability.

Correlation Heatmap: Reveals relationships between numeric features.

Sample EDA plots include:

Countplots (Survival by Gender, Class, Embarkation)

Histogram of Age vs. Survival

Fare boxplots across Passenger Classes

Correlation heatmap

⚙️ Model Building

Dataset: sns.load_dataset("titanic") (Seaborn built-in dataset)

Preprocessing Steps:

Dropped missing values

Encoded categorical variables using pd.get_dummies()

Split dataset into training (80%) and testing (20%) sets

Model: LogisticRegression() (from sklearn.linear_model)

Evaluation Metrics:

Accuracy Score

Confusion Matrix

Classification Report (Precision, Recall, F1-score)

📈 Results
Metric	Value (approx.)
Accuracy	~78–82%
Precision	~0.80
Recall	~0.78
F1-score	~0.79

Interpretation:

The model performs reasonably well given simple preprocessing.

Gender, passenger class, and fare were key predictors of survival.

💡 Key Insights

Gender is the strongest predictor of survival (females survived more).

Passenger class and fare also strongly influence survival chances.

Younger and wealthier passengers had higher survival odds.

🧠 Technologies Used

Python

Pandas, NumPy – data manipulation

Seaborn, Matplotlib – visualization

Scikit-learn – machine learning (Logistic Regression)

Jupyter Notebook – interactive development

🚀 How to Run

Clone this repository

git clone https://github.com/yourusername/titanic-logistic-regression.git
cd titanic-logistic-regression


Install dependencies

pip install pandas numpy seaborn matplotlib scikit-learn


Run the notebook or script

jupyter notebook titanic_logistic_regression.ipynb


or

python titanic_logistic_regression.py

🏁 Conclusion

The Titanic project demonstrates the use of logistic regression for binary classification on a classic dataset. Through careful EDA, preprocessing, and model interpretation, it highlights how simple statistical models can still yield meaningful insights.
