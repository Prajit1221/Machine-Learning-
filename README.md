# Machine-Learning-
House price predection
# Boston House Price Prediction using Machine Learning

## Overview

This project delves into the task of predicting housing prices in the Boston metropolitan area, a classic problem in machine learning.  Leveraging the well-known Boston Housing Dataset, it employs a suite of regression-based machine learning models to uncover the intricate relationships between various housing features and property values.  The project aims to move beyond simple correlation analysis, building a robust and accurate model capable of estimating house prices for unseen data.  This enables valuable insights into the key factors influencing real estate prices in Boston, with potential applications in real estate analysis, investment strategies, and property valuation.  Ultimately, this project serves as a practical demonstration of applying machine learning techniques to a real-world problem with a focus on clear explanation and model evaluation.

## Dataset

The project utilizes the **Boston Housing Dataset**, a classic dataset in the field of machine learning. This dataset comprises information collected by the U.S. Census Service concerning housing in the area of Boston Mass. It contains **506 instances** and **13 features** that describe various aspects of the houses and their neighborhoods. The **target variable** is the **median value of owner-occupied homes (MEDV)**, expressed in thousands of U.S. dollars.

**Source:** The dataset is accessible through various Python libraries, including older versions of scikit-learn. Due to its deprecation in newer versions, alternative sources like the UCI Machine Learning Repository or Kaggle might be used.

## Features

The dataset includes the following features used for predicting house prices:

1.  **CRIM:** Per capita crime rate by town.
2.  **ZN:** Proportion of residential land zoned for lots over 25,000 sq.ft.
3.  **INDUS:** Proportion of non-retail business acres per town.
4.  **CHAS:** Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).
5.  **NOX:** Nitrogen oxides concentration (parts per 10 million).
6.  **RM:** Average number of rooms per dwelling.
7.  **AGE:** Proportion of owner-occupied units built prior to 1940.
8.  **DIS:** Weighted distances to five Boston employment centres.
9.  **RAD:** Index of accessibility to radial highways.
10. **TAX:** Full-value property-tax rate per \$10,000.
11. **PTRATIO:** Pupil-teacher ratio by town.
12. **B:** 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town.
13.  **LSTAT:** Percentage lower status of the population.

## Machine Learning Models

The project implements and evaluates the performance of the following regression-based machine learning models:

* **Linear Regression:** A fundamental linear model that aims to find a linear relationship between the features and the target variable.
* **Decision Tree Regression:** A non-linear model that partitions the feature space into rectangular regions, with a simple prediction rule within each region.
* **Random Forest Regression:** An ensemble learning method that constructs multiple decision trees during training and outputs the average prediction of the individual trees, reducing overfitting and generalization.
* **Gradient Boosting Regression:** Another ensemble learning technique that builds trees sequentially, with each new tree trying to correct the errors made by the previous ones, often leading to high predictive accuracy.

## Python Libraries

The project utilizes the following Python libraries for data manipulation, numerical computation, visualization, and machine learning:

* **pandas:** For data manipulation and analysis, particularly for working with DataFrames.
* **numpy:** For numerical computations, especially for handling arrays and matrices.
* **matplotlib:** For creating static, interactive, and animated visualizations in Python.
* **seaborn:** For making informative and attractive statistical graphics in Python.
* **scikit-learn (sklearn):** A comprehensive machine learning library in Python, providing tools for data preprocessing, model selection, training, and evaluation.

## Evaluation Metrics

The performance of the trained regression models is assessed using the following evaluation metrics:

* **Mean Absolute Error (MAE):** The average of the absolute differences between predicted and actual values.  MAE provides a measure of the magnitude of errors, regardless of their direction.
* **Mean Squared Error (MSE):** The average of the squared differences between predicted and actual values. MSE gives higher weight to larger errors, making it useful for penalizing significant deviations.
* **Root Mean Squared Error (RMSE):** The square root of the MSE. RMSE is expressed in the same units as the target variable, making it more interpretable than MSE.
* **R-squared (R²):** Represents the proportion of the variance in the dependent variable that is predictable from the independent variables. A value closer to 1 indicates a better fit of the model to the data.

## Visualization

To better understand the data and the model performance, the project employs the following visualization techniques:

* **Correlation Heatmap:** A heatmap is used to visualize the correlation matrix of the features, showing the linear relationships between them. This helps in identifying potential multicollinearity issues and understanding which features are most strongly related to the target variable (MEDV).
* **Feature Importance Plots:** For tree-based models (Decision Tree, Random Forest, Gradient Boosting), feature importance plots display the relative importance of each feature in predicting the house prices. This provides insights into which factors have the most significant influence on the model's predictions.
* **Actual vs. Predicted Price Graphs:** Scatter plots of actual house prices versus the model's predicted prices are used to visualize the model's accuracy. Points close to the diagonal line indicate accurate predictions, while deviations from the line represent prediction errors.
* **Residual Plots:** Residual plots show the difference between the actual and predicted prices (residuals) plotted against the predicted prices. These plots are used to diagnose the model's errors and check for patterns that may indicate violations of the assumptions of the regression models.

## Project Description

The Boston House Price Prediction project follows a standard machine learning workflow. It begins with loading and exploring the Boston Housing Dataset to gain insights into its characteristics and potential relationships between features. Data preprocessing steps, such as handling missing values (if any) and scaling features, are then applied to prepare the data for model training.

Several regression-based machine learning models, including Linear Regression, Decision Tree Regression, Random Forest Regression, and Gradient Boosting Regression, are trained on the preprocessed training data. The performance of each trained model is subsequently evaluated on a separate test dataset using appropriate regression metrics, namely Mean Squared Error (MSE) and R-squared (R²).

The project may also include visualizations to aid in understanding the data and the performance of the models. For instance, scatter plots of actual vs. predicted prices can provide a visual assessment of the model's accuracy, and feature importance plots (for tree-based models) can highlight which features have the most significant impact on the predicted house prices.

By comparing the evaluation metrics across different models, the project aims to identify the most suitable model for predicting house prices in the Boston area based on the given dataset. The findings and the trained model can potentially be used for real estate analysis and price estimation.

