# House-Price-Prediction-in-Tehran-Using-Divar-Data-and-RandomForest-with-WandB-for-Model-Tracking
A machine learning project to predict house prices in Tehran using data from Divar. The project uses RandomForestRegressor for prediction, WandB for tracking the training process, and XGBoost for comparison. The model is evaluated using MSE and RMSE, and a Web app is created for user interaction.
# House Price Prediction in Tehran Using Divar Data and RandomForest with WandB for Model Tracking

This project demonstrates how to predict house prices in Tehran using data scraped from the Divar website. The project uses the **RandomForestRegressor** model for house price prediction, **WandB** for model tracking, and **XGBoost** for comparison. It also includes a Web App for user interaction.

## Project Description

In this project, the following steps are applied:

1. **Importing Libraries**:
   - Necessary libraries for data processing, model building, and tracking are imported:
     - **WandB** for tracking the training process.
     - **RandomForestRegressor** from **Scikit-learn** for building the regression model.
     - **XGBoost** for comparison.
     - **Pandas**, **NumPy** for data manipulation.
     - **Matplotlib**, **Seaborn** for visualization.
   
2. **Data Loading and Exploration**:
   - The data from Divar is loaded using **Pandas**.
   - The first few rows of the dataset are displayed using `head()` to check the data.
   - Data information is reviewed using `info()` to examine types and non-null values.

3. **Data Preprocessing**:
   - **Handling Missing Values**: Missing values are handled by replacing them with the median for numeric columns.
   - **Feature Engineering**: Boolean columns like `Parking`, `Warehouse`, and `Elevator` are converted to numerical values (1 for True, 0 for False).
   - **One-Hot Encoding**: Categorical variables like `country` are converted to numeric using one-hot encoding.
   
4. **Feature Selection**:
   - The features (`Area`, `Room`, `Parking`, `Warehouse`, `Elevator`) are selected for predicting the house price (`total_price`).
   - The dataset is split into training (80%) and testing (20%) sets using **train_test_split**.

5. **Model Training with Random Forest**:
   - The **RandomForestRegressor** model is built using the parameters set by **WandB**.
   - The model is trained and tracked using **WandB**.

6. **Model Evaluation**:
   - The model is evaluated using **Mean Squared Error (MSE)** and **Root Mean Squared Error (RMSE)**, which are logged in **WandB**.

7. **Model Comparison with XGBoost**:
   - The **XGBoost** model is trained and compared with the Random Forest model to check which model performs better.
   
8. **Saving the Model**:
   - The final trained model is saved using **WandB** and can be loaded later for future predictions.

9. **Web App Creation**:
   - An interactive Web App is created to allow users to input new data and predict house prices in Tehran using the trained model.

## Libraries Used
- **WandB**
- **Scikit-learn** (RandomForestRegressor)
- **XGBoost**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**
1. **Clone the repository**:
2. 2. **Install required libraries**
3. **Run the code**:
Open the `project1_(2).ipynb` file in Jupyter Notebook and follow the steps for data preprocessing, model training, and evaluation.

4. **Track with WandB**:
Log in to **WandB** to track the training process, log metrics, and visualize results.

5. **Web App**:
To run the interactive Web App, use the following command in your terminal:
