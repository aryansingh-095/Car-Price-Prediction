# Car-Price-Prediction
A machine learning project to predict the selling price of used cars based on their features. This model uses a Random Forest Regressor to analyze various attributes of a car and estimate its market value.
Project Overview

The goal of this project is to build a regression model that can accurately predict the selling price of a used car. By training on a dataset of previously sold cars, the model learns the relationships between a car's features (like its age, mileage, and fuel type) and its final selling price. This can be a valuable tool for both buyers and sellers in the used car market.

Technologies Used - 

Python: The core programming language.

Pandas: For data manipulation and analysis.

NumPy: For numerical operations.

Scikit-learn: For building and evaluating the machine learning model.

Matplotlib / Seaborn (Optional): For data visualization.

Key Features - 

Data Cleaning and Preprocessing: Handles categorical data and text-based features.

Feature Engineering: Creates a new, more insightful car_age feature from the manufacturing year.

Robust Model: Uses a Random Forest Regressor, an ensemble model known for its high accuracy and resistance to overfitting.

Model Evaluation: Provides clear metrics like R-squared and Root Mean Squared Error (RMSE) to assess performance.


Dataset - 


This project uses the car_data.csv dataset, which contains information about used cars.

File: car_data.csv

Rows: 4000+

Columns:

name: The brand and model of the car.

year: The year the car was manufactured.

selling_price: (Target Variable) The price the car was sold for.

km_driven: The total kilometers driven by the car.

fuel: The type of fuel the car uses (e.g., Petrol, Diesel).

seller_type: The type of seller (e.g., Individual, Dealer).

transmission: The transmission type (e.g., Manual, Automatic).

owner: The number of previous owners (e.g., First Owner, Second Owner).


Methodology - 


The project follows a standard machine learning workflow:

Data Loading:-

The car_data.csv file is loaded into a pandas DataFrame.

Preprocessing:-

A new feature, car_age, is engineered from the year column.

The owner column is converted from text (e.g., "First Owner") to numerical values (e.g., 0).

Categorical features like fuel, seller_type, and transmission are converted into numerical format using one-hot encoding.

Unnecessary columns (name, year) are dropped.

Model Training:- 

The dataset is split into an 80% training set and a 20% testing set.

A RandomForestRegressor model is trained on the training data.

Model Evaluation: The model's performance is evaluated on the unseen test data using standard regression metrics.

Prediction: The trained model is used to predict the price of a new, sample car.
