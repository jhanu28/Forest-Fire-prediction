# Forest Fire Predictor  

## About the Project  

This project aims to predict forest fires using machine learning techniques based on weather conditions. The dataset used is the **Algerian Forest Fires** dataset from UCI, which contains fire observations from two regions in Algeria: Bejaia and Sidi Bel-Abbes. By analyzing weather conditions such as temperature, humidity, and wind speed, the model can determine the likelihood of a fire occurring.  

The dataset is stored in **MongoDB** for efficient handling, and the application is built using **Flask** for easy deployment. Various machine learning models are trained and evaluated using **Scikit-Learn** to identify the best-performing model for both classification and regression tasks. Data processing and visualization are performed using **Pandas, NumPy, and Matplotlib** to extract insights and improve the model’s accuracy.  

## Dataset and Data Processing  

The dataset contains forest fire records from June 2012 to September 2012. It includes multiple weather parameters, which help in determining fire risks. The first step in the project involves downloading and loading the dataset into a **Pandas DataFrame**. The data is then cleaned, processed, and stored in **MongoDB** using **PyMongo** to enable efficient retrieval and manipulation.  

Since machine learning models require structured and well-processed data, various preprocessing steps such as handling missing values, feature selection, and normalization are performed.  

## Exploratory Data Analysis (EDA)  

Before building predictive models, it is essential to analyze the dataset to understand patterns and relationships between features. **Exploratory Data Analysis (EDA)** is conducted using **Pandas and Matplotlib** to visualize trends and distributions in the data. Important features contributing to forest fire predictions are identified, and correlations between variables are analyzed. This helps in selecting the most relevant features for model training and improving overall prediction accuracy.  

## Model Building and Training  

For predicting forest fires, both **classification and regression models** are built. The classification task involves determining whether a fire will occur based on weather conditions, treating it as a **binary classification problem**. The regression task focuses on predicting the **Fire Weather Index (FWI)**, which is highly correlated with fire occurrence.  

Multiple machine learning algorithms are used, including:  
- **Classification Models:** Logistic Regression, Decision Trees, Random Forest, XGBoost, K-Nearest Neighbors.  
- **Regression Models:** Linear Regression, Support Vector Regression (SVR), Random Forest Regressor.  

The models are evaluated using performance metrics such as **accuracy for classification** and **R² score for regression**. **Hyperparameter tuning** is performed using **GridSearchCV** to optimize model performance.  

## Flask Application and Deployment  

A **Flask** web application is created to allow users to input weather parameters and receive fire predictions. The application consists of multiple routes, including:  
- A default homepage  
- An API endpoint for testing  
- Specific routes for classification and regression predictions  

Flask is chosen for its simplicity and flexibility in building web applications with machine learning models.  

## Technologies and Tools Used  

- **Programming Language:** Python  
- **Machine Learning:** Scikit-Learn  
- **Web Framework:** Flask  
- **Database:** MongoDB (PyMongo)  
- **Data Processing:** Pandas, NumPy  
- **Visualization:** Matplotlib, Seaborn  
- **Deployment:** Heroku  
- **Version Control:** Git, GitHub  

This project demonstrates how machine learning can be applied to real-world problems such as **forest fire prediction**. By leveraging data science techniques, weather conditions can be analyzed to provide early warnings, potentially reducing fire damage and improving safety measures.  
