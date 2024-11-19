# **Flight Price Prediction Using AWS SageMaker**

This project involves building a machine learning model to predict flight prices based on various features such as departure and arrival cities, dates, and airline data. The model is developed using **AWS SageMaker** to leverage scalable cloud-based machine learning infrastructure.

## **Project Overview**
The aim of this project is to predict the price of flights using machine learning algorithms. By analyzing historical flight data and applying feature engineering techniques, we train a model that can provide accurate price predictions.

## **Features of the Project**
- **Data Preprocessing**: Cleaned and processed flight data, including handling missing values, encoding categorical variables, and feature scaling.
- **Modeling**: Trained a regression model to predict flight prices using algorithms such as XGBoost, and deployed it using AWS SageMaker.
- **Evaluation**: Used metrics such as Mean Absolute Error (MAE) and Root Mean Square Error (RMSE) to evaluate model performance.
- **Deployment**: Deployed the model using AWS SageMaker's endpoint for real-time inference and predictions.

## **Tech Stack**
- **AWS SageMaker**: Model training, evaluation, and deployment.
- **Python**: Programming language used for data processing, feature engineering, and model development.
- **Pandas, NumPy**: For data manipulation and preprocessing.
- **Scikit-learn**: For machine learning algorithms and evaluation metrics.
- **XGBoost**: The main algorithm used for flight price prediction.
- **Matplotlib, Seaborn**: For data visualization.

## **Steps to Run the Project**
1. **Data Collection**: Collect historical flight data containing features such as departure and arrival cities, dates, and airlines.
2. **Data Preprocessing**: Clean and preprocess the data using Python libraries like `Pandas` and `Scikit-learn`.
   - Handle missing values.
   - Encode categorical variables.
   - Scale numerical features.
3. **Model Training**:
   - Split the data into training and testing sets.
   - Train a regression model (e.g., XGBoost) using AWS SageMaker.
4. **Model Evaluation**:
   - Evaluate the model using metrics like MAE and RMSE.
   - Fine-tune hyperparameters to improve accuracy.
5. **Model Deployment**:
   - Deploy the trained model on AWS SageMaker to create a real-time endpoint for flight price predictions.
6. **Prediction**:
   - Use the deployed model to make flight price predictions based on new input data.

## **Results**
The model was able to predict flight prices with reasonable accuracy. Key performance metrics:
- **Mean Absolute Error (MAE)**: 
- **Root Mean Squared Error (RMSE)**:

## **Challenges Faced**
- Handling missing values and noisy data.
- Feature engineering to improve model performance.
- Optimizing model performance through hyperparameter tuning.

## **Future Work**
- Experiment with additional algorithms such as Random Forest or Neural Networks.
- Use external data (weather, holidays) to enhance predictions.
- Improve model performance with advanced feature engineering techniques.

## **Conclusion**
This project demonstrates how AWS SageMaker can be used to build and deploy a machine learning model for predicting flight prices. The final model helps users estimate flight prices accurately based on various features, providing valuable insights for future travelers.

