# Stock Market Prediction using LSTM

## Project Overview
This project aims to predict stock prices using **Long Short-Term Memory (LSTM)** networks, a type of Recurrent Neural Network (RNN) suited for time series forecasting. The goal is to predict future stock prices based on historical data to help investors make informed decisions.

## Table of Contents
- [Problem Statement](#problem-statement)
- [Solution Approach](#solution-approach)
- [Methodology](#methodology)
- [Results](#results)
- [Analysis](#analysis)
- [Conclusion](#conclusion)
- [Libraries Used](#libraries-used)
- [References](#references)

## Problem Statement
The stock market is highly volatile, and predicting future prices accurately is a challenging task. Stock price prediction can assist investors and traders in making better financial decisions. This project focuses on predicting the **closing price** of a stock using past price data with the help of LSTM, which is a powerful tool for time series forecasting.

## Solution Approach
The problem is approached by:
1. **Data Collection**: Using a stock price dataset from **Tiingo API**.
2. **Data Preprocessing**: Cleaning and scaling the data.
3. **Model Development**: Training an LSTM model on historical stock prices.
4. **Evaluation**: Evaluating the model's performance using metrics like RMSE, MAE, and R-squared.
5. **Prediction**: Predicting future stock prices.

## Methodology
1. **Data Acquisition**: Stock price data is fetched using the Tiingo API, which provides historical stock prices.
2. **Data Preprocessing**: The data is scaled using MinMaxScaler to normalize the values and make them suitable for model input.
3. **Model Building**: A deep learning model using LSTM is built to capture patterns in the time series data.
4. **Training**: The LSTM model is trained using historical stock prices.
5. **Evaluation**: The model is evaluated based on RMSE, MAE, and R-squared scores to assess its performance.
6. **Prediction**: After training, the model is used to predict future stock prices.

## Results
The LSTM model performs well in predicting stock prices with acceptable error metrics:
- **RMSE**: Root Mean Square Error of the predicted vs actual prices.
- **MAE**: Mean Absolute Error between the predicted and true values.
- **R-squared**: Measures the proportion of variance explained by the model.

## Analysis
- The model's predictions are compared with actual stock prices to analyze its accuracy.
- Residual analysis is conducted to evaluate the difference between predicted and actual values.
- The stock price prediction’s success is measured in terms of error metrics, and the model’s generalization capability is also assessed.

## Conclusion
The project demonstrates the feasibility of using LSTM for stock price prediction. The model provides a good prediction of stock prices, though stock market data is inherently noisy and volatile. While the model performs well, further improvements and fine-tuning can enhance its predictive capabilities.

## Libraries Used
This project utilizes several libraries for data manipulation, modeling, and evaluation:
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations.
- **Matplotlib** & **Seaborn**: For data visualization.
- **Scikit-learn**: For data preprocessing and evaluation metrics.
- **TensorFlow/Keras**: For building and training the LSTM model.
- **Requests**: For fetching stock data via API.

## References
- **Tiingo API**: [https://www.tiingo.com/](https://www.tiingo.com/)
- **Kaggle**: [https://www.kaggle.com/](https://www.kaggle.com/)
- **LSTM for Time Series Forecasting**: [https://machinelearningmastery.com/long-short-term-memory-networks-for-machine-learning/](https://machinelearningmastery.com/long-short-term-memory-networks-for-machine-learning/)
- **Stock Market Prediction using LSTM**: [https://towardsdatascience.com/stock-price-prediction-using-lstm-9a7e9d315e58](https://towardsdatascience.com/stock-price-prediction-using-lstm-9a7e9d315e58)

---

**Thank you for reviewing this repository!**
