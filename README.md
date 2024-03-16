# ml-weight-classifier

This project aims to predict weight categories based on gender, height, and weight information using machine learning techniques.

## Dataset

The dataset used in this project is named "500_Person_Gender_Height_Weight_Index.csv". It contains information about the gender, height, weight, and weight index of 500 individuals.

## Usage

1. **Setup:** Make sure you have Python installed along with the required libraries listed in `requirements.txt`.

2. **Data Preprocessing:** Run the script `data_preprocessing.py` to load and preprocess the dataset. This script includes steps such as handling categorical variables, scaling numerical features, and splitting the data into training and testing sets.

3. **Model Training:** Execute the script `train_model.py` to train the machine learning model. The script uses a Random Forest classifier and performs hyperparameter tuning using Grid Search to find the best parameters.

4. **Model Evaluation:** After training, the model is evaluated using various metrics such as classification report, confusion matrix, and accuracy score. These metrics are printed in the console for analysis.

5. **Live Prediction:** You can make live predictions using the function `index_prediction.py`. Provide the gender, height, and weight information for an individual, and the function will return the predicted weight category.

## Requirements

- Python 3.x
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn

Install the required packages using the following command:

