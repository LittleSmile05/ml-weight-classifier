import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load data
data = pd.read_csv('500_Person_Index.csv')

def map_index_to_category(index):
    categories = ['Extremely Weak', 'Weak', 'Normal', 'Overweight', 'Obesity', 'Extremely Obese']
    return categories[index]


data['Index'] = data['Index'].apply(map_index_to_category)


sns.lmplot('Height', 'Weight', data, hue='Index', size=7, aspect=1, fit_reg=False)

data = pd.get_dummies(data, columns=['Gender'])  # One-hot encoding for gender
y = data['Index']
X = data.drop('Index', axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=101)

param_grid = {'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800, 1000]}
grid_search = GridSearchCV(RandomForestClassifier(random_state=101), param_grid, verbose=3)
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)

y_pred = grid_search.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred) * 100)

def predict_weight_category(details, model, scaler):
    gender = details[0]
    height = details[1]
    weight = details[2]

    gender_encoded = {'Male': [1, 0], 'Female': [0, 1]}
    gender_vector = gender_encoded[gender]
    details_array = np.array([[height, weight] + gender_vector])
    scaled_details = scaler.transform(details_array)
    prediction = model.predict(scaled_details)
    return prediction[0]

your_details = ['Male', 175, 80]
print("Predicted weight category:", predict_weight_category(your_details, grid_search, scaler))




