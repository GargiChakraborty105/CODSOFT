import tkinter as tk
from tkinter import ttk
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer


# Load the pre-trained model
best_params = {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 2, 'min_samples_leaf': 1}
best_model = RandomForestClassifier(random_state=42, **best_params)

# Load and clean the dataset
data = pd.read_csv('tested.csv')
data['Age'].fillna(data['Age'].median(), inplace=True)
data = pd.get_dummies(data, columns=['Sex'], drop_first=True)
features = ['Pclass', 'Age', 'Fare', 'Sex_male']
X = data[features]
y = data['Survived']

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Train the model
best_model.fit(X, y)

# Create the GUI
def predict_survival():
    input_data = []
    for entry in entry_fields:
        input_data.append(float(entry.get()))

    prediction = best_model.predict([input_data])[0]
    if prediction == 0:
        prediction_label.config(text="Prediction: Did not survive")
    else:
        prediction_label.config(text="Prediction: Survived")

# Create the main window
root = tk.Tk()
root.title("Titanic Survival Predictor")

# Create input labels and entry fields
entry_fields = []
for i, feature in enumerate(features):
    label = ttk.Label(root, text=feature + ":", font=("Algerian", 12))  
    label.grid(row=i, column=0, padx=10, pady=5)
    var = tk.StringVar()
    entry = ttk.Entry(root, textvariable=var)
    entry.grid(row=i, column=1, padx=10, pady=5)
    entry_fields.append(entry)

# Create prediction button
predict_button = ttk.Button(root, text="Predict Survival", command=predict_survival)
predict_button.grid(row=len(features), columnspan=2, padx=10, pady=10)

# Create label to display the prediction result
prediction_label = ttk.Label(root, text="", font=("Arial", 14, "bold"))  
prediction_label.grid(row=len(features) + 1, columnspan=2, padx=10, pady=5)

root.mainloop()
