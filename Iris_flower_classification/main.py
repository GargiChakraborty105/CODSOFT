import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import tkinter as tk
from tkinter import ttk
from sklearn.datasets import load_iris

# Load the dataset
data = pd.read_csv('IRIS.csv')

# Split into features (X) and labels (y)
X = data.drop('species', axis=1)
y = data['species']


# Create and train the SVM model
model = SVC(kernel='linear', C=1.0)
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# Calculate accuracy and print classification report
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

target_names = data['species'].unique()
print(classification_report(y_test, y_pred, target_names=target_names))



# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the SVM model
model = SVC(kernel='linear', C=1.0)
model.fit(X_train, y_train)

# Create the main application window
app = tk.Tk()
app.title("Iris Flower Species Predictor")

# Create labels and entry fields for sepal and petal measurements
ttk.Label(app, text="Sepal Length:").pack()
sepal_length_entry = ttk.Entry(app)
sepal_length_entry.pack()

ttk.Label(app, text="Sepal Width:").pack()
sepal_width_entry = ttk.Entry(app)
sepal_width_entry.pack()

ttk.Label(app, text="Petal Length:").pack()
petal_length_entry = ttk.Entry(app)
petal_length_entry.pack()

ttk.Label(app, text="Petal Width:").pack()
petal_width_entry = ttk.Entry(app)
petal_width_entry.pack()

# Function to predict the species
def predict_species():
    sepal_length = float(sepal_length_entry.get())
    sepal_width = float(sepal_width_entry.get())
    petal_length = float(petal_length_entry.get())
    petal_width = float(petal_width_entry.get())
    
    # Make the prediction
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    species = iris.target_names[prediction][0]
    
    result_label.config(text=f"Predicted Species: {species}")

# Create a predict button
predict_button = ttk.Button(app, text="Predict", command=predict_species)
predict_button.pack()

# Create a label to display the prediction result
result_label = ttk.Label(app, text="")
result_label.pack()

# Start the main loop
app.mainloop()
