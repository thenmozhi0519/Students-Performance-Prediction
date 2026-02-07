import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

df = pd.read_excel(r'C:\Users\LENOVO\OneDrive\github projects\studentspre.py')

print("First 5 rows of the dataset:\n", df.head())

print("\nColumn names:", df.columns)

df = df.dropna()

required_columns = ['Attendance (%)', 'Study Hours/Day', 'Participation (%)', 'Marks (%)']
if not all(col in df.columns for col in required_columns):
    raise ValueError("One or more required columns are missing from the dataset!")

X = df[['Attendance (%)', 'Study Hours/Day', 'Participation (%)']]
y = df['Marks (%)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# here we  initialize models
lr_model = LinearRegression()
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
svr_model = SVR(kernel='rbf')

# we  train models here
lr_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
svr_model.fit(X_train, y_train)

model = rf_model  

y_pred = model.predict(X_test)

# Evaluation process
print("\nModel Evaluation Metrics:")
print("R² Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# Predict for New Student 
while True:
  print("\n--- Enter new student details to predict performance ---")
  attendance = float(input("Enter attendance (%): "))
  study_hours = float(input("Enter study hours per day: "))
  participation = float(input("Enter participation level (%): "))
  student_input = np.array([[attendance, study_hours, participation]])
  predicted_score = model.predict(student_input)[0]

  if predicted_score < 50:
    grade = "Improvement needed"
  elif predicted_score < 70:
    grade = "Good"
  else:
    grade = "Excellent"

  print(f"\nPredicted Final Percentage: {predicted_score:.2f}%")
  print(f"Performance Category: {grade}")
  repeat=input("\nDO  you want to predict another student ? (yes/no):").lower()
  if repeat!='yes':
    print("Prediction ended.")
    break
