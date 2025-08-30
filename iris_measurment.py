from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

iris = load_iris()
x, y = iris.data, iris.target
x_train, x_test,y_train, y_test = train_test_split(x, y, test_size =0.2, random_state=42)
model =RandomForestClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
report = classification_report(y_test, y_pred, target_names = iris.target_names)


s_l = float(input("Enter sepal length: "))
s_w = float(input("Enter sepal width: "))
p_l = float(input("Enter petal length: "))
p_w = float(input("Enter petal width: "))

flower = np.array([[s_l,s_w,p_l,p_w]])
pred_idx = model.predict(flower)[0]
pred_species = iris.target_names[pred_idx]
print(f"The predicted species is: {pred_species}")