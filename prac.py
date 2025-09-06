from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import numpy as np

iris = load_iris()
x, y = iris.data, iris.target
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state =42)
model = RandomForestClassifier()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

s_l = float(input("Enter sepal length: "))
s_w = float(input("Enter sepal width: "))
p_l = float(input("Enter petal length: "))
p_w = float(input("Enter petal width: "))

flower = np.array([[s_l,s_w,p_l,p_w]])
pred = model.predict(flower)[0]
pred_name = iris.target_names[pred]
print(f"The predicted species is: {pred_name}")

accuracy = metrics.accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy:.2f}")
# report = metrics.classification_report(y_test, y_pred, target_names=iris.target_names)
# print(report)
precision = metrics.precision_score(y_test,y_pred, average='macro')
recall = metrics.recall_score(y_test,y_pred, average='macro')
f1 = metrics.f1_score(y_test,y_pred, average='macro')
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-score: {f1:.3f}")