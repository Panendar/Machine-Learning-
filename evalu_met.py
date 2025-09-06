from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import  numpy as np

cancer = load_breast_cancer()
# print(cancer)
x, y = cancer.data, cancer.target
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)
# pred_name = cancer.target_names[y_pred]
# print(f"The predicted class is: {pred_name}")
accuracy = metrics.accuracy_score(y_test,y_pred)
precision = metrics.precision_score(y_test,y_pred)
recall = metrics.recall_score(y_test,y_pred)
F1 = metrics.f1_score(y_test,y_pred)
confusion = metrics.confusion_matrix(y_test,y_pred)
TN_TP = confusion[0,0] + confusion[1,1]
total = confusion.sum()
confusion_metrics_score = TN_TP / total
print(f"Accuracy: {accuracy:.2f}")
print("==============================")
print(f"Precision: {precision:.2f}")
print("==============================")
print(f"Recall: {recall:.2f}")
print("==============================")
print(f"F1-score: {F1:.2f}")
print("==============================")
print("Confusion Matrix:")
print(confusion)
print("==============================")
print(f"Confusion Matrix Metrics Score: {confusion_metrics_score:.2f}")