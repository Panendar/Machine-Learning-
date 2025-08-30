from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

iris = load_iris()
x, y = iris.data, iris.target

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)

model = RandomForestClassifier()
model.fit(x_train, y_train)

predictions = model.predict(x_test)
print(predictions)