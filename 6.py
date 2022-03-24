from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
dataset=datasets.load_diabetes()
model=GaussianNB()
model.fit(dataset.data,dataset.target)
expected=dataset.target
predicted=model.predict(dataset.data)
print(metrics.confusion_matrix(expected,predicted))
print(metrics.accuracy_score(expected,predicted))
