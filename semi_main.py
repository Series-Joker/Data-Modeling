import numpy as np
from sklearn import datasets

iris=datasets.load_iris()
labels=np.copy(iris.target)
random_unlabeled_points=np.random.rand(len(iris.target))
random_unlabeled_points=random_unlabeled_points<0.7
Y=labels[random_unlabeled_points]
labels[random_unlabeled_points]=-1
print("Unlabeled Number:",list(labels).count(-1))

from sklearn.semi_supervised import LabelPropagation
label_prop_model=LabelPropagation()
label_prop_model.fit(iris.data,labels)
Y_pred=label_prop_model.predict(iris.data)
Y_pred=Y_pred[random_unlabeled_points]
from sklearn.metrics import accuracy_score,recall_score,f1_score
print("ACC:",accuracy_score(Y,Y_pred))
print("REC:",recall_score(Y,Y_pred,average="micro"))
print("F-Score",f1_score(Y,Y_pred,average="micro"))