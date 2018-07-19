from sklearn.datasets import load_breast_cancer
from sklearn import tree
import numpy as np
can = load_breast_cancer()
print(can.feature_names)
print("-----------")
print(can.target_names)
print(can.data[0])
print(can.data[212])
print(can.data[424])
#print(can)
can.target[[0, 212, 424]]
removed=[10,50,85]
parameters=[8,55,89]
newtarget=np.delete(can.target,removed)
newdata=np.delete(can.data,removed,axis=0)
clf=tree.DecisionTreeClassifier()
clf1=clf.fit(newdata,newtarget)
print("-----------")
predicts=clf.predict(can.data[parameters])
print(predicts)
print(can.target[removed])
