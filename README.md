# Python-developer
import pandas as pd
df=pd.read_csv("Utterances.txt-Python-Developer.csv")

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

from sklearn.neighbors import KNeighborsClassifier
n = KNeighborsClassifier(n_neighbors=3)
n.fit(x_train, y_train)

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
y_predict = n.predict(x_test)
print(accuracy_score(y_test,y_predict))
print(confusion_matrix(y_test,y_predict))
print(classification_report(y_test,y_predict))

from sklearn.ensemble import RandomForestClassifier
r=RandomForestClassifier()
r.fit(x_train,y_train)

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
y_predict1 = r.predict(x_test)
print(accuracy_score(y_test,y_predict1))
print(confusion_matrix(y_test,y_predict1))
print(classification_report(y_test,y_predict1))

from sklearn.linear_model import LogisticRegression
l=LogisticRegression()
l.fit(x_train,y_train)


from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
y_predict2 = l.predict(x_test)
print(accuracy_score(y_test,y_predict2))
print(confusion_matrix(y_test,y_predict2))
print(classification_report(y_test,y_predict2))

from sklearn.tree import DecisionTreeClassifier
d=DecisionTreeClassifier()
d.fit(x_train,y_train)

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
y_predict3 = l.predict(x_test)
print(accuracy_score(y_test,y_predict3))
print(confusion_matrix(y_test,y_predict3))
print(classification_report(y_test,y_predict3))

