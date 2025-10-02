#Ensemble Voting Classifier for Handwritten Digits

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score 

from sklearn.ensemble import VotingClassifier

digits=load_digits()
x,y=digits.data, digits.target
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.3, random_state=42)


log=LogisticRegression(max_iter=1000)
rf=RandomForestClassifier(n_estimators=100)
knn=KNeighborsClassifier()
svm=SVC(probability=True)

for clf in [log, rf, knn, svm]:
 clf.fit(x_train, y_train)
 y_pred=clf.predict(x_test)
 print(clf.__class__.__name__, accuracy_score(y_test,y_pred))

voting=VotingClassifier(estimators=[('lr', log), ('rf', rf), ('knn', knn), ('svm', svm)], voting='hard')

voting.fit(x_train,y_train)
y_pred=voting.predict(x_test)
print("Voting classifier(Hard) accuracy is ", accuracy_score(y_test, y_pred))

votingsoft=VotingClassifier(estimators=[('lr', log), ('rf', rf), ('knn', knn), ('svm', svm)], voting='soft')

votingsoft.fit(x_train, y_train)
ysoft=votingsoft.predict(x_test)
print("Voting classifier(soft) accuracy is ", accuracy_score(y_test, ysoft))

