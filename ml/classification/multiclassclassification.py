#✨🐼 MNIST Multiclass Classifier 🐼✨
#We’re going to train SVM & SGD models to classify digits (0–9) 🌸
#First we try plain SVM (but on a smaller subset to save time 💖),
#then wrap it into One-vs-Rest (OvR),
#then use a SGD Classifier with scaling 🏃💙
#Let’s see who wins! 🏆💙💚

#🌟 Imports
from sklearn.svm import SVC
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
import numpy as np

#🌸 Load MNIST dataset
print("Loading MNIST... please wait ⏳")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y=mnist["data"], mnist["target"].astype("uint8")

#🧩 Split into train/test
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

#🌀 Scaling (important for SGD!)
scaler = StandardScaler()
X_train_scaled=scaler.fit_transform(X_train.astype(np.float64))
X_test_scaled=scaler.transform(X_test.astype(np.float64))

#🎯 Pick one sample digit to test
some_digit=X[0]
some_label = y[0]

#💙 Train a plain SVM 
#❌ Full dataset (too slow!)
#svm_clf=SVC()
#svm_clf.fit(X_train, y_train)

#✅ Faster: use smaller subset (e.g., 10,000 samples)
print("Training SVM on 10,000 samples instead of full dataset ✨")
X_small = X_train[:10000]
y_small = y_train[:10000]

svm_clf = SVC()
svm_clf.fit(X_small, y_small)
print("💙 SVM predicts:", svm_clf.predict([some_digit]))
print("💖 True label:", some_label)
#Check decision scores (confidence per class)
some_digit_scores = svm_clf.decision_function([some_digit])
print("💙 SVM raw scores (per class):", some_digit_scores)


#💚 One-vs-Rest SVM (multiclass mode, still on small set!)
ovr_clf = OneVsRestClassifier(SVC())
ovr_clf.fit(X_small, y_small)
print("💚 OvR SVM predicts:", ovr_clf.predict([some_digit]))
print("🌸 Number of classifiers trained:", len(ovr_clf.estimators_))

#🐥 SGD Classifier (linear model, trained on full scaled data!)
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train_scaled, y_train)
print("🐥 SGD predicts:", sgd_clf.predict(scaler.transform([some_digit])))
print("🐥 SGD decision scores:", sgd_clf.decision_function(scaler.transform([some_digit])))

#🍀 Cross-validation accuracy for SGD
scores = cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")
print("🐥 SGD CV accuracy:", scores)
print("🌸 Average accuracy:", scores.mean())

print("✨ Done! Daily Reminder: Being ready is not a feeling, its a decision.")
