# ✨🐼 MNIST Classifier 🐼✨
# We’re going to train two classifiers (SGD & RandomForest)
# to detect whether a digit is a "5" or not (💖5 vs. not-5💖).
# The MNIST dataset is often called the "Hello World" of ML 🌸
# This is my attempt to write the code myself from the book 
# "Hands on Machine Learning with Scikit-Learn, Keras and Tensorflow"
# chapter-Classification 

#Here, we first fit a SGD classifer
#The we do cross validation with K Fold Stratification on a clone SGD classifier 
#Then we run a RandomForestClassifier
#Then we plot all of this and compare the models 
#Don't forget to read about these terms first to understand the code better
#Its a bit shabby and all over the place because I'm just starting, apologies!

# 🌟 Imports

from sklearn.datasets import fetch_openml
#get the dataset from here

from sklearn.linear_model import SGDClassifier
#SGD classifier is a classification algorithm that employs the Stochastic Gradient Descent to find the optimal parameters.
 
from sklearn.ensemble import RandomForestClassifier
#yet to learn about this one, but I have still tried to implement it as much as i could. 

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.base import clone
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

#🌸 Load MNIST dataset
print("Loading MNIST... please wait ⏳")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist["data"], mnist["target"].astype("uint8")

#🧩 Split into train/test (60k train, 10k test)
# The MNIST dataset is already split properly in train and test
# So that the data needed is put properly in train and test sets without bias
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

#🎯 Binary labels: Is it a 5?
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

#🐣 Train SGD Classifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

#🍀 What happens under the hood for fit()? 
#- Start with random weights 🎲
#- Make predictions ✨
#- Compare vs. truth (is it 5?) 🤔
#- Adjust weights a little using SGD 🏃
#- Repeat many times until the model learns 💡



#🎀 Cross-validation with StratifiedKFold
#split the training data into 3 folds.
#2 for training, 1 for testing
#dataset reused 3 times but it is split differently.

#clone the model
#train 3 new models and test each 

skfolds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_fold = X_train[test_index]
    y_test_fold = y_train_5[test_index]
    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    acc = sum(y_pred == y_test_fold) / len(y_pred)
    print(f"🌸 SGD fold accuracy: {acc:.3f}")

#🌳 Random Forest Classifier
forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(
    forest_clf, X_train, y_train_5, cv=5, method="predict_proba"
)

#🌀 ROC Curve calculation
# SGD → use decision_function
y_scores_sgd = cross_val_predict(
    sgd_clf, X_train, y_train_5, cv=5, method="decision_function"
)
fpr_sgd, tpr_sgd, _ = roc_curve(y_train_5, y_scores_sgd)

# Random Forest → use positive class probability
y_scores_forest = y_probas_forest[:, 1]
fpr_forest, tpr_forest, _ = roc_curve(y_train_5, y_scores_forest)

#🎨 Plot ROC Curves
plt.rcParams['font.family'] = 'Segoe UI Emoji'
plt.figure(figsize=(8,6))
plt.plot(fpr_sgd, tpr_sgd, "b:", linewidth=2, label="💙 SGD Classifier")
plt.plot(fpr_forest, tpr_forest, "g-", linewidth=2, label="💚 Random Forest")
plt.plot([0,1],[0,1],"k--", label="Random Guess")


plt.xlabel("False Positive Rate (💔)", fontsize=12)
plt.ylabel("True Positive Rate (💖 Recall)", fontsize=12)
plt.title("🌸ROC Curve - Detecting Digit '5'🌸", fontsize=14)
plt.legend(loc="lower right", fontsize=10)
plt.grid(alpha=0.3)
plt.show()

# 🌟 Print AUC scores
print("💙 SGD AUC:", roc_auc_score(y_train_5, y_scores_sgd))
print("💚 Random Forest AUC:", roc_auc_score(y_train_5, y_scores_forest))

print("✨ Conclusion: You did it! Listen to your favourite song now! 💖")

#Final notes:
#SGD is a linear classifier, tends to be fast but less flexible
#Random Forest is non-linear, usually gives better Area Under the Curve/ROC