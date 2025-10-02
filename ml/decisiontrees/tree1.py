from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

iris=load_iris()
x=iris.data[:,2:]
y=iris.target

tree_clf=DecisionTreeClassifier(max_depth=2)
tree_clf.fit(x,y)

export_graphviz(tree_clf, out_file=image_path("iris_tree.dot"), features_names=iris.feature_names[2:], class_names=iris.target_names,rounded=True,filled=True)