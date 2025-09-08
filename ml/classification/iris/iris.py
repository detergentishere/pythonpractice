#✨🐼 Iris Logistic Regression Classifier 🐼✨
#We’ll train Logistic Regression on ALL 4 features of the Iris dataset 🌸
#Then reduce data to 2D with PCA just for visualization 🎨
#Let’s see how well our model separates flowers 💖🌼

#🌟 Imports
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

#🌸 Load dataset (Iris has 150 flowers, 3 species, 4 features each)
iris=load_iris()
X, y=iris.data, iris.target
print("🌼 Features:", iris.feature_names)
print("🌸 Classes:", iris.target_names)

#🧩 Split dataset → Train (80%) + Test (20%)
X_train, X_test, y_train, y_test =train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#🌀 Scale features → standardize (mean=0, variance=1)
scaler = StandardScaler()
X_train_scaled= scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

#💙 Train Logistic Regression on ALL 4 features
clf=LogisticRegression(max_iter=200, random_state=42, multi_class="multinomial")
clf.fit(X_train_scaled, y_train)

#🌸 Evaluate the model
print("💖 Train accuracy:", clf.score(X_train_scaled, y_train))
print("💖 Test accuracy:", clf.score(X_test_scaled, y_test))

#🐝 Try predicting one flower
sample=X_test[0].reshape(1, -1)
print("🌼 Sample flower features:", sample)
print("🌸 True label:", iris.target_names[y_test[0]])
print("🌸 Predicted label:", iris.target_names[clf.predict(sample)[0]])

#🎨 Visualization with PCA
#Reduce 4D → 2D (just for human eyes 👀✨)
pca=PCA(n_components=2)
X_pca=pca.fit_transform(scaler.transform(X))

#🌀 Make a mesh grid in PCA space (like painting a canvas 🎨)
x_min, x_max=X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max=X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

#🌈 Transform grid back into 4D → scale → predict class
grid_pca=np.c_[xx.ravel(), yy.ravel()]
grid_original=pca.inverse_transform(grid_pca)
Z=clf.predict(scaler.transform(grid_original))
Z=Z.reshape(xx.shape)

#🎨 Plot decision regions + flowers
plt.figure(figsize=(8,6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Set3)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, edgecolor="k", cmap=plt.cm.Set3, s=60)

plt.xlabel("PCA Feature 1 🌼")
plt.ylabel("PCA Feature 2 🌸")
plt.title("🌸 Logistic Regression on Iris (Trained on 4 Features, PCA Visualized) 🌸")
plt.grid(alpha=0.3)
plt.show()

print("✨ Done! Be like a flower 🌼—calm, bright, and blooming at your own pace 💖")
