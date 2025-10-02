# 🌸🐼 Iris Classification 🌼✨

This project uses the classic Iris dataset to classify flowers into three species — **Setosa**, **Versicolor**, and **Virginica** — based on their physical features (sepal length, sepal width, petal length, petal width).

The project contains two implementations of **Logistic Regression**:

---

## 📂 Files

### `iris.py`
- Logistic Regression implemented using **Scikit-learn**.  
- Demonstrates how machine learning is typically done in practice using pre-built libraries.  
- Handles:
  - Loading the Iris dataset.
  - Train/test split.
  - Model training with `LogisticRegression`.
  - Accuracy evaluation on the test set.

---

### `irisnolib.py`
- Logistic Regression implemented **from scratch** using **NumPy only**.  
- Shows the inner mechanics of the algorithm without relying on ML libraries.  
- Includes:
  - Shuffling and splitting data into training/testing sets.
  - Standardizing features (zero mean, unit variance).
  - One-hot encoding of labels.
  - Manual implementation of **softmax**, **cross-entropy loss**, and **L2 regularization**.
  - Gradient descent training loop with **backpropagation**.
  - Progress printing every 100 epochs(total 500 epochs).
  - Final evaluation of accuracy on test data.

---

## 🧠 Key Learning Points
- Difference between **using ML libraries** and **building algorithms from scratch**.  
- How logistic regression works step by step:
  - Predictions via softmax.  
  - Error measurement via cross-entropy loss.  
  - Weight updates via backpropagation and gradient descent.  

---

## 📊 Dataset
- 150 flowers, 50 from each species.  
- 4 numerical features per flower.  
- 3 output classes (0 = Setosa, 1 = Versicolor, 2 = Virginica).  

---

## 💡 Takeaway
This project bridges the gap between:
- **Practical ML (Scikit-learn)** → quick, efficient, industry-style.  
- **Conceptual ML (NumPy scratch implementation)** → deeper understanding of what happens under the hood.
- **Don't forget to stay hydrated! Keep blooming like a flower.**
