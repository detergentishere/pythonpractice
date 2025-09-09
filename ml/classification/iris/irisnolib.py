#✨🐼Logistic Regression from Scratch on Iris Dataset🐼✨
#No scikit-learn training, just NumPy magic!🌸💖

from sklearn.datasets import load_iris
import numpy as np

#🌼 Step 1: Load the dataset
iris= load_iris()
X =iris.data.astype(np.float64)   
#Features (150 flowers × 4 features)
y=iris.target.astype(int)        
#Labels (0=setosa, 1=versicolor, 2=virginica)

#🌸 Step 2: Train/Test split (80/20)
#Stack features + labels together
dataset=np.c_[X, y]    # shape = (150, 5) → last column is label
np.random.shuffle(dataset)  # shuffles rows in place

#Split
train_size= int(0.8 * len(dataset))
train_set=dataset[:train_size]
test_set=dataset[train_size:]

#Separate features and labels again
X_train, y_train=train_set[:, :-1], train_set[:, -1].astype(int)
X_test, y_test=test_set[:, :-1], test_set[:, -1].astype(int)


#🌼 Step 3: Standardize (mean=0, variance=1) for stable training
mean=X_train.mean(axis=0)
std=X_train.std(axis=0)
std_safe=np.where(std==0,1,std)   # avoid division by zero 🌸
X_train_scaled=(X_train-mean) / std_safe
X_test_scaled =(X_test-mean) / std_safe

#🌸 Step 4: Setup model dimensions
n_features=X_train.shape[1]          # 4 features
K=len(np.unique(y_train))            # 3 classes (flowers)
N=X_train_scaled.shape[0]            # number of training samples

#🌼 Step 5: One-hot encode labels
Y_onehot=np.zeros((N, K))
Y_onehot[np.arange(N), y_train]=1    # 💖 magic one-hot encoding

#🌸 Step 6: Hyperparameters
reg_lambda=0.01      # regularization strength
learning_rate=0.1    # step size for gradient descent
epochs=500           # number of training iterations

#🌼 Step 7: Initialize weights & bias
W=0.01 * np.random.randn(n_features, K)   # small random weights
b=np.zeros((K,))                          # bias starts at 0

#🌸 Helper: Softmax function
def softmax(Z):
    Z_max=np.max(Z, axis=1, keepdims=True)   # stabilize numerically
    expZ= np.exp(Z-Z_max)
    return expZ/np.sum(expZ, axis=1, keepdims=True)

#🌼 Helper: Compue loss (cross-entropy + L2 regularization)
def compute_loss(Y_true, probs, W, reg_lambda):
    N=Y_true.shape[0]
    core_loss=-np.sum(Y_true*np.log(probs + 1e-15))/N   # 💔 CE loss
    reg_loss=0.5*reg_lambda*np.sum(W*W)               # 🔒 penalty
    return core_loss+reg_loss, core_loss

#🌸 Step 8: Training loop (Gradient Descent)
for epoch in range(1, epochs+1):
    #Forward pass 🌀
    scores=X_train_scaled.dot(W)+b
    probs=softmax(scores)

    #Loss 📉
    loss, core=compute_loss(Y_onehot, probs, W, reg_lambda)

    #Backpropagation 🔄
    dScores=(probs-Y_onehot)/N
    dW=X_train_scaled.T.dot(dScores)+reg_lambda*W
    db=np.sum(dScores, axis=0)

    #Update parameters 💪
    W-=learning_rate*dW
    b-=learning_rate*db

    #Every 100 epochs, print progress 🌟
    if epoch%100==0:
        preds=np.argmax(probs, axis=1)
        acc=np.mean(preds == y_train)
        print(f"🌸 Epoch {epoch}, Loss={loss:.4f}, Train Accuracy={acc:.4f}")

#🌼 Step-9 : Evaluate on Test Set
scores_test=X_test_scaled.dot(W)+b
probs_test=softmax(scores_test)
y_pred=np.argmax(probs_test, axis=1)
test_acc=np.mean(y_pred==y_test)

print("\n✨ Training Complete! ✨")
print("💖 Final Test Accuracy:", test_acc)
print("Reminder: Being ready isn't a feeling, it's a choice")