from sklearn.datasets import fetch_mldata  
import numpy as np  
mnist = fetch_mldata('MNIST original')  
X_train = mnist.data[:60000]/255.0  
Y_train = mnist.target[:60000]

X_test = mnist.data[60000:]/255.0  
Y_test = mnist.target[60000:]

Y_train[Y_train > 1.0] = 0.0  
Y_test[Y_test > 1.0] = 0.0  
# Lets do logistic regression using Sci-kit Learn.

from sklearn import linear_model  
clf = linear_model.LogisticRegression()  
clf.fit(X_train,Y_train)

Y_pred = clf.predict(X_test)  
def logistic(x):  
    return 1.0/(1.0+np.exp(-x))

# The loss function
def cross_entropy_loss(X,Y,w,N):  
    Z = np.dot(X,w)
    Y_hat = logistic(Z)

    L = (Y*np.log(Y_hat)+(1-Y)*np.log(1-Y_hat))
    return (-1.0*np.sum(L))/N

# Gradient of the loss function
def D_cross_entropy_loss(X,Y,w,N):  
    Z = np.dot(X,w)
    Y_hat = logistic(Z)

    DL = X*((Y_hat-Y).reshape((N,1)))
    DL = np.sum(DL,0)/N
    return DL


def gradient_descent(X_train,Y_train,alpha,epsilon):  
    # Append "1" before the vectors
    N,K = X_train.shape
    X = np.ones((N,K+1))
    X[:,1:] = X_train
    Y = Y_train

    w = np.random.randn(K+1)
    DL = D_cross_entropy_loss(X,Y,w,N)

    while np.linalg.norm(DL)>epsilon:
        L = cross_entropy_loss(X,Y,w,N)
        #Gradient Descent step
        w = w - alpha*DL
        print "Loss:",L,"\t Gradient norm:", np.linalg.norm(DL)
        DL = D_cross_entropy_loss(X,Y,w,N)

    L = cross_entropy_loss(X,Y,w,N)
    DL = D_cross_entropy_loss(X,Y,w,N)
    print "Loss:",L,"\t Gradient norm:", np.linalg.norm(DL)

    return w

# After playing around with different values, I found these to be satisfactory 
alpha = 1  
epsilon = 0.01

w_star = gradient_descent(X_train,Y_train,alpha,epsilon)

N,K = X_test.shape  
X = np.ones((N,K+1))  
X[:,1:] = X_test  
Y = Y_test  
Z = np.dot(X,w_star)  
Y_pred = logistic(Z)

Y_pred[Y_pred>=0.5] = 1.0  
Y_pred[Y_pred<0.5] = 0.0  

from sklearn.metrics import classification_report  
print classification_report(Y_test,Y_pred)  