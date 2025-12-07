import numpy as np
import pandas as pd
class LogisticRegression:
    def __init__(self,lr=0.1,n_iters=1000, batch_size=None,l2=0.0,verbose=False):
        self.lr=lr 
        self.n_iters=n_iters
        self.w=None
        self.b=None
        self.batch_size=batch_size
        self.l2=l2
        self.verbose=verbose
        self.loss_history=[] 
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    def compute_loss(self,X,y):
        m=X.shape[0]
        z=X @self.w+self.b
        y_hat=self.sigmoid(z)
        ce=-(y*np.log(y_hat)+(1-y)*np.log(1-y_hat)).mean()
        reg=0.5*self.l2*np.sum(self.w**2)/m
        return ce+reg
    def fit(self,X,y):
        m,n=X.shape
        self.w=np.zeros(n)
        self.b=0
        if self.batch_size is None:
            batch_size=m
        else:
            batch_size=self.batch_size
        for _ in range(self.n_iters):
            for start in range(0,m,batch_size):
                end=start+batch_size
                X_batch=X[start,end]
                y_batch=y[start,end]
                bs=X_batch.shape[0]
                z=X_batch@self.w+self.b
                y_hat=self.sigmoid(z)
                dw=(1/bs)*X_batch.T @ (y_hat-y_batch)+(self.l2/m)*self.w
                db=(1/bs)*np.sum(y_hat-y_batch)
                self.w-=self.lr*dw
                self.b-=self.lr*db
            loss=self.compute_loss(X,y)
            self.loss_history.append(loss)
            if self.verbose and (_%max(1,self.n_iters//10)==0):
                print(f"iter{_}/{self.n_iters}loss={loss:.6}")
    def predict(self,X):
        return ((self.sigmod(X @ self.w+self.b)>=0.5).astype(int)
