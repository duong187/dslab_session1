import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
#import statsmodels.api as sm


def normalize_add_ones(X):
    X=np.array(X)
    X_max=np.array([[np.amax(X[:,column_id])
                    for column_id in range(X.shape[1])]
                for _ in range(X.shape[0])])
    X_min=np.array([[np.amin(X[:,column_id])
                    for column_id in range(X.shape[1])]
                for _ in range(X.shape[0])])
    X_normalized=(X-X_min)/(X_max-X_min)
    ones=np.ones(X_normalized.shape[0])
    return np.column_stack((ones,X_normalized))

    
class RidgeRegression:
    def __init__(self):
        return
    def fit(self,X_train,Y_train,LAMBDA):
        assert len(X_train.shape)==2 and X_train.shape[0]==Y_train.shape[0]
        W=np.linalg.inv(X_train.transpose().dot(X_train)+LAMBDA*np.identity(X_train.shape[1])).dot(X_train.transpose()).dot(Y_train)
        return W
    def fit_gradient(self,X_train,Y_train,LAMBDA,learning_rate,max_num_epoch=100,batch_size=128):
        W=np.random.randn(X_train.shape[1])
        last_loss=10e+8
        for ep in range(max_num_epoch):
            arr=np.array(range(X_train.shape[0]))
            np.random.shuffle(arr)
            X_train=X_train[arr]
            Y_train=Y_train[arr]
            total_minibatch=int(np.ceil(X_train.shape[0]/batch_size))
            for i in range(total_minibatch):
                index=i*batch_size
                X_train_sub=X_train[index:index+batch_size]
                Y_train_sub=Y_train[index:index+batch_size]
                grad=X_train_sub.transpose().dot(X_train_sub.dot(W)-Y_train_sub)+LAMBDA*W
                W=W-learning_rate*grad
            new_loss=self.compute_RSS(self.predict(W,X_train),Y_train)
            if(np.abs(new_loss-last_loss)<=1e-5):
                break
            last_loss=new_loss
        return W
    def predict(self,W,X_new):
        X_new=np.array(X_new)
        Y_new=X_new.dot(W)
        return Y_new
    def compute_RSS(self,Y_new,Y_predict):
        loss=1./Y_new.shape[0]*(np.sum((Y_new-Y_predict)**2))
        return loss
    def get_the_best_LAMBDA(self,X_train,Y_train):
        def cross_validation(num_folds,LAMBDA):
            row_id=np.array(range(X_train.shape[0]))
            valid_id=np.split(row_id[:len(row_id)-len(row_id)%num_folds],num_folds)
            valid_id[-1]=np.append(valid_id[-1],row_id[len(row_id)-len(row_id)%num_folds:])
            train_id=[[k for k in row_id if k not in valid_id[i]] for i in range(num_folds)]
            aver_RSS=0
            for i in range(num_folds):
                valid_part={'X':X_train[valid_id[i]],'Y':Y_train[valid_id[i]]}
                train_part={'X':X_train[train_id[i]],'Y':Y_train[train_id[i]]}
                W=self.fit(train_part['X'],train_part['Y'],LAMBDA)
                Y_new=self.predict(W,valid_part['X'])
                aver_RSS+=self.compute_RSS(Y_new,valid_part['Y'])
            return aver_RSS/num_folds
        def range_scan(best_LAMBDA,min_RSS,LAMBDA_values):
            for current_LAMBDA in LAMBDA_values:
                current_RSS=cross_validation(num_folds=5,LAMBDA=current_LAMBDA)
                if current_RSS<min_RSS:
                    min_RSS=current_RSS
                    best_LAMBDA=current_LAMBDA
            return best_LAMBDA,min_RSS
        best_LAMBDA,min_RSS=range_scan(best_LAMBDA=0,min_RSS=10000**2,LAMBDA_values=range(50))
        LAMBDA_values=[k*1./1000 for k in range(max(0,(best_LAMBDA-1)*1000),(best_LAMBDA+1)*1000,1)]
        best_LAMBDA,min_RSS=range_scan(best_LAMBDA=best_LAMBDA,min_RSS=min_RSS,LAMBDA_values=LAMBDA_values)
        return best_LAMBDA

if __name__=='__main__':
    data = pd.read_csv('raw_data.txt', delimiter='\s+')
    data.columns = ["I","A1","A2","A3","A4","A5","A6","A7","A8","9","10","11","A12","A13","A14","A15","B"]
    X=data[["A1","A2","A3","A4","A5","A6","A7","A8","9","10","11","A12","A13","A14","A15"]]
    Y=data[["B"]]
    X=np.array(X)
    Y=np.array(Y)
    X_norm =normalize_add_ones(X)
    X_train,Y_train=X_norm[:50],Y[:50]
    X_test,Y_test=X_norm[50:],Y[50:]
    ridge_regression=RidgeRegression()
    best_lambda=ridge_regression.get_the_best_LAMBDA(X_train=X_train,Y_train=Y_train)
    print("best lambda:",best_lambda)
    W=ridge_regression.fit(X_train=X_train,Y_train=Y_train,LAMBDA=best_lambda)
    Y_predict=ridge_regression.predict(W=W,X_new=X_test)
    loss=ridge_regression.compute_RSS(Y_new=Y_test,Y_predict=Y_predict)
    print(loss)


