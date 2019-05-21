# --------------------------------------------------------------------------
# ----------------  System Analysis and Decision Making --------------------
# --------------------------------------------------------------------------
#  Assignment 3: Logistic regression
#  Authors: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------


import numpy as np
import functools


def sigmoid(x):
    '''
    :param x: input vector (size: Nx1)
    :return: vector of sigmoid function values calculated for x (size: Nx1)
    '''
    return np.divide(1,np.add(1,np.exp(-x)))


def logistic_cost_function(w, x_train, y_train):
    '''
    :param w: model parameters (size: Mx1)
    :param x_train: training set features (size: NxM)
    :param y_train: training set labels (size: Nx1)
    :return: function returns tuple (val, grad), where val is a value of logistic function and grad is its gradient (calculated for parameters w)
    '''
    sig=sigmoid(x_train@w)
    log_cost_fun=np.divide(np.sum(y_train*np.log(sig)+(1-y_train)*np.log(1-sig)),-1*sig.shape[0])
    grad=x_train.transpose()@((sig-y_train)/sig.shape[0])
    return log_cost_fun,grad


def gradient_descent(obj_fun, w0, epochs, eta):
    '''
    :param obj_fun: objective function that is minimized. To call the function use expression "val,grad = obj_fun(w)".
    :param w0: starting point (size: Mx1)
    :param epochs: number of epochs / iterations of an algorithm
    :param eta: learning rate
    :return: function optimizes obj_fun using gradient descent. It returns (w,func_values),
    where w is vector of optimal model parameters and func_valus is vector of objective function values [epochs x 1], calculated for each epoch
	'''
    w=w0
    func_values=[]
    f_val,w_grad=obj_fun(w)
    for k in range(epochs):
        w=w-eta*w_grad
        f_val,w_grad=obj_fun(w)
        func_values.append(f_val)
    return w,np.reshape(np.array(func_values),(epochs,1))


def stochastic_gradient_descent(obj_fun, x_train, y_train, w0, epochs, eta, mini_batch):
    """
	:param obj_fun: objective function that is minimized. To call the function use expression "val,grad = obj_fun(w,x,y)",
	where x,y indicates mini-batches.
    :param x_train: training data (size: NxM)
    :param y_train: training data (size: Nx1)
    :param w0: starting point (size: Mx1)
    :param epochs: number of epochs
    :param eta: learning rate
    :param mini_batch: size of mini-batches
    :return: function optimizes obj_fun using stochastic gradient descent. It returns tuple (w,func_values),
    where w is vector of optimal value of model parameters and func_valus is vector of objective function values [epochs x 1], calculated for each epoch.
    REMARK! Value of func_values is calculated for entire training set!
    """
    w=w0
    w_values=[]
    x_views=[]
    y_views=[]
    m_amount=int(y_train.shape[0]/mini_batch)
    for m in range(m_amount):
        x_views.append(x_train[m*mini_batch:(m+1)*mini_batch])
        y_views.append(y_train[m*mini_batch:(m+1)*mini_batch])
    for k in range(epochs):
        for m in range(m_amount):
            v,w_grad=obj_fun(w,x_views[m],y_views[m])
            w=w-eta*w_grad
        w_values.append(w)
    f=lambda w_val:obj_fun(w_val,x_train,y_train)
    xx=list(map(f,w_values))
    func_values,v=zip(*xx)
    return w,np.reshape(np.array(func_values),(epochs,1))


def regularized_logistic_cost_function(w, x_train, y_train, regularization_lambda):
    '''
    :param w: model parameters (size: Mx1)
    :param x_train: training set - features (size: NxM)
    :param y_train: training set - labels (size: Nx1)
    :param regularization_lambda: regularization coefficient
    :return: function returns tuple (val, grad), where val is a value of logistic function with regularization l2,
    and grad is (calculated for model parameters w)
    '''
    ws=np.delete(w,0)
    sig=sigmoid(x_train@w)
    norm=regularization_lambda/2*(pow(np.linalg.norm(ws),2))
    log_cost_fun=np.divide(np.sum(y_train*np.log(sig)+(1-y_train)*np.log(1 - sig)),-1*sig.shape[0])
    log_cost_fun_reg=log_cost_fun+norm
    w=w.transpose()
    wz=w.copy().transpose()
    wz[0]=0
    grad=(x_train.transpose()@(sig-y_train))/sig.shape[0]+regularization_lambda*wz
    return log_cost_fun_reg,grad


def prediction(x, w, theta):
    '''
    :param x: observation matrix (size: NxM)
    :param w: vector of model parameters (size: Mx1)
    :param theta: classification threshold [0,1]
    :return: function calculates vector y (size: Nx1) of labels {0,1}, calculated for observations x
    using model parameters w and classification threshold theta
    '''
    sig=sigmoid(x@w)
    return (sig>theta).astype(int).reshape(x.shape[0], 1)


def f_measure(y_true, y_pred):
    '''
    :param y_true: vector of ground truth labels (size: Nx1)
    :param y_pred: vector of predicted labels (size: Nx1)
    :return: value of F-measure
    '''
    tp=np.sum(np.bitwise_and(y_true,y_pred))
    fp=np.sum(np.bitwise_and(np.bitwise_not(y_true),y_pred))
    fn=np.sum(np.bitwise_and(y_true,np.bitwise_not(y_pred)))
    return (2*tp)/(2*tp+fp+fn)


def model_selection(x_train, y_train, x_val, y_val, w0, epochs, eta, mini_batch, lambdas, thetas):
    '''
    :param x_train: trainig set - features (size: NxM)
    :param y_train: training set - labels (size: Nx1)
    :param x_val: validation set - features (size: Nval x M)
    :param y_val: validation set - labels (size: Nval x 1)
    :param w0: initial value of w
    :param epochs: number of SGD iterations
    :param eta: learning rate
    :param mini_batch: mini-batch size
    :param lambdas: list of lambda values that are evaluated in model selection procedure
    :param thetas: list of theta values that are evaluated in model selection procedure
    :return: Functions makes a model selection. It returs tuple (regularization_lambda, theta, w, F), where regularization_lambda
    is the best value of regularization parameter, theta is the best classification threshold, and w is the best model parameter vector.
    Additionally function returns matrix F, which stores F-measures calculated for each pair (lambda, theta).
    REMARK! Use SGD and training criterium with l2 regularization for training.
    '''
    tuples=[]
    fmeasure_list=[]
    wlist=[]
    alen=int(len(thetas))
    blen=int(len(lambdas))
    min_index=0
    def generate(index):
        nonlocal wlist
        (w,_)=stochastic_gradient_descent(functools.partial(regularized_logistic_cost_function,regularization_lambda=lambdas[index]),x_train, y_train, w0,epochs,eta,mini_batch)
        wlist.append(w)
    def select(index):
        nonlocal min_index
        i=int(index/alen)
        j=int(index%alen)
        measure=f_measure(y_val,prediction(x_val,wlist[i],thetas[j]))
        tuples.append((i,j,wlist[i]))
        fmeasure_list.append(measure)
        if fmeasure_list[min_index]<measure:
            min_index=index
    list(map(generate,range(blen)))
    list(map(select,range(alen*blen)))
    requlization_lambda=lambdas[tuples[min_index][0]]
    theta=thetas[tuples[min_index][1]]
    w=tuples[min_index][2]
    F=np.array(fmeasure_list).reshape(blen, alen)
    return (requlization_lambda,theta,w,F)