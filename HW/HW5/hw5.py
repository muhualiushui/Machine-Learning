import hw5_utils as utils
import numpy as np
import torch
import matplotlib.pyplot as plt

def svm_solver(x_train, y_train, lr, num_iters,
               kernel=utils.poly(degree=1), c=None):
    '''
    Computes an SVM given a training set, training labels, the number of
    iterations to perform projected gradient descent, a kernel, and a trade-off
    parameter for soft-margin SVM.

    Arguments:
        x_train: 2d tensor with shape (n, d).
        y_train: 1d tensor with shape (n,), whose elements are +1 or -1.
        lr: The learning rate.
        num_iters: The number of gradient descent steps.
        kernel: The kernel function.
           The default kernel function is 1 + <x, y>.
        c: The trade-off parameter in soft-margin SVM.
           The default value is None, referring to the basic, hard-margin SVM.

    Returns:
        alpha: a 1d tensor with shape (n,), denoting an optimal dual solution.
               Initialize alpha to be 0.
               Return alpha.detach() could possibly help you save some time
               when you try to use alpha in other places.

    Note that if you use something like alpha = alpha.clamp(...) with
    torch.no_grad(), you will have alpha.requires_grad=False after this step.
    You will then need to use alpha.requires_grad_().
    Alternatively, use in-place operations such as clamp_().
    '''
    n,d=x_train.size()
    alpha=torch.zeros(n)
    K=kernel(x_train,x_train)
    for i in range(num_iters):
        loss =torch.sum(alpha)- 0.5 * torch.sum(alpha * alpha * y_train * y_train * K)
        loss.backward()
        with torch.no_grad():
            alpha -= lr * alpha.grad
            alpha.clamp_(0, c)
            alpha.requires_grad=False
    return alpha

def svm_predictor(alpha, x_train, y_train, x_test,
                  kernel=utils.poly(degree=1)):
    '''
    Returns the kernel SVM's predictions for x_test using the SVM trained on
    x_train, y_train with computed dual variables alpha.

    Arguments:
        alpha: 1d tensor with shape (n,), denoting an optimal dual solution.
        x_train: 2d tensor with shape (n, d), denoting the training set.
        y_train: 1d tensor with shape (n,), whose elements are +1 or -1.
        x_test: 2d tensor with shape (m, d), denoting the test set.
        kernel: The kernel function.
           The default kernel function is 1 + <x, y>.

    Return:
        A 1d tensor with shape (m,), the outputs of SVM on the test set.
    '''
    pass

def logistic(X, Y, lrate=.01, num_iter=1000):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels

    Returns:
        (d + 1) x 1 FloatTensor: the parameters w
    '''
    n,d=X.size()
    new_X=torch.ones((n,d+1),dtype=torch.float)
    new_X[:,1:d+1]=X
    w=torch.zeros((d+1,1),dtype=torch.float)
    for i in range(num_iter):
        grad=(lrate/n)*(-new_X.T@Y)/(1+torch.exp(Y.T@new_X@w))
        w=w-(grad)
    return w


def logistic_vs_ols():
    '''
    Returns:
        Figure: the figure plotted with matplotlib
    '''
    # utils.svm_contour(predict_linear)
    utils.svm_contour(predict_logistic)
    pass



X,Y=utils.load_logistic_data()
def linear_normal(X, Y):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels

    Returns:
        (d + 1) x 1 FloatTensor: the parameters w
    '''
    n,d=X.size()
    new_X=torch.ones((n,d+1),dtype=torch.float)
    new_X[:,1:d+1]=X
    return new_X.pinverse()@Y
w_linear=linear_normal(X,Y)
w_logistic=logistic(X,Y,num_iter=1000)
def predict_linear(X):
    n,d=X.size()
    new_X=torch.ones((n,d+1),dtype=torch.float)
    new_X[:,1:d+1]=X
    return new_X@w_linear
def predict_logistic(X):
    n,d=X.size()
    new_X=torch.ones((n,d+1),dtype=torch.float)
    new_X[:,1:d+1]=X
    return  1/(1+torch.exp(-new_X@w_logistic))

logistic_vs_ols()