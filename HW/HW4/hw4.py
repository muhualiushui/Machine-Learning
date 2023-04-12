import torch
import hw4_utils as utils
import matplotlib.pyplot as plt
'''
    Important
    ========================================
    The autograder evaluates your code using FloatTensors for all computations.
    If you use DoubleTensors, your results will not match those of the autograder
    due to the higher precision.

    PyTorch constructs FloatTensors by default, so simply don't explicitly
    convert your tensors to DoubleTensors or change the default tensor.

    Be sure to modify your input matrix X in exactly the way specified. That is,
    make sure to prepend the column of ones to X and not put the column anywhere
    else, and make sure your feature-expanded matrix in Problem 3 is in the
    specified order (otherwise, your w will be ordered differently than the
    reference solution's in the autograder).
'''

# Problem 2
def linear_gd(X, Y, lrate=0.01, num_iter=1000):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels
        num_iter (int): iterations of gradient descent to perform

    Returns:
        (d + 1) x 1 FloatTensor: the parameters w
    '''
    n,d=X.size()
    new_X=torch.ones((n,d+1),dtype=torch.float)
    new_X[:,1:d+1]=X
    w=torch.zeros(d+1,dtype=torch.float).unsqueeze(1)
    for i in range(num_iter):
        h=(X@w)
        grad=((1/n)*X.t())@(h-Y)
        w=w-(lrate*grad)
    return w

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

def plot_linear():
    '''
        Returns:
            Figure: the figure plotted with matplotlib
    '''
    X,Y=utils.load_reg_data()
    w=linear_normal(X, Y)
    n,d=X.size()
    New_X=torch.ones((n,d+1),dtype=torch.float)
    New_X[:,1:d+1]=X
    plt.scatter(X,Y)
    plt.plot(X,New_X@w)
    plt.show()
    pass

# Problem 3
def poly_gd(X, Y, lrate=0.01, num_iter=1000):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels
        lrate (float): the learning rate
        num_iter (int): number of iterations of gradient descent to perform

    Returns:
        (1 + d + d * (d + 1) / 2) x 1 FloatTensor: the parameters w
    '''
    n,d=X.size()
    new_X=torch.ones((n,int(1 + d + d * (d + 1) / 2)),dtype=torch.float)
    new_X[:,1:d+1]=X
    index=0
    for i in range(d):
        for j in range(d-i):
            new_X[:,index+d+1]=X[:,i]*X[:,j+i]
            index+=1
    w=torch.zeros((int(1 + d + d * (d + 1) / 2),1),dtype=torch.float)
    for i in range(num_iter):
        h=(new_X@w)
        grad=((1/n)*new_X.t())@(h-Y)
        w=w-(lrate*grad)
    return w

def poly_normal(X,Y):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels

    Returns:
        (1 + d + d * (d + 1) / 2) x 1 FloatTensor: the parameters w
    '''
    n,d=X.size()
    length=int(1 + d + d * (d + 1) / 2)
    new_X=torch.ones((n,length),dtype=torch.float)
    new_X[:,1:d+1]=X
    # new_feature=torch.ones((n,length-d-1),dtype=torch.float)
    index=0
    for i in range(d):
        for j in range(d-i):
            new_X[:,index+d+1]=X[:,i]*X[:,j+i]
            index+=1
    return new_X.pinverse()@Y

def plot_poly():
    '''
    Returns:
        Figure: the figure plotted with matplotlib
    '''
    X,Y=utils.load_reg_data()
    w=poly_normal(X,Y)
    n,d=X.size()
    length=int(1 + d + d * (d + 1) / 2)
    new_X=torch.ones((n,length),dtype=torch.float)
    new_X[:,1:d+1]=X
    index=0
    for i in range(d):
        for j in range(d-i):
            new_X[:,index+d+1]=X[:,i]*X[:,j+i]
            index+=1
    plt.scatter(X,Y)
    plt.plot(X,new_X@w)
    plt.show()
    pass

def poly_xor():
    '''
    Returns:
        n x 1 FloatTensor: the linear model's predictions on the XOR dataset
        n x 1 FloatTensor: the polynomial model's predictions on the XOR dataset
    '''
    X,Y=utils.load_xor_data()
    utils.contour_plot(-1,1,min(Y),max(Y),predict_linear)
    utils.contour_plot(-1,1,min(Y),max(Y),predict_poly)
    pass

X,Y=utils.load_xor_data()
w1=linear_normal(X,Y)
w2=poly_normal(X,Y)

def predict_linear(X):
    n,d=X.size()
    new_X=torch.ones((n,d+1),dtype=torch.float)
    new_X[:,1:d+1]=X
    return new_X@w1

def predict_poly(X):
    n,d=X.size()
    length=int(1 + d + d * (d + 1) / 2)
    new_X=torch.ones((n,length),dtype=torch.float)
    new_X[:,1:d+1]=X
    index=0
    for i in range(d):
        for j in range(d-i):
            new_X[:,index+d+1]=X[:,i]*X[:,j+i]
            index+=1
    return new_X@w2
poly_xor()