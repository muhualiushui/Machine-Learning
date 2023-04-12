import torch
import hw1_utils as sc
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la

def initial(data,init_c):
    X=data[0]
    Y=data[1]
    s1,s2=init_c
    plt.scatter(X,Y)
    plt.scatter(s1,s2,marker="x",color=["red","blue"],s=100)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    pass
def graph(c1,c2,init_c):
    X1,Y1=c1
    X2,Y2=c2
    s1,s2=init_c
    plt.scatter(X1,Y1,color="red")
    plt.scatter(X2,Y2,color="blue")
    plt.scatter(s1,s2,marker="x",color=["red","blue"],s=100)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    pass

def k_means(X=None, init_c=None, n_iters=50):
    """K-Means.

    Argument:
        X: 2D data points, shape [2, N].
        init_c: initial centroids, shape [2, 2]. Each column is a centroid.
    
    Return:
        c: shape [2, 2]. Each column is a centroid.
    """
    if X is None:
        X, init_c = sc.load_data()
    # initial(X,init_c)
    c1=torch.empty((0,2))
    c2=torch.empty((0,2))
    for i in range(n_iters):    
        print(init_c) 
        d1,d2=step2(X,init_c)        
        if(c1.size()==d1.size() and torch.all(torch.eq(c1, d1))):
            break
        c1,c2=d1,d2
        # graph(c1.T,c2.T,init_c)  
        init_c=step3(c1,c2).view(2,2).T
        # graph(c1.T,c2.T,init_c)   
    print()
    return init_c

def step2(X,init_c):
    data=X.T
    c1=torch.empty((0,2))
    c2=torch.empty((0,2))
    for x in data:
        if la.norm(x-init_c[:,0],2) < la.norm(x-init_c[:,1],2):
            c1=torch.cat((c1,x.unsqueeze(0)), dim=0)
        else:
            c2=torch.cat((c2,x.unsqueeze(0)), dim=0)
    return c1,c2
def step3(c1,c2):
    return torch.cat((sum(c1)/(len(c1)),sum(c2)/(len(c2))),dim=0)
print(k_means())

# X, init_c = sc.load_data()
# c1=torch.empty((0,2))
# c2=torch.empty((0,2))
# print(init_c)
# # for i in range(n_iters):
# d1,d2=step2(X,init_c.T)
# if(c1.size()==d1.size() and torch.all(torch.eq(c1, d1))):
#     print("any")
# c1,c2=d1,d2
# init_c=step3(c1,c2)
# init_c=init_c.view(2,2)
# graph(c1.T,c2.T,init_c)

# Y=np.array([[],[]])
# X=np.array([[-2,-2,2,2],[-1,1,-1,1]])
# print(X.T@X/3)

# X=np.array([[4.5,6],[6,8]])
# eigvalue,eigvector=la.eig(X)
# diagonal = np.diag(eigvalue)
# full_matrix = np.dot(eigvector, diagonal)
# full_matrix = np.dot(full_matrix, np.linalg.inv(eigvector))
# print(full_matrix)



# pseudo-code:
# def K-mean():
#     2D vector: clusters (store points)
#     while loop: loop continue if step2(X,K)(new clusters) not equal to clusters
#         renew clusters: clusters=step2(X,K)
#         renew K: K=step3(clusters) 
#     return K

# def step3(clusters):
#     mean_K(1d vector with size m)= mean of clusters 
#     return mean_K
    
# def step2(X,K):
#     for i in size of X:
#         minimal_distance=Maximun_distance
#         index= m+1
#         for k in size of K:
#             if distance(x(i),u(k))< minimal_distance{
#                 minimal_distance=distance(x(i),u(k))
#                 index=k
#             }
#         put x(i) in clusters at index
#     return clusters
# X=np.array([12,6,20,10])
# diagnal=np.diag(X)
# print(diagnal)
# eigv,eigvv=la.eig(diagnal)
# print(eigvv@diagnal@la.inv(eigvv))