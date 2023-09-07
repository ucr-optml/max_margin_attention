import numpy as np
import numpy.linalg as npl
import torch
import torch.nn as nn
from cvxopt import matrix
from cvxopt import solvers
from sklearn.svm import LinearSVC


def p_svm_solver(X, W, ids, zero_ids=None):
    n, T, d = X.shape
    X_ = X.clone() @ W
    for i in range(X_.shape[0]):
        temp = X_[i,0].clone()
        X_[i,0] = X_[i,ids[i]].clone()
        X_[i,ids[i]] = temp
    X_ = X_.detach().numpy()

    P = matrix(np.eye(d))
    q = matrix(np.zeros(d))
    h = matrix(-np.ones(int(X_.shape[0]*(T-1))))
    if zero_ids is not None:
        for i in zero_ids:
            h[i*(T-1):(i+1)*(T-1)] = 0
    G = np.zeros((int(X_.shape[0]*(T-1)), d))
    for i in range(X_.shape[0]):
        for t in range(1,T):
            G[int(i*(T-1))+t-1] = X_[i,t] - X_[i,0]
    G = matrix(G)
    sol = solvers.qp(P,q,G,h)['x']
    return sol, X_

def w_svm_solver(X, Y, ids):
    WFirst=np.zeros((X.shape[0],X.shape[-1]))
    for i in range(X.shape[0]):
        WFirst[i]=X[i,ids[i]]

    svm = LinearSVC(fit_intercept=False, C=1e9)
    svm.fit(WFirst, Y)
    c0=svm.coef_[0]/npl.norm(svm.coef_[0])
    return c0

class PromptAttn(nn.Module):
    
    def __init__(self, input_size, identity_W=True):
        super(PromptAttn, self).__init__()
        
        if identity_W:
            self.query = nn.Identity()
            self.key = nn.Identity()
        else:
            self.query = nn.Linear(input_size, input_size, bias=False)
            self.key = nn.Linear(input_size, input_size, bias=False)

        self.value = nn.Identity()
        self.w = nn.Parameter(torch.randn(input_size) * 0.01)
        self.prompt = nn.Parameter(torch.randn(input_size) * 0.01)

    def forward(self, input_seq):
        prompt_seq = self.prompt.unsqueeze(0).expand(input_seq.size(0), 1, -1)
        
        Q = self.query(prompt_seq)
        K = self.key(input_seq)
        V = self.value(input_seq)
        # A = torch.softmax(Q @ K.transpose(-2, -1) / torch.sqrt(torch.tensor(K.size(-1), dtype=torch.float)), dim=-1)
        A = torch.softmax(Q @ K.transpose(-2, -1), dim=-1)
        self.attention = A[:,0]
        out = A @ V
        return out[:,0] @ self.w