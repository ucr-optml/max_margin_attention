# Margin Maximization in Attention Mechanism
This repository holds the official code for the paper [Max-Margin Token Selection in Attention Mechanism](https://arxiv.org/abs/2306.13596)

### Abstract
Attention mechanism is a central component of the transformer architecture which led to the phenomenal success of large language models. We explore the seminal softmax-attention model 
$$f(X)=v^\top X^\top \texttt{softmax}(X W^\top p),$$
where, $X$ is the tokenized input, $v$ is the value weights, $W$ is the key-query weights, and $p$ is a tunable token/prompt. We show that running gradient descent on $p$, or equivalently $W$, converges to a max-margin solution that separates locally-optimal tokens from non-optimal ones. When optimizing $v$ and $p$ simultaneously with logistic loss, we identify conditions under which the regularization paths converge to their respective max-margin solutions where $v$ separates the input features based on their labels.  

### Experimental Details
We implemented an attention layer using PyTorch. During training, we utilized the SGD optimizer with a learning rate of 0.1 and trained the model for 1000 iterations. To visualize the generalization path more effectively, we normalized the gradient of the variables $p$ and $v$ at each iteration.

Once we obtained the solution, denoted as $\hat{p}$ (p-hat), we determined the locally optimal indices as those with the highest softmax scores. To solve the optimization problem described in equation (attnsvm), we used the Python package cvxopt. By solving this problem, we obtained the solution 𝑝. Subsequently, we verified that these indices satisfied our local-optimal definition.

In our paper, the examples we used were constructed in such a way that it was straightforward to verify their optimality. In the joint optimization of $(p,v)$, the v-optimization was solved using the Python package sklearn.svm.

### Requirements

```
torch
cvxopt
sklearn
```

### Reproducing Results 

- Local vs global convergence for fixed $v$:

  - *single_input_converge.ipynb*: Visualization of gradient iteration path when there is single input. Set ```g_converge=True``` to perform global convergence and ```g_converge=False``` for local convergence.
  - *multi_input_converge.ipynb*: Visualization of gradient iteration path considering multiple inputs and global convergence. 

- Jointly training $v$ and $p$:

  - *joint_training.ipynb*: Visualization of gradient iteration paths for both $v$ and $p$. Set ```tight_converge=True``` to perform scenario where all inputs are support vectors and ```tight_converge=False``` to the contrary.

- Comparison for different loss functions: 

  - *diff_loss_function.ipynb*: Implementation of different settings when loss functions are: $\ell(x)=-x$ or $\ell(x)=\log(1+e^{-x})$.
