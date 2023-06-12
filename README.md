# Margin Maximization in Attention Mechanism
This repository holds the official code for the paper [Margin Maximization in Attention Mechanism]()

### 🦸‍ Abstract
Attention mechanism is a central component of the transformer architecture which led to the phenomenal success of large language models. However, the theoretical principles underlying the attention mechanism are poorly understood, especially its nonconvex optimization dynamics. In this work, we explore the seminal softmax-attention model 
$$f(X)=v^\top X^\top \texttt{softmax}(X W^\top p),$$
where, $X$ is the tokenized input, $v$ is the value weights, $W$ is the key-query weights, and $p$ is a tunable token/prompt. We prove that running gradient descent on $p$, or equivalently $W$, converges to a max-margin solution that separates locally-optimal tokens from non-optimal ones. We also develop regularization path analysis that generalizes these findings to nonlinear classifier head -- rather than linear $v$. When optimizing $v$ and $p$ simultaneously with logistic loss, we identify conditions under which the regularization paths converge to their respective max-margin solutions where $v$ separates the input features based on their labels. Finally, we verify our results through numerical insights. 
