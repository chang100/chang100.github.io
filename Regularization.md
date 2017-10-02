# Regularization

A central problem in machine learning is how to make an algorithm that performs well not just on the training data, but also on new inputs.

**Regularization** is defined as any modification we make to a learning algorithm that is intended to reduce its generalization error but not its training error. Regularization of an estimator works by trading increased bias for reduced variance. 

## Parameter Norm Penalties

- Create new cost function $J'(\theta,X,y) = J(\theta; X, y) + \alpha \Omega(\theta)$, where $\alpha \in [0, \infty)$
- Usually, for affine layers, weights of affine transform are regularized and biases are unregularized
- Denote $\theta$ as all parameters and $w$ as the terms we want to regularize

### $L^2$ Parameter Regularization 

- Drives weights closer to the origin by adding the regularization term $\Omega(\theta) = \frac{1}{2}||w||_2^2$

- $L^2$ regularization is also known as **weight decay**, **ridge regression**, and **Tikhonov regularization**

- Case study: truly quadratic objective function (linear regression model with mean squared error):

  - $\hat J(\theta) = J(w^*) + \frac{1}{2}(w - w^*)^\top H(w-w^*)$, where $H$ is the hessian matrix of $J$, and $w^* = \text {argmin}_w J(w)$

  - Let $\widetilde{w}$ denote denote the minimum of the regularized function. Then, 
    $$
    \begin{eqnarray}
    \alpha \widetilde{w} + H(\widetilde{w} - w^*) &=& 0 \\
    (H + \alpha I) \widetilde{w} &=& Hw^* \\
    \widetilde{w} &=& (H+\alpha I)^{-1} Hw^*\\
    &=& (Q\Lambda Q^\top + \alpha I)^{-1} Q\Lambda Q^\top w^*  \\
    &=& Q(\Lambda + \alpha I) Q^\top \Lambda Q^\top w^*
    \end{eqnarray}
    $$

  - This rescales $w^*$ along the axes defined by the eigenvectors of $H$ such that the component of $w^*$ aligned with the $i$th eigenvector is $\frac{\lambda_i}{\lambda_i + \alpha}$

    ​

### $L^1$ Regularization

- While $L^2$ weight decay is most common, there are other ways to penalize the size of the model parameters such as $L^1$ regularization, where $\Omega(\theta) = ||w||_1 =\sum_i|w_i|$
- Effect:

$$
\begin{eqnarray}
\tilde J(w;X,y) &=& \alpha||w||_1 + J(w; X,y) \\
\nabla_w \tilde J(w;X,y) &=& \alpha \text{sign}(w) + \nabla_w J(X,y;w) \\
\end{eqnarray}
$$

- ​
- ​

