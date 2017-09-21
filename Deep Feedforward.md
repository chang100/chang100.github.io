# Deep Feedforward Networks

## Gradient-Based Learning

- Stochastic gradient descent applied to nonconvex loss functions has no such convergence guarantee and is sensitive to the values of the initial parameters. 

#### Cost Functions

- Cost function is usually **cross-entropy loss** between training data and the model distribution
- The gradient of the cost function must be large and predictable enough to serve as a good guide for the learning algorithm.
- One unusual property of the cross-entropy loss is that it does not have a minimum value when applied to models commonly used in practice.

### Output units

- Utilize softmax function:
  $$
  \text{softmax}(\bf{z})_i = \frac{\exp(z_i)}{\sum_j \exp(z_j)}
  $$






## Hidden Units

- **ReLU:** Rectified Linear Unit. $f(x) = \max(0,x)$
  - **Absolute value rectification:** $f(x) = \max(0,x) - \min(0,x) = |z|$
  - **Leaky ReLU:** $f(x) = \max(0,x) - \alpha_i \min(0,x)$ where $\alpha_i$ is a small positive value
  - **Parametric ReLU:** $f(x) = \max(0,x) - \alpha_i \min(0,x)$, where $\alpha_i$ is trainable
  - **Maxout Unit:** Divide input into groups of $k$ values. Each maxout unit then outputs the maximum element of one of these groups. $f(x)_i = \max_{j\in G^{(i)}} z_j$.
- **Sigmoid** and **tanh**
  - Note that $\tanh(z) = 2\sigma(2z) - 1$
  - When a sigmoidal activation function must be used, the hyperbolic tangent activation function typically performs better than the logistic sigmoid
- Other hidden units
  - **Radial basis function:** $h_i =\exp\left(-\frac{1}{\sigma_i^2} ||W_{:,i} - x || ^2\right)$. As $x$ approaches a template $W_{:,i}$, it saturates to 0 for most $x$ which makes it difficult to optimize. 
  - **Softplus:** $f(x) = \log(1+e^x)$ . Use of softplus is generally discouraged. The softplus demonstrates that the performance of hidden unit types can be very counterintuitive - softplus should have an advantage over the rectifier due to being differentiable everywhere but empirically it does not
  - **Hard Tanh:** $f(x) = \max(-1, \min(1,a))$

## Architecture Design

### Universal Approximation

- **Universal approximation theorem:** A feedforward network with a linear output layer and at least one hidden layer with any "squashing" activation function can approximate any Borel measurable function from one finite-dimensional space to another with any desired nonzero amount of error, provided that the network is given enough hidden untis
- Montufar et al. states that the number of linear regions carved out by a deep rectifier network with $m$ inputs, $d$ depth, and $n$ units per hidden layer is $O({n \choose m}^{m(d-1)} n^m)$.
- Empirically, greater depth does seem to result in better generalization for a wide variety of tasks

## Back Prop and Other Differentiation Algorithms

- During training, forward propagation can continue onward until it produces a scalar cost $J(\theta)$. 
- **Back-propagation** allows the information from the cost to then flow backward through the network in order to compute the gradient
- Calculus chain rule is applied recursively to obtain backprop
- **Krylov methods**: A set of iterative techniques for performing various operations, such as approximately inverting a matrix or finding approximations to its eigenvectors or eigenvalues



