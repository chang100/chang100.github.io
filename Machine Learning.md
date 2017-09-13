#Machine Learning

<center><img src="mlicon.png" style="background-size:cover"></center>







Deep learning is a subset of machine learning. To understand Deep Learning, it is important to understand the fundamentals of machine learning. These notes are meant to be highlights of the chapter on Machine Learning Basics in *Deep Learning* by Goodfellow, Bengio, and Courville.

## Learning Algorithms

###Task

There are many different types of learning algorithms including:

- **Classification:** Computer program is asked to **classify** an input as one of $k$ categories. For example, modern image recognition algorithms such as AlexNet attempt to classify images as one out of 1000 categories.

- **Regression:** Computer program is asked to **predict** a numerical value given some input. For example, given the square footage and number of bathrooms in a house, predict the per month rent.

- **Transcription:** Computer program observes a relatively unstructured representation of some kind of data and asked to transcribe the information into discrete textual form. For example, converting speech to text.

- **Machine Translation:** Computer program is given a sequence of symbols in some language, and attempts to convert the sequence of symbols in another language.


### Performance Measure

We use quantitative metrics to evaluate the abilities of a machine learning algorithm.

For tasks such as classification, we can often utilize the **accuracy** (or equivalently, the misclassification rate) of the model. The misclassification rate is often referred to as the 0-1 loss. In some cases, it is inappropriate to use pure accuracy as a metric such as when the data is not uniformly distribution among all classes.

The performance measure is usually evaluated on an unseen **test set** of data that is separate from the **training data** as this is a better way to evaluate how the model performs when it is deployed in the real world.

### Experience

Machine learning algorithms are often categorized into supervised or unsupervised learning algorithms.

**Unsupervised learning algorithms** learn useful properties form the structure of this dataset. Each data point in the dataset, however, does not have a label associated with it. Examples of unsupervised algorithms include k-means, word2vec, and autoencoders.

**Supervised learning algorithms** train on a dataset in which each data point contains a label. Examples of supervised learning algorithms include image classification, linear regression, and many more examples.



## Capacity, Overfitting, and Underfitting

One challenge in machine learning is to ensure that our model **generalizes** well. That is to say, the model can perform well on inputs that it has not been trained on.

Typically, a machine learning model is trained over a training set and performance metrics are calculated on the training set which we summarize as the **training error**. While the model may perform well on this training set, it may perform poorly on the unseeen test set. This is known as **overfitting**. On the other hand, **underfitting** occurs when the model does not obtain a low **training error**.

**Model capacity** refers to a model's ability to fit a wide variety of functions. By decreasing a model's capacity, a model will tend to underfit. Model capacity can be quantified using **Vapnik-Chervonenkis (VC) dimension** which is the largest value of $m$ such that the classifier can correctly label a training set of $m$ labels. 

### Regularization

Suppose we have some learning algorithm which will learn one function in its hypothesis space. By introducing **regularization**, we can create a predisposition for the learning algorithm such that it is more likely to learn one function rather than another. 

One simple example is **weight decay** where we modify the loss function. Suppose we have weights $\mathbf{w}$, and our loss function is originally defined as
$$
J(\mathbf{w}) = \text{Loss}
$$
We can add a $L_2$ loss term to our loss function to create a preference for smaller values of weights which may help prevent overfitting.
$$
J(\mathbf{w}) = \text{Loss} + \lambda \mathbf{w}^\top \mathbf{w}
$$


## Hyperparameters and Validation Sets

Previously, we mentioned that there was a **training set** to train on and a separate **test set** to evaluate performance metrics of our model. Suppose we are evaluating multiple different models and want to choose the best model. In theory, we do not want to use any data from our test set in our choice of evaluating the best model. For this reason, we create a separate **validation set** from the training data. We do not train our model on data from the validation set but only use it to evaluate different models. 

### Cross-Validation	

Dividing the dataset up into a training and testing set is problematic if the dataset is too small because an overly small test set results in statistical uncertainty. This is where the **k-fold cross-validation** procedure comes into play. The dataset is split into $k$ different subsets. We perform $k$ trials, each time selecting a different one of the $k$ different subsets as the test set and the rest of the data as training set. The test error is then estimated by the average test error across the $k$ trials.



## Bias and Variance

The **bias** of a model refers to erroneous assumptions in the learning algorithm. High bias can cause the algorithm to miss relevant patterns/features.

The **variance** of a model refers to sensitivity to small fluctuations in the training set. High variance can cause the algorithm to model the noise in the training data.

Often times, there is a **bias-variance tradeoff**. If our model was extremely simple, then it may have large bias (but small variance). If our model increases in complexity, it may have small bias (but large variance).



## Supervised Learning Algorithms

### Probabilistic Supervised Learning

Many supervised learning algorithms are trained to estimate a probability distribution $p(y | x)$ where $x$ is our inputs. For example, **logistic regression** trains weights $\theta$ and models:
$$
p(y=1 | x;\theta ) = \sigma(\theta^\top x), \text{ where} \\
\sigma(x)=\frac{1}{1+e^{-x}}
$$
To train a logistic regression model, we generally use **gradient descent** to search for the optimal weights.

### Support Vector Machines

Support Vector Machines are an easy to use, simple, out of the box supervised learning algorithm.

A support vector machine trains parameters $\theta$ and $b$ and predicts one class when $\theta^\top x + b > 0$ and the other class when $\theta^\top x + b < 0$. Note that unlike logistic regression, a SVM does not output class probabilities.

One inspiration associated SVMs is the **kernel trick**. Before applying our SVM, we utilize a function $\phi(x)$ to transform our input then training the parameters $\theta$ and $b$ to train a linear separator on our transformed inputs. The power in the kernel trick is that it allows our model to create a nonlinear separator for our inputs.



## Unsupervised Learning Algorithms

### Principal Component Analysis

- PCA learns a representation of the data which is lower dimension than original

- learns an orthogonal linear transformation of the data that projects input x into a representation z

- Find $z=W^\top x$. 
  $$
  \begin{eqnarray}
  X &=& U\Sigma W^\top \\
  X^\top X &=& (U\Sigma W^\top)^\top U\Sigma W^\top \\
  &=& W \Sigma^2 W^\top
  \end{eqnarray}
  $$

- PCA transforms data into a representation where the elements are mutually uncorrelated

### K-means Clustering

- Initialize k different centroids $\{ \mu_1, \mu_2, \cdots, \mu_k \}$. Alternate these two steps until convergence
  - Each training example is assigned to cluster $i$ that is closest
  - Cluster means are recomputed

## Stochastic Gradient Descent

- Original gradient descent:

  - If we have $m$ data points in our training set, 

  $$
  \begin{eqnarray}
  J(\theta) &=& \frac{1}{m} \sum_{i=1}^m L(x^{(i)}, y^{(i)}, \theta) \\
  \nabla J(\theta) &=& \frac{1}{m} \sum_{i=1}^m \nabla L(x^{(i)}, y^{(i)}, \theta) \\
  \end{eqnarray}
  $$

  - Computational cost of this operation is $O(m)$.  

- The insight of SGD is that the gradient is an expectation. The expectation may be approximately estimated using a small set of samples. Specifically, on each step of the algorithm, we can sample a minibatch of examples.

## Challenges Motivating Deep Learning

- **Curse of dimensionality:** Many machine learning problems become exceedingly difficult when the number of dimensions in the data is high
- **Local Constancy and Smoothness Regularization:** Machine learning algorithms need to be guided by prior beliefs about what kind of function they should learn. Among the most widely used of these implicit "priors" is the smoothness prior. This states that the function we learn should not change very much within a small region
- **Manifold Learning:** A manifold is a connected region or a set of points associated with a neighborhood around each point.