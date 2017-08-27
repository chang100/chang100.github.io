#Machine Learning

<center>!<img src="mlicon.png" style="background-size:cover"></center>







Deep learning is a subset of machine learning. To understand Deep Learning, it is important to understand the fundamentals of machine learning. These notes are meant to be highlights of the chapter on Machine Learning Basics in *Deep Learning* by Goodfellow, Bengio, and Courville.

## Learning Algorithms

###Task

There are many different types of learning algorithms including:

- **Classification:** Computer program is asked to **classify** an input as one of $k$ categories. For example, modern image recognition algorithms such as AlexNet attempt to classify images as one out of 1000 categories.

- **Regression:** Computer program is asked to **predict** a numerical value given some input. For example, given the square footage and number of bathrooms in a house, predict the per month rent.

- **Transcription:** Computer program observes a relatively unstructured representation of some kind of data and asked to transcribe the information into discrete textual form. For example, converting speech to text.

- **Machine Translation:** Computer program is given a sequence of symbols in some language, and attempts to convert the sequence of symbols in another language.

  ​

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





