# Cross-Entropy Loss

When I created my first neural net using numpy, I learned about Softmax Cross-Entropy Loss. The softmax portion made intuitive sense to me: the outputs of the softmax layer represented the probability of each class. From afar, the cross entropy portion also made sense: it was the negative log probability of the correct class. However, I was confused why we didn't use something simple such as Mean-Squared-Error loss which is what was used for linear regression, one of the first examples we covered in CS229, Stanford's Machine Learning class. As such, I took a deeper dive to investigate why we used cross entropy loss.

*<center>Given a final hidden state* $h_\text{final}$, *the predictions,* $\hat{y}$ *and the loss are computed as follows:* </center> 
$$
\begin{eqnarray}
\hat{y} &=& \text{softmax}(h_\text{final}) \\
\text{Loss} &=& \text{CE}(y, \hat{y}) = H(y, \hat{y}) -\sum_i y_i \log \hat{y}_i
\end{eqnarray}
$$

### Kullback-Leibler (KL) Divergence

Kullback-Leibler Divergence is defined as:
$$
D_\text{KL} (P||Q) = \mathbf{E}_{x\sim P} \left[\log \frac{P(x)}{Q(x)}\right] = \mathbf{E}_{x\sim P} \left[\log P(x) - \log Q(x) \right]
$$
The simple description of KL divergence is that it measures the difference between two distributions, $P$ and $Q$ with $D_\text{KL}(P||Q) = 0$ if $P$ and $Q$ are the same distribution.  It is sometimes thought of as the *distance* between the two distributions although this is not quite true as the KL divergences is asymmetric. I.e.$D_\text{KL}(P||Q) \neq D_\text{KL}(Q||P$).  



As it turns out, KL divergence and cross-entropy loss are closely related. With entropy function $H$, the cross-entropy can be written as:
$$
H(y, \hat{y}) = H(y) + D_{KL}(y, \hat{y})
$$
During the training of our neural net, we are attempting solve $\text{argmin}_{\hat{y}} H(y,\hat{y}) $. We can derive that minimizing the cross entropy loss minimizes the KL divergence between $y$ and $\hat{y}$ which minimizes the difference in distribution between the ground truth labels and our predicted labels.



### Information Theory Perspective

It was also interesting to see Cross-Entropy Loss in an information theory perspective. The basic intuition behind information theory is that unlikely events such as "The sky is purple" are weighted higher than common events such as "The sky is blue". In Deep Learning by Goodfellow et al., the authors enumerate three rules:

* Likely events should have low content. Events that are guaranteed to occur are given no information content.

* Less likely events should have high information content.

* Independent events have additive information.

  ​

As such, to satisfy all three properties, the information of an event $x$ is the negative log likelihood of the event happening:
$$
I(x) = -\log P(x)
$$
In the case of classification with a one-hot ground truth label, the cross entropy loss is equivalent to the information associated with our output probability distribution. Pretty neat!





