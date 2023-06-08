# Normalization in Deep Neural Networks

Reference:

[1] Ioffe, S. &amp; Szegedy, C.. (2015). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. *Proceedings of the 32nd International Conference on Machine Learning*, in *Proceedings of Machine Learning Research* 37:448-456 Available from [here](https://proceedings.mlr.press/v37/ioffe15.html).

[2] Kevin Zakka's Blog. [*Deriving the Gradient for the Backward Pass of Batch Normalization*](https://kevinzakka.github.io/2016/09/14/batch_normalization/).

[3] Santurkar, S., Tsipras, D., Ilyas, A., & Madry, A. (2018). How does batch normalization help optimization? *Advances in neural information processing systems*, 31. Available from [here](https://papers.nips.cc/paper_files/paper/2018/hash/905056c1ac1dad141560467e0a99e1cf-Abstract.html).

## Input-Level Normalization

One issue of deep neural nets is that the learning capability of the network deteriorates as the network goes deeper due to vanishing or exploding gradients. Adding normalization layers provides a common remedy to this kind of problem by *standardizing the statistics of the hidden units* (zero mean & unit variance, also known as *whitening*). Specifically, for input feature set $\{ \mathbf{x}_i \}_{i = 1}^N$ with dimension $D$, i.e., $\mathbf{x}_i \in \mathbb{R}^D$, we have the "standardization" of the input features on dimension $j$ ($j$-th feature) can be represented as (the scaled feature will have zero mean and unit variance),

$$
\hat{x}_{i,j} = \frac{x_{i,j} - \mu_j}{\sigma_j},
$$

where,

$$
\begin{aligned}
\mu_j &= \frac{1}{N} \sum_{i = 1}^N x_{j, i} \\
\sigma_j^2 &= \frac{1}{N} \sum_{i = 1}^N (x_{i, j} - \mu_j)^2
\end{aligned}
$$

Normalization also helps reduce the impact from features of different scales, as we are applying the same learning rate for all weight parameters, so large parameters will dominate the weight updates while using stochastic gradient descent algorithms.

## Extending Input Normalization to the Hidden Layers --- Batch Normalization

However, normalizing the inputs only affects the first hidden layer, to standardize deeper hidden layer activations, one approach is called the **Batch Normalization**, which ensures the distribution of the hidden layer activations for each layer has a zero mean and unit variance. Some key properties of the batch normalization,

- Normalizes hidden layer inputs.
- Helps with exploding/vanishing gradient problems.
- Can increase training stability and convergence rate.
- Can be understood as additional (normalization) layers (with additional parameters).

When the input distribution to a learning system changes, it is said to experience *covariate shift*, furthermore, the idea of covariate shift can be extended to part of the learning system, such as a layer of the network. Consider the computation on the second layer of a neural network with input $u$

$$
\ell = F_2 (F_1 (u, \Theta_1), \Theta_2),
$$

where $F_1$ and $F_2$ are arbitrary network activation functions, and the parameters $\Theta_1, \Theta_2$ are to be learned to minimize the loss $\ell$. While using Stochastic Gradient Descent (SGD) methods, the training proceeds in minibatches, i.e., for the training set $\{ \mathbf{x}_i\}_{i=1}^N$, at each training step we consider a minibatch $\{ \mathbf{x}_i \}_{i=1}^m$, denote the learning rate as $\alpha$, we then have for a gradient descent step, the parameter update rule for $\Theta_2$ can be expressed as,

$$
\Theta_2 \gets \Theta_2 - \frac{\alpha}{m} \sum_{i=1}^m \frac{\partial F_2 (\mathbf{x}_i, \Theta_2)}{\partial \Theta_2}
$$

### Understanding Internal Covariate Shift

Internal Covariate Shift refers to scenario when the layer input distribution changes ("feature shift" in hidden layers), however there's no guarantee or strong evidence shows that BatchNorm actually helps with that.

## Chain Rule Primer

Suppose we have a function $u(x, y)$ where $x(r, t)$ and $y(r, t)$ are also two functions. Then to compute $\frac{\partial u}{\partial r}$ and $\frac{\partial u}{\partial t}$ we apply the chain rule,

$$
\begin{aligned}
\frac{\partial u}{\partial r} &= \frac{\partial u}{\partial x} \cdot \frac{\partial x}{\partial r} + \frac{\partial u}{\partial y} \cdot \frac{\partial y}{\partial r} \\
\frac{\partial u}{\partial t} &= \frac{\partial u}{\partial x} \cdot \frac{\partial x}{\partial t} + \frac{\partial u}{\partial y} \cdot \frac{\partial y}{\partial t}
\end{aligned}
$$

## Batch Normalization: Normalization via Mini-Batch Statistics

Consider a mini-batch $\mathcal{B} = \{ \mathbf{x}_i \}_{i = 1}^m$, denote the normalized values as $\{ \hat{\mathbf{x}}_i\}_{i=1}^m$ and the corresponding linear transformation as $\{\mathbf{y}_i\}_{i=1}^m$, we refer to the transform,
$$\mathtt{BN}_{\boldsymbol{\gamma}, \boldsymbol{\beta}}: \{ \mathbf{x}_i \}_{i = 1}^m \to \{\mathbf{y}_i\}_{i=1}^m$$
as the *Batch Normalization Transform*, which can be performed in the following steps,
> **procedure 1**
> 
> **Input**: values of $\mathbf{x}$ over a mini-batch: $\mathcal{B} = \{ \mathbf{x}_i \}_{i = 1}^m$;
>
> **Learnable parameters**: $\boldsymbol{\gamma}$, $\boldsymbol{\beta}$
>
> **Output**: $\{ y_i = \mathtt{BN}_{\boldsymbol{\gamma}, \boldsymbol{\beta}} (\mathbf{x}_i) \}_{i=1}^m$.
>
> $\boldsymbol{\mu}_\mathcal{B} \gets \frac{1}{m} \sum_{i=1}^m \mathbf{x}_i$ // mini-batch mean
>
> $\boldsymbol{\sigma}_\mathcal{B}^2 \gets \frac{1}{m} \sum_{i=1}^m (\mathbf{x}_i - \mu_\mathcal{B})^2$ // mini-batch variance
>
> $\hat{\mathbf{x}}_i \gets$ // normalization / whitening
>
> $\mathbf{y}_i \gets \boldsymbol{\gamma} \odot \hat{\mathbf{x}}_i + \boldsymbol{\beta} \equiv \mathtt{BN}_{\boldsymbol{\gamma}, \boldsymbol{\beta}} (\mathbf{x}_i)$ // scale and shift

where $\epsilon$ is a small constant for numerical stability, say $1e-5$ and $\odot$ denotes the Hadamard product. Note that here the parameter $\boldsymbol{\gamma}$ controls the spread or scale and the parameter $\boldsymbol{\beta}$ controls the mean, which also makes the original bias terms in the network redundant. Since the transformation is differentiable, we can easily pass the gradients back to the input of the layer and to the batch normalization parameters $\boldsymbol{\gamma}$ and $\boldsymbol{\beta}$. Specifically, we have during the backpropagation (not simplified),

$$
\begin{aligned}
\frac{\partial \ell}{\partial \hat{\mathbf{x}}_i} &= \frac{\partial \ell}{\partial \mathbf{y}_i} \odot \boldsymbol{\gamma} \\
\frac{\partial \ell}{\partial \boldsymbol{\sigma}_\mathcal{B}^2} &= \sum_{i=1}^m \frac{\partial \ell}{\partial \hat{\mathbf{x}}_i} \cdot (\mathbf{x}_i - \boldsymbol{\mu}_\mathcal{B}) \cdot \left(-\frac{1}{2} \right) (\boldsymbol{\sigma}_\mathcal{B}^2 + \epsilon)^{-3/2} \\
\frac{\partial \ell}{\partial \boldsymbol{\mu}_\mathcal{B}} &= \left(\sum_{i=1}^m \frac{\partial \ell}{\partial \hat{\mathbf{x}}_i} \cdot \frac{-1}{\sqrt{\boldsymbol{\sigma}_\mathcal{B}^2 + \epsilon}}\right) + \frac{\partial \ell}{\partial \boldsymbol{\sigma}_\mathcal{B}^2} \cdot \frac{\sum_{i=1}^m -2(\mathbf{x}_i - \boldsymbol{\mu}_\mathcal{B})}{m} \\
\frac{\partial \ell}{\partial \mathbf{x}_i} &= \frac{\partial \ell}{\partial \hat{\mathbf{x}}_i} \cdot \frac{1}{\sqrt{\boldsymbol{\sigma}_\mathcal{B}^2 + \epsilon}} + \frac{\partial \ell}{\partial \boldsymbol{\sigma}_\mathcal{B}^2} \cdot \frac{2 (\mathbf{x}_i - \boldsymbol{\mu}_\mathcal{B}^2)}{m} + \frac{\partial \ell}{\partial \boldsymbol{\mu}_\mathcal{B}} \cdot \frac{1}{m} \\
\frac{\partial \ell}{\partial \boldsymbol{\gamma}} &= \sum_{i=1}^m \frac{\partial \ell}{\partial \mathbf{y}_i} \cdot \hat{\mathbf{x}}_i\\
\frac{\partial \ell}{\partial \boldsymbol{\beta}} &= \sum_{i=1}^m \frac{\partial \ell}{\partial \mathbf{y}_i}
\end{aligned}
$$

### Derivations

We first express the forward pass in the diagram below, for a mini-batch $\mathcal{B} = \{ \mathbf{x}_i \}_{i=1}^m$, we have

$$
\mathbf{x} \to \hat{\mathbf{x}}(\boldsymbol{\mu}_\mathcal{B}, \boldsymbol{\sigma}_\mathcal{B}^2, \mathbf{x}) \to \mathbf{y}(\hat{\mathbf{x}}, \boldsymbol{\gamma}, \boldsymbol{\beta}) \to \ell(\bold{y}),
$$

hence the backward pass can be expressed as,

$$
\ell (\mathbf{y}) \to \mathbf{y}(\hat{\mathbf{x}}, \boldsymbol{\gamma}, \boldsymbol{\beta}) \to \hat{\mathbf{x}} (\boldsymbol{\mu}_\mathcal{B}, \boldsymbol{\sigma}_\mathcal{B}^2, \mathbf{x})
$$

First we have $\frac{\partial \ell}{\partial \mathbf{y}_i}$ available as the upstream derivative, which is given in the function parameter.

Then to compute $\mathbf{y}(\hat{\mathbf{x}}, \boldsymbol{\gamma}, \boldsymbol{\beta})$, which consists of three variables, we first compute the gradients with respect to each variable,

$$
\begin{aligned}
    \frac{\partial \ell}{\partial \bold{\gamma}} &= \frac{\partial \ell}{\partial \mathbf{y}_i} \cdot \frac{\partial \mathbf{y}_i}{\partial \boldsymbol{\gamma}} \\
    &= \sum_{i=1}^m \frac{\partial \ell}{\partial \mathbf{y}_i} \cdot \hat{\mathbf{x}}_i,
\end{aligned}
$$

the summation indicates the computation is for the batches. As for $\boldsymbol{\beta}$ we have,

$$
\begin{aligned}
    \frac{\partial \ell}{\partial \boldsymbol{\beta}} &= \frac{\partial \ell}{\partial \mathbf{y}_i} \cdot \frac{\partial \mathbf{y}_i}{\partial \boldsymbol{\beta}} \\
    &= \sum_{i=1}^m \frac{\partial \ell}{\partial \mathbf{y}_i},
\end{aligned}
$$

and finally $\hat{\mathbf{x}}_i$,

$$
\begin{aligned}
    \frac{\partial \ell}{\partial \hat{\mathbf{x}}_i} &= \frac{\partial \ell}{\partial \mathbf{y}_i} \cdot \frac{\partial \mathbf{y}_i}{\partial \hat{\mathbf{x}}_i} \\
    &= \frac{\partial \ell}{\partial \mathbf{y}_i} \odot \boldsymbol{\gamma}
\end{aligned}
$$

To compute the gradient with respect to $\mathbf{x}_i$, we have since $\boldsymbol{\mu}$ and $\boldsymbol{\sigma}^2$ is a function of $\boldsymbol{\mu}$,

$$
\frac{\partial \ell}{\partial \bold{\mu}_\mathcal{B}} = \frac{\partial \ell}{\partial \hat{\mathbf{x}}_i} \cdot \frac{\partial \hat{\mathbf{x}}_i}{\partial \boldsymbol{\mu}_\mathcal{B}} + \frac{\partial \ell}{\partial \boldsymbol{\sigma}_\mathcal{B}^2} \cdot \frac{\partial \boldsymbol{\sigma}^2_\mathcal{B}}{\partial \boldsymbol{\mu}_\mathcal{B}},
$$

we have,

$$
\frac{\partial \hat{\mathbf{x}}_i}{\partial \boldsymbol{\mu}_\mathcal{B}} = -\frac{1}{\sqrt{\boldsymbol{\sigma}_\mathcal{B}^2 + \epsilon}},
$$

and,

$$
\frac{\partial \boldsymbol{\sigma}_\mathcal{B}^2}{\partial \boldsymbol{\mu}_\mathcal{B}} = -\frac{1}{m} \sum_{i=1}^m 2 \cdot (\boldsymbol{x}_i - \boldsymbol{\mu}_\mathcal{B}).
$$

Next we compute the partial,

$$
\begin{aligned}
\frac{\partial \ell}{\partial \boldsymbol{\sigma}_\mathcal{B}^2} &= \frac{\partial \ell}{\partial \hat{\mathbf{x}}} \cdot \frac{\partial \hat{\mathbf{x}}}{\partial \boldsymbol{\sigma}_\mathcal{B}^2} \\
&= \frac{\partial \ell}{\partial \hat{\mathbf{x}}} \cdot \left( - \frac{1}{2} \sum_{i=1}^m \frac{\mathbf{x}_i - \boldsymbol{\mu}_\mathcal{B}}{(\boldsymbol{\sigma}_\mathcal{B}^2 + \epsilon)^{3/2}} \right).
\end{aligned}
$$

Thus we have,

$$
\begin{aligned}
    \frac{\partial \ell}{\partial \boldsymbol{\mu}_\mathcal{B}} &= \frac{\partial \ell}{\partial \hat{\mathbf{x}}} \cdot \frac{\partial \hat{\mathbf{x}}}{\partial \boldsymbol{\mu}_\mathcal{B}} + \frac{\partial \ell}{\partial \boldsymbol{\sigma}_\mathcal{B}^2} \cdot \frac{\partial \boldsymbol{\sigma}^2_\mathcal{B}}{\partial \boldsymbol{\mu}_\mathcal{B}}\\
    &= \left( \sum_{i=1}^m \frac{\partial \ell}{\partial \hat{\mathbf{x}}_i} \cdot \left( - \frac{1}{\sqrt{\boldsymbol{\sigma}_\mathcal{B}^2 + \epsilon}} \right) \right) + \\
    &\quad\quad \frac{\partial \ell}{\partial \hat{\mathbf{x}}} \cdot \left( - \frac{1}{2} \sum_{i=1}^m \frac{\mathbf{x}_i - \boldsymbol{\mu}_\mathcal{B}}{(\boldsymbol{\sigma}_\mathcal{B}^2 + \epsilon)^{3/2}} \right) \cdot \left( -\frac{1}{m} \sum_{i=1}^m 2 \cdot (\mathbf{x}_i - \boldsymbol{\mu}_\mathcal{B})\right)\\
    &= \left( \sum_{i=1}^m \frac{\partial \ell}{\partial \hat{\mathbf{x}}_i} \cdot \left( - \frac{1}{\sqrt{\boldsymbol{\sigma}_\mathcal{B}^2 + \epsilon}} \right)\right) + \frac{\partial \ell}{\partial \hat{\mathbf{x}}} \cdot \left( - \frac{1}{2} \sum_{i=1}^m \frac{\mathbf{x}_i - \boldsymbol{\mu}_\mathcal{B}}{(\boldsymbol{\sigma}_\mathcal{B}^2 + \epsilon)^{3/2}} \right) \cdot \mathbf{0}\\
    &= \sum_{i=1}^m \frac{\partial \ell}{\partial \hat{\mathbf{x}}_i} \cdot \left( - \frac{1}{\sqrt{\boldsymbol{\sigma}_\mathcal{B}^2 + \epsilon}} \right).
\end{aligned}
$$

Finally we derive the partial for $\mathbf{x}$, which is a function of variables $\hat{\mathbf{x}}$, $\boldsymbol{\mu}_\mathcal{B}$, and $\boldsymbol{\sigma}_\mathcal{B}^2$, as follows,

$$
\begin{aligned}
\frac{\partial \ell}{\partial \mathbf{x}_i} &= \frac{\partial \ell}{\partial \hat{\mathbf{x}}_i} \cdot \frac{\partial \hat{\mathbf{x}}_i}{\partial \mathbf{x}_i} + \frac{\partial \ell}{\partial \boldsymbol{\mu}_\mathcal{B}} \cdot \frac{\partial \boldsymbol{\mu}_\mathcal{B}}{\partial \mathbf{x}_i} + \frac{\partial \ell}{\partial \boldsymbol{\sigma}_\mathcal{B}^2} \cdot \frac{\partial \boldsymbol{\sigma}_\mathcal{B}^2}{\partial \mathbf{x}_i} \\
&= \frac{\partial \ell}{\partial \hat{\mathbf{x}}_i} \cdot \frac{1}{\sqrt{\boldsymbol{\sigma}_\mathcal{B}^2 + \epsilon}} + \frac{\partial \ell}{\partial \boldsymbol{\sigma}_\mathcal{B}^2} \cdot \frac{2 (\mathbf{x}_i - \boldsymbol{\mu}_\mathcal{B}^2)}{m} + \frac{\partial \ell}{\partial \boldsymbol{\mu}_\mathcal{B}} \cdot \frac{1}{m}\\
&= \frac{\partial \ell}{\partial \mathbf{y}_i} \cdot \boldsymbol{\gamma} \cdot \frac{1}{\sqrt{\boldsymbol{\sigma}_\mathcal{B}^2 + \epsilon}} + \sum_{i=1}^m \frac{\partial \ell}{\partial \hat{\mathbf{x}}_i} \cdot \left( - \frac{1}{\sqrt{\boldsymbol{\sigma}_\mathcal{B}^2 + \epsilon}} \right) \cdot \frac{1}{m} + \\
&\quad\quad \frac{\partial \ell}{\partial \hat{\mathbf{x}}_i} \cdot \left( - \frac{1}{2} \sum_{i=1}^m \frac{\mathbf{x}_i - \boldsymbol{\mu}_\mathcal{B}}{(\boldsymbol{\sigma}_\mathcal{B}^2 + \epsilon)^{3/2}} \right) \cdot \frac{2 (\mathbf{x}_i - \boldsymbol{\mu}_\mathcal{B})}{m} \\
&= \frac{\partial \ell}{\partial \mathbf{y}_i} \cdot \boldsymbol{\gamma} \cdot \frac{1}{\sqrt{\boldsymbol{\sigma}_\mathcal{B}^2 + \epsilon}} + \sum_{i=1}^m \frac{\partial \ell}{\partial \hat{\mathbf{x}}_i} \cdot \left( - \frac{1}{\sqrt{\boldsymbol{\sigma}_\mathcal{B}^2 + \epsilon}} \right) \cdot \frac{1}{m} + \\
&\quad\quad \frac{\partial \ell}{\partial \hat{\mathbf{x}}_i} \cdot \left( - \frac{1}{2} \sum_{i=1}^m \frac{\mathbf{x}_i - \boldsymbol{\mu}_\mathcal{B}}{\sqrt{\boldsymbol{\sigma}_\mathcal{B}^2 + \epsilon}} \right) \cdot \frac{2 (\mathbf{x}_i - \boldsymbol{\mu}_\mathcal{B})}{\sqrt{\boldsymbol{\sigma}_\mathcal{B}^2 + \epsilon}} \cdot \frac{1}{m \cdot \sqrt{\boldsymbol{\sigma}_\mathcal{B}^2 + \epsilon}},
\end{aligned}
$$

also note that,

$$
\hat{\mathbf{x}}_i = \frac{\mathbf{x}_i - \boldsymbol{\mu}_\mathcal{B}}{\sqrt{\boldsymbol{\sigma}_\mathcal{B}^2 + \epsilon}},
$$

we have,

$$
\begin{aligned}
\frac{\partial \ell}{\partial \mathbf{x}_i} &= \frac{\partial \ell}{\partial \hat{\mathbf{x}}_i} \cdot \frac{\partial \hat{\mathbf{x}}_i}{\partial \mathbf{x}_i} + \frac{\partial \ell}{\partial \boldsymbol{\mu}_\mathcal{B}} \cdot \frac{\partial \boldsymbol{\mu}_\mathcal{B}}{\partial \mathbf{x}_i} + \frac{\partial \ell}{\partial \boldsymbol{\sigma}_\mathcal{B}^2} \cdot \frac{\partial \boldsymbol{\sigma}_\mathcal{B}^2}{\partial \mathbf{x}_i} \\
&= \frac{\partial \ell}{\partial \hat{\mathbf{x}}_i} \cdot \frac{1}{\sqrt{\boldsymbol{\sigma}_\mathcal{B}^2 + \epsilon}} + \frac{\partial \ell}{\partial \boldsymbol{\sigma}_\mathcal{B}^2} \cdot \frac{2 (\mathbf{x}_i - \boldsymbol{\mu}_\mathcal{B}^2)}{m} + \frac{\partial \ell}{\partial \boldsymbol{\mu}_\mathcal{B}} \cdot \frac{1}{m}\\
&= \frac{\partial \ell}{\partial \mathbf{y}_i} \cdot \boldsymbol{\gamma} \cdot \frac{1}{\sqrt{\boldsymbol{\sigma}_\mathcal{B}^2 + \epsilon}} + \sum_{i=1}^m \frac{\partial \ell}{\partial \hat{\mathbf{x}}_i} \cdot \left( - \frac{1}{\sqrt{\boldsymbol{\sigma}_\mathcal{B}^2 + \epsilon}} \right) \cdot \frac{1}{m} + \\
&\quad\quad \frac{\partial \ell}{\partial \hat{\mathbf{x}}_i} \cdot \left( - \frac{1}{2} \sum_{i=1}^m \hat{\mathbf{x}}_i \right) \cdot \frac{2 \hat{\mathbf{x}}_i}{m \cdot \sqrt{\boldsymbol{\sigma}_\mathcal{B}^2 + \epsilon}}\\
&= \frac{1}{m\sqrt{\boldsymbol{\sigma}_\mathcal{B}^2 + \epsilon}} \cdot \left[ m \frac{\partial \ell}{\partial \hat{\mathbf{x}}_i} - \sum_{j=1}^m \frac{\partial \ell}{\partial \hat{\mathbf{x}}_j} - \hat{\mathbf{x}}_i \sum_{i=1}^m \frac{\partial \ell}{\partial \hat{\mathbf{x}}_j} \cdot \hat{\mathbf{x}}_j \right] \\
&= \frac{m \frac{\partial \ell}{\partial \hat{\mathbf{y}}_i} - \sum_{j=1}^m \frac{\partial \ell}{\partial \hat{\mathbf{y}}_j} - \hat{\mathbf{x}}_i \sum_{i=1}^m \frac{\partial \ell}{\partial \hat{\mathbf{y}}_j} \cdot \hat{\mathbf{x}}_j}{m\sqrt{\boldsymbol{\sigma}_\mathcal{B}^2 + \epsilon}} \odot \boldsymbol{\gamma},
\end{aligned}
$$

where $\{\partial \ell / \partial \mathbf{y}_i \}_{i=1}^m$ are available as the function parameter.

Note that after the "whitening" process, we have the input to the next activation layer,

$$
\mathbf{y}_i \gets \boldsymbol{\gamma} \hat{\mathbf{x}}_i + \boldsymbol{\beta},
$$

where $\boldsymbol{\beta}$ makes the requirement for the bias term in the activation function redundant as the learning procedes. However, by whitening we lose the ability to perform deterministic test-time inference as we have quantities ($\boldsymbol{\mu}_\mathcal{B}$, $\boldsymbol{\sigma}_\mathcal{B}^2$) that depend on the mini-batch instead of on individual samples alone, where the rest of the samples in the mini-batch are randomly selected. So to have some deterministic and reproducible results at test time, people use running averages of the quantities dependent on the mini-batch during training. Specifically, for multiple mini-batches of training, compute the average (over mini-batch means and variances), by taking the running averages, those values are not dependent on the mini-batches anymore, so we have deterministic inference during test-time. The below procedure summarizes the training of batch-normalized networks,

> **procedure 2**
>
> **input:** network $N$ with trainable parameters $\Theta$; subset of activations $\{ \mathbf{x}^{(k)} \}_{k=1}^K$ at layer $k = 1, 2, \cdots, K$.
>
> **output:** batch-normalized network for inference, $N_\mathtt{BN}^\mathtt{inf}$.
>
> 1. $N_\mathtt{BN}^\mathtt{tr} \gets N$ // training the BN network
>
> 2. **for** $k=1, \cdots, K$ **do**
>
> 3. > add batch-norm transformation $\mathbf{y}^{(k)} = \mathtt{BN}_{\boldsymbol{\gamma}^{(k)}, \boldsymbol{\beta}^{(k)}} (\mathbf{x}^{(k)})$ to the network $N_\mathtt{BN}^\mathtt{tr}$ (as in procedure 1)
> 4. > Modify each layer in $N_\mathtt{BN}^\mathtt{tr}$ with input $\mathbf{x}^{(k)}$ to take $\mathbf{y}^{(k)}$ instead
>
> 5. **end for**
>
> 6. Train the batch-normalized network $N_\mathtt{BN}^\mathtt{tr}$ to optimize the model parameters $\Theta \cup \{ \boldsymbol{\gamma}^{(k)}, \boldsymbol{\beta}^{(k)} \}_{k=1}^K$.
>
> 7. $N_\mathtt{BN}^\mathtt{inf} \gets N_\mathtt{BN}^\mathtt{tr}$ // freeze batch-normalized network parameters for inference
>
> 8. **for** $k=1, \cdots, K$ **do**
>
> 9. > // for simplicity, denote $\mathbf{x} \equiv \mathbf{x}^{(k)}, \boldsymbol{\gamma} \equiv \boldsymbol{\gamma}^{(k)}, \boldsymbol{\mu}_\mathcal{B} \equiv \boldsymbol{\mu}_\mathcal{B}^{(k)}$, etc.
>
> 10. > Process multiple training mini-batches $\mathcal{B}$, each of size $m$, and compute the running average,
> $$\begin{aligned}\mathbb{E}[\mathbf{x}] &\gets \mathbb{E}_\mathcal{B}[\boldsymbol{\mu}_\mathcal{B}] \\\mathrm{Var}[\mathbf{x}] &\gets \frac{m}{m - 1} \mathbb{E}_\mathcal{B} [\sigma_\mathcal{B}^2],\end{aligned}$$
>
> 11. > In the batch-normalized network $N_\mathtt{BN}^\mathtt{inf}$, replace the transform $y = \mathtt{BN}_{\boldsymbol{\gamma}, \boldsymbol{\beta}} (\mathbf{x})$ with $\mathbf{y} = \frac{\boldsymbol{\gamma}}{\sqrt{\mathrm{Var}[\mathbf{x}] + \epsilon}} \cdot \mathbf{x} + \left( \boldsymbol{\beta} - \frac{\boldsymbol{\gamma} \mathbb{E}[\mathbf{x}]}{\sqrt{\mathrm{Var}[\mathbf{x}] + \epsilon}} \right)$
>
> 12. **end for**
>
