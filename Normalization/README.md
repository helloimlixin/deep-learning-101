# Normalization in Deep Neural Networks

## Input-Level Normalization

One issue of deep neural nets is that the learning capability of the network deteriorates as the network goes deeper due to vanishing or exploding gradients. Adding normalization layers provides a common remedy to this kind of problem by *standardizing the statistics of the hidden units* (zero mean & unit variance, also known as *whitening*). Specifically, for input feature set $\{ \mathbf{x}^{(i)} \}_{i = 1}^N$ with dimension $D$, i.e., $\mathbf{x}^{(i)} \in \mathbb{R}^D$, we have the "standardization" of the input features on dimension $j$ ($j$-th feature) can be represented as,
$$
\hat{x}^{(i)}_j = \frac{x^{(i)}_j - \mu_j}{\sigma_j},\: \text{(thus the scaled feature will have zero mean and unit variance)}
$$
where,
$$
\begin{aligned}
\mu_j &= \frac{1}{N} \sum_{i = 1}^N x_j^{(i)} \\
\sigma_j^2 &= \frac{1}{N} \sum_{i = 1}^N (x^{(i)}_j - \mu_j)^2
\end{aligned}
$$
Normalization also helps reduce the impact from features of different scales, as we are applying the same learning rate for all weight parameters, so large parameters will dominate the weight updates while using stochastic gradient descent algorithms.

## Extending Input Normalization to the Hidden Layers --- Batch Normalization

However, normalizing the inputs only affects the first hidden layer, to standardize deeper hidden layer activations, one approach is called the **Batch Normalization**, which ensures the distribution of the hidden layer activations for each layer has a zero mean and unit variance. Some key properties of the batch normalization,

- Normalizes hidden layer inputs.
- Helps with exploding/vanishing gradient problems.
- Can increase training stability and convergence rate.
- Can be understood as additional (normalization) layers (with additional parameters).

### TODO Understanding Covariate Shift???

When the input distribution to a learning system changes, it is said to experience *covariate shift*, furthermore, the idea of covariate shift can be extended to part of the learning system, such as a layer of the network. Consider the computation on the second layer of a neural network with input $u$
$$\ell = F_2 (F_1 (u, \Theta_1), \Theta_2),$$
where $F_1$ and $F_2$ are arbitrary network activation functions, and the parameters $\Theta_1, \Theta_2$ are to be learned to minimize the loss $\ell$. While using Stochastic Gradient Descent (SGD) methods, the training proceeds in minibatches, i.e., for the training set $\{ \mathbf{x}^{(i)}\}_{i=1}^N$, at each training step we consider a minibatch $\{ \mathbf{x}^{(i)} \}_{i=1}^m$, denote the learning rate as $\alpha$, we then have for a gradient descent step, the parameter update rule for $\Theta_2$ can be expressed as,
$$\Theta_2 \gets \Theta_2 - \frac{\alpha}{m} \sum_{i=1}^m \frac{\partial F_2 (\mathbf{x}^{(i)}, \Theta_2)}{\partial \Theta_2}$$

## Chain Rule Primer

Suppose we have a function $u(x, y)$ where $x(r, t)$ and $y(r, t)$ are also two functions. Then to compute $\frac{\partial u}{\partial r}$ and $\frac{\partial u}{\partial t}$ we apply the chain rule,
$$
\begin{aligned}
\frac{\partial u}{\partial r} &= \frac{\partial u}{\partial x} \cdot \frac{\partial x}{\partial r} + \frac{\partial u}{\partial y} \cdot \frac{\partial y}{\partial r} \\
\frac{\partial u}{\partial t} &= \frac{\partial u}{\partial x} \cdot \frac{\partial x}{\partial t} + \frac{\partial u}{\partial y} \cdot \frac{\partial y}{\partial t}
\end{aligned}
$$

## Batch Normalization

Consider a mini-batch $\mathcal{B} = \{ \mathbf{x}^{(i)} \}_{i = 1}^m$, denote the normalized values as $\{ \hat{\mathbf{x}}^{(i)}\}_{i=1}^m$ and the corresponding linear transformation as $\{\mathbf{y}^{(i)}\}_{i=1}^m$, we refer to the transform,
$$\mathtt{BN}_{\boldsymbol{\gamma}, \boldsymbol{\beta}}: \{ \mathbf{x}^{(i)} \}_{i = 1}^m \to \{\mathbf{y}^{(i)}\}_{i=1}^m$$
as the *Batch Normalization Transform*, which can be performed in the following steps,
> **Input**: values of $\mathbf{x}$ over a mini-batch: $\mathcal{B} = \{ \mathbf{x}^{(i)} \}_{i = 1}^m$;
>
> **Learnable parameters**: $\boldsymbol{\gamma}$, $\boldsymbol{\beta}$
>
> **Output**: $\{ y^{(i)} = \mathtt{BN}_{\boldsymbol{\gamma}, \boldsymbol{\beta}} (\mathbf{x}^{(i)}) \}_{i=1}^m$.
>
> $\boldsymbol{\mu}_\mathcal{B} \gets \frac{1}{m} \sum_{i=1}^m \mathbf{x}^{(i)}$ // mini-batch mean
>
> $\boldsymbol{\sigma}_\mathcal{B}^2 \gets \frac{1}{m} \sum_{i=1}^m (\mathbf{x}^{(i)} - \mu_\mathcal{B})^2$ // mini-batch variance
>
> $\hat{\mathbf{x}}^{(i)} \gets \frac{\mathbf{x}^{(i)} - \boldsymbol{\mu}_\mathcal{B}}{\sqrt{\boldsymbol{\sigma}_\mathcal{B}^2 + \epsilon}}$ // normalize, $\epsilon$ is a small constant for numerical stability, say $1e-5$
>
> $\mathbf{y}^{(i)} \gets \boldsymbol{\gamma} \odot \hat{\mathbf{x}}^{(i)} + \boldsymbol{\beta} \equiv \mathtt{BN}_{\boldsymbol{\gamma}, \boldsymbol{\beta}} (\mathbf{x}^{(i)})$ // scale and shift

Since the transformation is differentiable, we can easily pass the gradients back to the input of the layer and to the batch normalization parameters $\boldsymbol{\gamma}$ and $\boldsymbol{\beta}$. Specifically, we have during the backpropagation (not simplified),
$$
\begin{aligned}
\frac{\partial \ell}{\partial \hat{\mathbf{x}}^{(i)}} &= \frac{\partial \ell}{\partial \mathbf{y}^{(i)}} \cdot \boldsymbol{\gamma} \\
\frac{\partial \ell}{\partial \boldsymbol{\sigma}_\mathcal{B}^2} &= \sum_{i=1}^m \frac{\partial \ell}{\partial \hat{\mathbf{x}}^{(i)}} \cdot (\mathbf{x}^{(i)} - \boldsymbol{\mu}_\mathcal{B}) \cdot \left(-\frac{1}{2} \right) (\boldsymbol{\sigma}_\mathcal{B}^2 + \epsilon)^{-3/2} \\
\frac{\partial \ell}{\partial \boldsymbol{\mu}_\mathcal{B}} &= \left(\sum_{i=1}^m \frac{\partial \ell}{\partial \hat{\mathbf{x}}^{(i)}} \cdot \frac{-1}{\sqrt{\boldsymbol{\sigma}_\mathcal{B}^2 + \epsilon}}\right) + \frac{\partial \ell}{\partial \boldsymbol{\sigma}_\mathcal{B}^2} \cdot \frac{\sum_{i=1}^m -2(\mathbf{x}^{(i)} - \boldsymbol{\mu}_\mathcal{B})}{m} \\
\frac{\partial \ell}{\partial \mathbf{x}^{(i)}} &= \frac{\partial \ell}{\partial \hat{\mathbf{x}}^{(i)}} \cdot \frac{1}{\sqrt{\boldsymbol{\sigma}_\mathcal{B}^2 + \epsilon}} + \frac{\partial \ell}{\partial \boldsymbol{\sigma}_\mathcal{B}^2} \cdot \frac{2 (\mathbf{x}^{(i)} - \boldsymbol{\mu}_\mathcal{B}^2)}{m} + \frac{\partial \ell}{\partial \boldsymbol{\mu}_\mathcal{B}} \cdot \frac{1}{m} \\
\frac{\partial \ell}{\partial \boldsymbol{\gamma}} &= \sum_{i=1}^m \frac{\partial \ell}{\partial \mathbf{y}^{(i)}} \cdot \hat{\mathbf{x}}^{(i)}\\
\frac{\partial \ell}{\partial \boldsymbol{\beta}} &= \sum_{i=1}^m \frac{\partial \ell}{\partial \mathbf{y}^{(i)}}
\end{aligned}
$$

### Derivations

We first express the forward pass in the diagram below, for a mini-batch $\mathcal{B} = \{ \mathbf{x}^{(i)} \}_{i=1}^m$, we have
$$\mathbf{x} \to \hat{\mathbf{x}}(\boldsymbol{\mu}_\mathcal{B}, \boldsymbol{\sigma}_\mathcal{B}^2, \mathbf{x}) \to \mathbf{y}(\hat{\mathbf{x}}, \boldsymbol{\gamma}, \boldsymbol{\beta}) \to \ell(\bold{y}),$$
hence the backward pass can be expressed as,
$$\ell (\mathbf{y}) \to \mathbf{y}(\hat{\mathbf{x}}, \boldsymbol{\gamma}, \boldsymbol{\beta}) \to \hat{\mathbf{x}} (\boldsymbol{\mu}_\mathcal{B}, \boldsymbol{\sigma}_\mathcal{B}^2, \mathbf{x})$$

First we have $\frac{\partial \ell}{\partial \mathbf{y}^{(i)}}$ available as the upstream derivative, which is given in the function parameter.

Then to compute $\mathbf{y}(\hat{\mathbf{x}}, \boldsymbol{\gamma}, \boldsymbol{\beta})$, which consists of three variables, we first compute the gradients with respect to each variable,
$$
\begin{aligned}
    \frac{\partial \ell}{\partial \bold{\gamma}} &= \frac{\partial \ell}{\partial \mathbf{y}^{(i)}} \cdot \frac{\partial \mathbf{y}^{(i)}}{\partial \boldsymbol{\gamma}} \\
    &= \sum_{i=1}^m \frac{\partial \ell}{\partial \mathbf{y}^{(i)}} \cdot \hat{\mathbf{x}}^{(i)},
\end{aligned}
$$
the summation indicates the computation is for the batches. As for $\boldsymbol{\beta}$ we have,
$$
\begin{aligned}
    \frac{\partial \ell}{\partial \boldsymbol{\beta}} &= \frac{\partial \ell}{\partial \mathbf{y}^{(i)}} \cdot \frac{\partial \mathbf{y}^{(i)}}{\partial \boldsymbol{\beta}} \\
    &= \sum_{i=1}^m \frac{\partial \ell}{\partial \mathbf{y}^{(i)}},
\end{aligned}
$$
and finally $\hat{\mathbf{x}}^{(i)}$,
$$
\begin{aligned}
    \frac{\partial \ell}{\partial \hat{\mathbf{x}}^{(i)}} &= \frac{\partial \ell}{\partial \mathbf{y}^{(i)}} \cdot \frac{\partial \mathbf{y}^{(i)}}{\partial \hat{\mathbf{x}}^{(i)}} \\
    &= \frac{\partial \ell}{\partial \mathbf{y}^{(i)}} \cdot \boldsymbol{\gamma}
\end{aligned}
$$
To compute the gradient with respect to $\mathbf{x}^{(i)}$, we have since $\boldsymbol{\mu}$ and $\boldsymbol{\sigma}^2$ is a function of $\boldsymbol{\mu}$,
$$
\frac{\partial \ell}{\partial \bold{\mu}_\mathcal{B}} = \frac{\partial \ell}{\partial \hat{\mathbf{x}}^{(i)}} \cdot \frac{\partial \hat{\mathbf{x}}^{(i)}}{\partial \boldsymbol{\mu}_\mathcal{B}} + \frac{\partial \ell}{\partial \boldsymbol{\sigma}_\mathcal{B}^2} \cdot \frac{\partial \boldsymbol{\sigma}^2_\mathcal{B}}{\partial \boldsymbol{\mu}_\mathcal{B}},
$$
we have,
$$
\frac{\partial \hat{\mathbf{x}}^{(i)}}{\partial \boldsymbol{\mu}_\mathcal{B}} = -\frac{1}{\sqrt{\boldsymbol{\sigma}_\mathcal{B}^2 + \epsilon}},
$$
and,
$$
\frac{\partial \boldsymbol{\sigma}_\mathcal{B}^2}{\partial \boldsymbol{\mu}_\mathcal{B}} = -\frac{1}{m} \sum_{i=1}^m 2 \cdot (\boldsymbol{x}^{(i)} - \boldsymbol{\mu}_\mathcal{B}).
$$
Next we compute the partial,
$$
\begin{aligned}
\frac{\partial \ell}{\partial \boldsymbol{\sigma}_\mathcal{B}^2} &= \frac{\partial \ell}{\partial \hat{\mathbf{x}}} \cdot \frac{\partial \hat{\mathbf{x}}}{\partial \boldsymbol{\sigma}_\mathcal{B}^2} \\
&= \frac{\partial \ell}{\partial \hat{\mathbf{x}}} \cdot \left( - \frac{1}{2} \sum_{i=1}^m \frac{\mathbf{x}^{(i)} - \boldsymbol{\mu}_\mathcal{B}}{(\boldsymbol{\sigma}_\mathcal{B}^2 + \epsilon)^{3/2}} \right).
\end{aligned}
$$
Thus we have,
$$
\begin{aligned}
    \frac{\partial \ell}{\partial \boldsymbol{\mu}_\mathcal{B}} &= \frac{\partial \ell}{\partial \hat{\mathbf{x}}} \cdot \frac{\partial \hat{\mathbf{x}}}{\partial \boldsymbol{\mu}_\mathcal{B}} + \frac{\partial \ell}{\partial \boldsymbol{\sigma}_\mathcal{B}^2} \cdot \frac{\partial \boldsymbol{\sigma}^2_\mathcal{B}}{\partial \boldsymbol{\mu}_\mathcal{B}}\\
    &= \left( \sum_{i=1}^m \frac{\partial \ell}{\partial \hat{\mathbf{x}}^{(i)}} \cdot \left( - \frac{1}{\sqrt{\boldsymbol{\sigma}_\mathcal{B}^2 + \epsilon}} \right) \right) + \\
    &\quad\quad \frac{\partial \ell}{\partial \hat{\mathbf{x}}} \cdot \left( - \frac{1}{2} \sum_{i=1}^m \frac{\mathbf{x}^{(i)} - \boldsymbol{\mu}_\mathcal{B}}{(\boldsymbol{\sigma}_\mathcal{B}^2 + \epsilon)^{3/2}} \right) \cdot \left( -\frac{1}{m} \sum_{i=1}^m 2 \cdot (\mathbf{x}^{(i)} - \boldsymbol{\mu}_\mathcal{B})\right)\\
    &= \left( \sum_{i=1}^m \frac{\partial \ell}{\partial \hat{\mathbf{x}}^{(i)}} \cdot \left( - \frac{1}{\sqrt{\boldsymbol{\sigma}_\mathcal{B}^2 + \epsilon}} \right)\right) + \frac{\partial \ell}{\partial \hat{\mathbf{x}}} \cdot \left( - \frac{1}{2} \sum_{i=1}^m \frac{\mathbf{x}^{(i)} - \boldsymbol{\mu}_\mathcal{B}}{(\boldsymbol{\sigma}_\mathcal{B}^2 + \epsilon)^{3/2}} \right) \cdot \mathbf{0}\\
    &= \sum_{i=1}^m \frac{\partial \ell}{\partial \hat{\mathbf{x}}^{(i)}} \cdot \left( - \frac{1}{\sqrt{\boldsymbol{\sigma}_\mathcal{B}^2 + \epsilon}} \right).
\end{aligned}
$$
Finally we derive the partial for $\mathbf{x}$, which is a function of variables $\hat{\mathbf{x}}$, $\boldsymbol{\mu}_\mathcal{B}$, and $\boldsymbol{\sigma}_\mathcal{B}^2$,
$$
\begin{aligned}
\frac{\partial \ell}{\partial \mathbf{x}^{(i)}} &= \frac{\partial \ell}{\partial \hat{\mathbf{x}}^{(i)}} \cdot \frac{\partial \hat{\mathbf{x}}^{(i)}}{\partial \mathbf{x}^{(i)}} + \frac{\partial \ell}{\partial \boldsymbol{\mu}_\mathcal{B}} \cdot \frac{\partial \boldsymbol{\mu}_\mathcal{B}}{\partial \mathbf{x}^{(i)}} + \frac{\partial \ell}{\partial \boldsymbol{\sigma}_\mathcal{B}^2} \cdot \frac{\partial \boldsymbol{\sigma}_\mathcal{B}^2}{\partial \mathbf{x}^{(i)}} \\
&= \frac{\partial \ell}{\partial \hat{\mathbf{x}}^{(i)}} \cdot \frac{1}{\sqrt{\boldsymbol{\sigma}_\mathcal{B}^2 + \epsilon}} + \frac{\partial \ell}{\partial \boldsymbol{\sigma}_\mathcal{B}^2} \cdot \frac{2 (\mathbf{x}^{(i)} - \boldsymbol{\mu}_\mathcal{B}^2)}{m} + \frac{\partial \ell}{\partial \boldsymbol{\mu}_\mathcal{B}} \cdot \frac{1}{m}\\
&= \frac{\partial \ell}{\partial \mathbf{y}^{(i)}} \cdot \boldsymbol{\gamma} \cdot \frac{1}{\sqrt{\boldsymbol{\sigma}_\mathcal{B}^2 + \epsilon}} + \sum_{i=1}^m \frac{\partial \ell}{\partial \hat{\mathbf{x}}^{(i)}} \cdot \left( - \frac{1}{\sqrt{\boldsymbol{\sigma}_\mathcal{B}^2 + \epsilon}} \right) \cdot \frac{1}{m} + \\
&\quad\quad \frac{\partial \ell}{\partial \hat{\mathbf{x}}^{(i)}} \cdot \left( - \frac{1}{2} \sum_{i=1}^m \frac{\mathbf{x}^{(i)} - \boldsymbol{\mu}_\mathcal{B}}{(\boldsymbol{\sigma}_\mathcal{B}^2 + \epsilon)^{3/2}} \right) \cdot \frac{2 (\mathbf{x}^{(i)} - \boldsymbol{\mu}_\mathcal{B})}{m} \\
&= \frac{\partial \ell}{\partial \mathbf{y}^{(i)}} \cdot \boldsymbol{\gamma} \cdot \frac{1}{\sqrt{\boldsymbol{\sigma}_\mathcal{B}^2 + \epsilon}} + \sum_{i=1}^m \frac{\partial \ell}{\partial \hat{\mathbf{x}}^{(i)}} \cdot \left( - \frac{1}{\sqrt{\boldsymbol{\sigma}_\mathcal{B}^2 + \epsilon}} \right) \cdot \frac{1}{m} + \\
&\quad\quad \frac{\partial \ell}{\partial \hat{\mathbf{x}}^{(i)}} \cdot \left( - \frac{1}{2} \sum_{i=1}^m \frac{\mathbf{x}^{(i)} - \boldsymbol{\mu}_\mathcal{B}}{\sqrt{\boldsymbol{\sigma}_\mathcal{B}^2 + \epsilon}} \right) \cdot \frac{2 (\mathbf{x}^{(i)} - \boldsymbol{\mu}_\mathcal{B})}{\sqrt{\boldsymbol{\sigma}_\mathcal{B}^2 + \epsilon}} \cdot \frac{1}{m \cdot \sqrt{\boldsymbol{\sigma}_\mathcal{B}^2 + \epsilon}},
\end{aligned}
$$
also note that,
$$
\hat{\mathbf{x}}^{(i)} = \frac{\mathbf{x}^{(i)} - \boldsymbol{\mu}_\mathcal{B}}{\sqrt{\boldsymbol{\sigma}_\mathcal{B}^2 + \epsilon}},
$$
we have,
$$
\begin{aligned}
\frac{\partial \ell}{\partial \mathbf{x}^{(i)}} &= \frac{\partial \ell}{\partial \hat{\mathbf{x}}^{(i)}} \cdot \frac{\partial \hat{\mathbf{x}}^{(i)}}{\partial \mathbf{x}^{(i)}} + \frac{\partial \ell}{\partial \boldsymbol{\mu}_\mathcal{B}} \cdot \frac{\partial \boldsymbol{\mu}_\mathcal{B}}{\partial \mathbf{x}^{(i)}} + \frac{\partial \ell}{\partial \boldsymbol{\sigma}_\mathcal{B}^2} \cdot \frac{\partial \boldsymbol{\sigma}_\mathcal{B}^2}{\partial \mathbf{x}^{(i)}} \\
&= \frac{\partial \ell}{\partial \hat{\mathbf{x}}^{(i)}} \cdot \frac{1}{\sqrt{\boldsymbol{\sigma}_\mathcal{B}^2 + \epsilon}} + \frac{\partial \ell}{\partial \boldsymbol{\sigma}_\mathcal{B}^2} \cdot \frac{2 (\mathbf{x}^{(i)} - \boldsymbol{\mu}_\mathcal{B}^2)}{m} + \frac{\partial \ell}{\partial \boldsymbol{\mu}_\mathcal{B}} \cdot \frac{1}{m}\\
&= \frac{\partial \ell}{\partial \mathbf{y}^{(i)}} \cdot \boldsymbol{\gamma} \cdot \frac{1}{\sqrt{\boldsymbol{\sigma}_\mathcal{B}^2 + \epsilon}} + \sum_{i=1}^m \frac{\partial \ell}{\partial \hat{\mathbf{x}}^{(i)}} \cdot \left( - \frac{1}{\sqrt{\boldsymbol{\sigma}_\mathcal{B}^2 + \epsilon}} \right) \cdot \frac{1}{m} + \\
&\quad\quad \frac{\partial \ell}{\partial \hat{\mathbf{x}}^{(i)}} \cdot \left( - \frac{1}{2} \sum_{i=1}^m \hat{\mathbf{x}}^{(i)} \right) \cdot \frac{2 \hat{\mathbf{x}}^{(i)}}{m \cdot \sqrt{\boldsymbol{\sigma}_\mathcal{B}^2 + \epsilon}}\\
&= \frac{1}{m\sqrt{\boldsymbol{\sigma}_\mathcal{B}^2 + \epsilon}} \cdot \left[ m \frac{\partial \ell}{\partial \hat{\mathbf{x}^{(i)}}} - \sum_{j=1}^m \frac{\partial \ell}{\partial \hat{\mathbf{x}}^{(j)}} - \hat{\mathbf{x}}^{(i)} \sum_{i=1}^m \frac{\partial \ell}{\partial \hat{\mathbf{x}}^{(j)}} \cdot \hat{\mathbf{x}}^{(j)} \right]
\end{aligned}
$$
