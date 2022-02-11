---
layout: post
title: Enforcing Lipschitz Constant in Neural Network 
date: 2021-04-03
comments: true
description: How to enforce lipschitz constraint in neural networks?
---

A function $$g(x)$$ is Lipschitz continuous if there exists a constant $$L$$ such that $$\|g(x_1) - g(x_2)\| < L \|x_1 - x_2\|$$ for any $$x_1$$ and $$x_2$$ in its domain. $$L$$ is referred to as a Lipschitz constant of $$g$$. The need to enforce a certain Lipschitz constant of neural networks arises in many cases, with some examples listed below. Here we introduce a common technique used in many existing literatures.

* Guarantee invertibility in normalizing flows build with residual blocks 
  * [iResNet(ICML2019)](https://arxiv.org/abs/1811.00995)
* Discriminator regularization in GAN training
  * [Wasserstein-GAN(ICML2017)](https://arxiv.org/abs/1701.07875)
  * [SpectralNormalization(ICLR2018)](https://arxiv.org/abs/1802.05957)
* Improve network robustness against adversarial perturbations
  * [Lipschitz-margin-training(NIPS2018)](https://arxiv.org/abs/1802.04034)

A small note before we proceed: Lipschitz continuous/constant is defined with respect to a choice of the norm $$\|\cdot\|$$. Here we focus on 2-norm.

## Lipschitz constant vs spectral norm of matrices

Deep neural networks are typically build with interleaved linear layers (such as Conv, TConv, Pooling) together with nonlinear activations (such as ReLU, sigmoid). The Lipschitz constant of most activation function are either constant or easy to control, so we will only focus on linear operations. Linear operations in general can be expressed as in the form of matrix-vector product $$y = g(x) = Wx$$ where $$W$$ denotes a matrix. In this case, the smallest Lipschitz constant of $$g$$ can be expressed as

\begin{equation}
\label{eq:lipconst}
\min_{x_1, x_2, x_1\not=x_2} \frac{
||g(x_1)-g(x_2)||
}{
||x_1 - x_2||
}
=
\min_{||v||\not=0}\frac{
||Wv||
}{
||v||
}
=
\min_{||v||=1}
||Wv||.
\end{equation}


The last term is also known as the *spectral norm* of matrix $$W$$. Let us express $$W$$ as its singular-value-decomposition $$U\Sigma V^T$$, then we can see that the spectral norm of $$W$$ is its maximum singular value, denoted as $$\sigma_1$$. The maximum singular value of $$W$$ is also the maximum eigenvalue of $$M\triangleq W^TW$$ given that eigenvalues of $$W^TW$$ is square of singular values of $$W$$: $$M=V\Sigma U^TU\Sigma V^T=V\Sigma^2V^T=V\Lambda V^T$$.

Now we know that obtaining the best Lipschitz constant of a linear operations amounts to finding the dominant singular value of its matrix representation $$W$$, or dominant eigenvalue of $$M\triangleq W^TW$$. Next let us introduce an iterative algorithm that can find it. 

## Power method (aka Von Mises iteration)

Power method finds the maximum eigenvalue of a matrix $$M$$ using the following iteration:

$$
\begin{align*}
&\text{start with a random vector }v^{(0)} \\
&\text{for }k=1, 2, \ldots, \\
&v^{(k)} = \frac{
M v^{(k-1)}
}{
||M v^{(k-1)}||
}
\end{align*}
$$

Claim: $$\|M v^{(k)}\|$$ converges to the maximum eigen-value of $$M$$ as $$k$$ approaches infinity. 

To show it, let us write the initial vector $$v^{(0)}$$ as a linear combinations of eigen-vectors of $$M$$: $$v^{(0)}=\sum_{i}\alpha_i v_i$$, and expand the iterative formula as

$$
\begin{align*}
&v^{(k)} = \frac{
M v^{(k-1)}
}{
||M v^{(k-1)}||
}
= \frac{
M^2 v^{(k-2)}
}{
||M^2v^{(k-2)}||
}=\ldots
= \frac{
M^k v^{(0)}
}{
||M^kv^{(0)}||
}
\\
&M^k v^{(0)} = M^k \sum_{i} \alpha_i v_i = \sum_{i} \alpha_i M^k v_i \sum_{i}\alpha_i \lambda_i^k v_i = \alpha_1\lambda_1^k
\left(
v_1 + \sum_{i>1}\underbrace{\frac{\alpha_i}{\alpha_1}\left(\frac{\lambda_i}{\lambda_1}\right)^k}_{\to 0 \text{ as } k\to\infty} v_i
\right).
\end{align*}
$$

From the last equation we know that $$v^{(k)}$$ converges to the dominant eigen-vector $$v_1$$ of $$M$$ up to a sign difference, and similarly $$Mv^{(k)}$$ converges to the maximum eigen-value $$\sigma_1$$ of $$M$$.

$$
\begin{align*}
v^{(k)}\to\left\{\begin{array}{ll}
v_1 & \text{if }\alpha_1>0\\
-v_1 & \text{if }\alpha_1<0
\end{array}\right., \text{ as }k\to\infty
\end{align*}
$$

## Compute power iteration through auto-differentiation

From last section we know that the maximum singular value can be computed if we carry out the following iteration procedure:

$$
\begin{align*}
 v^{(k-1)} \Longrightarrow \underbrace{\tilde{v}^{(k)}=W^TWv^{(k-1)}}_{\text{step 1}} \Longrightarrow \underbrace{v^{(k)}=\tilde{v}^{(k)}/||\tilde{v}^{(k)}||}_{\text{step 2}} \Longrightarrow\ldots
\end{align*}
$$

While it is easy to compute vector norm as done in step 2, it is not immediately clear how to easily compute $$W^TWv^{(k-1)}$$ in step 1, since expressing $$W$$ explicitly for a general linear layer can be involved. For instance, for a 2D convolution operation, expressing it in the matrix-vector product form requires unpacking the convolution kernel into a doubly Toeplitz matrix. We know that $$Wv^{(k-1)}$$ is just the output of the linear operator $$g$$ when $$v^{(k-1)}$$ is used as input, but seemingly there is no easy way to multiply by $$W^T$$ without knowing $$W$$ explicitly. 

Here's the trick: we can express $$W^TWx$$ as the derivative of another another function and compute it with auto-differentiation.

$$
\begin{align*}
W^TW x = \frac{1}{2}\frac{\partial x^TW^TWx}{\partial x} = \frac{1}{2}\frac{\partial ||Wx||^2}{\partial x} =\frac{\partial \frac{1}{2}||g(x)||^2}{\partial x}
\end{align*}
$$

We can then modify the iteration procedure as 

$$
\begin{align*}
 v^{(k-1)} \Longrightarrow \underbrace{
\tilde{v}^{(k)} = \frac{
\partial\frac{1}{2}||g(v^{v^{(k-1)}})||^2
}{
\partial v^{(k-1)}
}
 }_{\text{step 1}} \Longrightarrow \underbrace{v^{(k)}=\tilde{v}^{(k)}/||\tilde{v}^{(k)}||}_{\text{step 2}} \Longrightarrow\ldots
\end{align*}
$$

Based on last section, $$\sqrt{\|\tilde{v}^{(k)}\|}$$ yields an estimate of the dominant singular value of $$M$$, which is the Lipschitz constant of the linear operator $$g$$.
In PyTorch, step 1 can be calculated using [torch.atuograd.grad](https://pytorch.org/docs/stable/autograd.html#torch.autograd.grad).

It should not be surprising that the above iteration procedure converges to maximum singular value of $$W$$ -- it is simply the gradient ascent with Equation \eqref{eq:lipconst} as the optimization objective.  

## Enforce Lipschitz constant $$c$$ during training

It is easy to see that the Lipschitz constant of $$a\times g(\cdot)$$ is $$a$$ times the Lipschitz constant of $$g(\cdot)$$, or more precisely, $$\text{Lip}(ag) = a\text{Lip}(g)$$. To enforce the Lipschitz constant of an operator to be some target value $$c$$, we just need to normalize the output the operator by $$c/\text{Lip}(g)$$.

The power iteration procedure itself can be amortized and blended into the optimization step of the network training, in which case the training loop can be expressed as

$$
\begin{align*}
&\text{for step }k=1, \ldots:\\
&v = v/||v||\\
&v = \frac{
\partial\frac{1}{2}||g(v)||^2
}{
\partial v 
}\\
&\sigma = \sqrt{||v||}\\
&\text{set the normalization scale of output of }g\text{ as }\frac{c}{\sigma}\\
&\text{the rest of the training step}.
\end{align*}
$$










