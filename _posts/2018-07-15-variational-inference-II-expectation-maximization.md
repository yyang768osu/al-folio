---
layout: post
title: "A step-by-step guide to variational inference (2): expectation maximization"
date: 2018-07-15
comments: true
description: how to optimize ELBO when your approx posterior can be easily obtained 
---

In the previous post we went through the derivation of variational lower-bound, and showed how it helps convert the Bayesian inference and density estimation problem to an optimization problem. Let's briefly recap the problem setup and restate some key points.

Consider a very general generative graphic model where each data point $$x^{(n)}$$ is generated from a latent variable $$z^{(n)}$$ conforming to a given distribution $$p(X\|Z;\theta)$$, with $$z^{(n)}$$ itself drawn from a given prior distribution $$p(Z; \theta)$$. $$\theta$$ captures the set of variables that the two probabilities are parameterized with. Two fundamental problems are to (1) estimate the density of existing dataset $$X$$, i.e. $$p(X;\theta)$$ and (2) derive the posterior probability of the latent variable $$Z$$ given the observed data $$X$$, i.e., $$p(Z\|X;\theta)$$. The exact solution of both problems requires the evaluation of the often intractable integral $$\int P(X\| Z;\theta)P(Z;\theta)dZ$$. 

With the introduction of a variational/free distribution function $$q(Z)$$, we have the following identity:

$$
\begin{align*}
\ln p(X;\theta) = \text{KL}{\big(}q||p(Z|X;\theta){\big)} + \mathcal{L}(q,\theta),
\end{align*}
$$

which says that the marginalized probability of dataset $$X$$ can be decomposed into a sum of two terms with the first one being the KL divergence between $$q(Z)$$ and the true posterior distribution $$p(Z\|X;\theta)$$ and the second one expressed below.

$$
\begin{align*}
\mathcal{L}(q,\theta)=\int q(Z) \ln \frac{P(Z,X;\theta)}{q(Z)}dZ,
\end{align*} 
$$

which is referred to as the variational lower bound: it is called a lower-bound as it is always less than $$\ln p(X;\theta)$$ due to the non-negativity of KL divergence, and it is called variational as it is itself a functional that maps a variational/free distribution function $$q$$ to a scalar value. This identity is quite exquisite in that it turns both the density estimation problem and the latent variable inference problem into an optimization problem, evident from the two equations below

$$
\begin{align*}
\ln p(X;\theta) &= \max_{q}\mathcal{L}(q,\theta)\\
p(Z|X;\theta) &= \arg\max_{q}\mathcal{L}(q,\theta).
\end{align*}
$$

The problem that Expectation Maximization algorithm designs to solve is the maximum-likelihood (ML) estimate of the parameter $$\theta$$. Mind you that $$\theta$$ is the parameter of the graphic model, and the task is to find a $$\theta$$ such that the model best explain the existing data. In precise term, the problem is

$$
\begin{align*}
\max_{\theta}\ln p(X;\theta).
\end{align*}
$$

Now, resorting to the variational lower bound, equivalently we can also focus on the following maximization-maximization problem

$$
\begin{align*}
\max_{\theta}\ln p(X;\theta) = \max_{\theta}\max_{q}\mathcal{L}(q,\theta),
\end{align*}
$$

A natural question is: why would this be any easier to evaluate compared with maximizing $$\ln p(X;\theta)$$ head on? did we increase our burden by considering a nested-maximization optimization problem rather than a single-maximization one? 

To answer we need to have the objective function under scrutiny. Looking at the detailed expression of $$\mathcal{L}(q,\theta)$$, the main hurdle is the evaluation the log-likelihood of the joint observed-latent variable $$p(Z,X;\theta)$$. We want to emphasis that the two probability distributions $$p(Z;\theta)$$ and $$p(X\|Z;\theta)$$ are given as part of the model assumption, and they usually come in the form of well-known distributions, e.g., Gaussian, multinomial, exponential, etc. Thus, the joint likelihood of observed and hidden variable $$p(Z,X;\theta)=p(Z;\theta)p(X\|Z;\theta)$$ is in an amenable form. Also, quite often, taking logarithm on it would break up all the multiplicative terms as summation, resulting in quite tractable from. Better yet, the parameters $$\theta$$ that we need to compute gradient with, may naturally be decomposed into different terms in the summation, making the calculation of derivative easy with respect to individual parameters.  

On the other hand, to compute the marginalized likelihood of the observed data only, i.e., $$P(X;\theta)$$, one need to sum or integrate out the effect of $$Z$$ from $$p(Z,X;\theta)$$, which may lead to complicated expression. While the evaluation of $$P(X;\theta)$$ may still be fine when, e.g., the marginalization only requires the summation of a finite number of terms (which is the case for the Gaussian mixture model), the real deal breaker is the difficulty in taking derivative of the log-likelihood with respective to the parameters: taking logarithm on $$P(X;\theta)$$ almost surely won't result in nice decomposition, as the logarithm is gated by the integral or summation, and the log-sum expression is a lot harder to break when we compute the derivative with respect to the parameters $$\theta$$.

Coming back to the maximization-maximization problem, it is natural to devise an iterative algorithm that maximize the objective function $$\mathcal{L}(q,\theta)$$ with alternating axis:

$$
\begin{align*}
\text{E step: }&q^{(t)} = \arg\max_{q}\mathcal{L}(q,\theta^{(t)})\\
\text{M step: }&\theta^{(t+1)} = \arg\max_{\theta}\mathcal{L}(q^{(t)}, \theta) 
\end{align*}
$$

It is worth mentioning that the first optimization problem is in general a very difficult one, as it requires searching through the whole function space. According to the derivation of the variational lower bound derivation we know that the optimal solution is the posterior distribution $$p(Z\|X;\theta^{(t)})$$, which is hard to obtain. Actually finding an approximated posterior by maximizing the variational lower bound is the main theme in variational inference. Techniques of mean-field-approximation, and variational auto-encoder, which we cover in subsequent posts, targets at this problem.

To proceed, we make a very strong assumption that $$p(Z\|X;\theta^{(t)})$$ can be easily obtained. As we will see later that with certain simple model (e.g., Gaussian mixture model), it is indeed a valid assumption, nevertheless it is the key assumption that significantly limits the application of the expectation maximization algorithm.

Anyway, for now let us live with this strong assumption, then the E-step results in the following  expression 

$$
\begin{align*}
\text{E step: }q^{(t)} = p(Z|X;\theta^{(t)}).
\end{align*}
$$

Coming back to the second maximization problem (M-step), with $$q^{(t)}$$ fixed, we can decompose the variational lower bound as 

$$
\begin{align*}
\mathcal{L}(q^{(t)}, \theta) = \int q^{(t)}(Z)\ln p(X,Z;\theta)dZ + \int q^{(t)}(Z) \ln\frac{1}{q^{(t)}(Z)}dZ.
\end{align*}
$$

The second term above is just a constant term reflecting the entropy of $$q^{(t)}$$, so let us ignore it, and then the second maximization problem reduces to 

$$
\begin{align*}
\text{M step: }\theta^{(t+1)} =&\max_{\theta} \int p(Z|X;\theta^{(t)}) \ln P(Z,X;\theta)dZ. 
\end{align*}
$$

The maximization target above can be viewed as finding the expectation of complete data (combining observed variable and latent variable) log likelihood, where the expectation is with respect to a fixed distribution on the latent variable $$Z$$.

Let's put the two steps together and review the whole iterative process. We are given a model with a set of parameters captured in $$\theta$$. The task is find the values of the parameters $$\theta$$ such that the model best explain the existing observed data at hand. At the beginning, we take a random guess on the value of the parameters. With that initial parameters, we find the posterior probability of the latent variable for each data point $$x$$ in the training data set $$X$$. Then, using that posterior probability, we calculate the expected complete-data log-likelihood, and try to find parameters $$\theta$$ so that the complete-data log-likelihood is maximized. With $$\theta$$ updated, we refresh our calculation of the posterior probability and iterative the process. 

In fact, K-means clustering algorithm is one instance of expectation-maximization procedure with certain model assumption. It is helpful to think of the E-M iterative process from the perspective of K-means clustering: for K-means clustering, the latent variable is one-dimensional with value from $$1$$ to $$K$$, implying the registration to one of the $$K$$ clusters. The parameter of the model is the center of the clusters, denoted as $$\theta=\{c_1, \ldots, c_K\}$$. In the initial setup, we randomly set these $$K$$ cluster centers. For each data, we assign it to the nearest cluster, which is effectively assigning its latent variable. This step corresponds to finding the posterior distribution (E-step), with one of the clustering having probability $$1$$. After each data is assigned to its cluster with the initial values of the cluster centers, which gives us complete data in the form of (observed data, latent variable) pair, the next step is to adjust the center based on its constituent. This step corresponds to the maximizing of the expected complete-data log-likelihood (M-step), although this expectation is taken in a degenerated form as the posterior probability for the latent variable is in the form of $$0/1$$.

We finish the treatment of E-M algorithm with the following closing remarks:

1. The E-M iterative algorithm is guaranteed to reach a local maximum on the log-likelihood of the observed data $$p(X;\theta)$$, as both steps increases it.
2. It is not necessary to find the maximum in the M-step. So long as the updated $$\theta$$ increase the complete-data log-likelihood, we are still in the right direction.
3. So far we focused on finding the maximum-likelihood (ML) solution to $$\theta$$ (local maximum). In the case when there is prior distribution $$p_\text{prior}(\theta)$$ on $$\theta$$, we can use the same process to find a maximum-a-posterior (MAP) solution (local maximum), utilizing the fact that $$p(\theta\|X) \propto p(X\|\theta)p_\text{prior}(\theta)$$. The problem is modified as

$$
\begin{align*}
\max_{\theta}\ln p(\theta|X) = \max_{\theta}\left(\max_{q}\mathcal{L}(q,\theta) {\color{red} + \ln p_\text{prior}(\theta)}\right),
\end{align*}
$$

with slightly modified procedure below

$$
\begin{align*}
\text{E step: }&q^{(t)} = p(Z|X,\theta^{(t)})\\
\text{M step: }&\theta^{(t+1)} =\max_{\theta} \int p(Z|X,\theta^{(t)}) \ln P(Z,X|\theta)dZ {\color{red} + \ln p_\text{prior}(\theta)}. 
\end{align*}
$$

