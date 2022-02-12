---
layout: post
title: "A step-by-step guide to variational inference (3): mean field approximation"
date: 2018-07-22
comments: true
description: posterior approximation before age of deep learning
---

We have learned in the previous post that E-M algorithm tries to find a ML or MAP solution to the parameters of a generative model. It is build on top of two major premises:

1. $$\ln p(X,Z;\theta)$$ is in a much simpler form than $$\ln P(X;\theta)$$,
2. $$\ln p(Z\|X;\theta)$$ is easy to obtain.

While the first one is often true in that both $$p(X\|Z;\theta)$$ and $$p(Z;\theta)$$ given as part of the model and are usually designed to be simple, the second one is a very strong assumption and does not hold in most cases. In this post, we remove the second premise, and introduce a way to obtain an approximation of $$p(Z\|X;\theta)$$.

In what follows, we modify notation slightly by assuming that there is prior distribution on any parameters of interest, and conceptually merge $$\theta$$ as part of the latent variable $$Z$$ (which is common across different data samples) and remove $$\theta$$ from the notation.

As we discussed in the previous posts, the Bayesian inference problem is to find the posterior probability $$p(Z\|X)$$, which is in general very hard due to the integral/summation (in most cases multi-dimensional integral/summation) in the denominator below

$$
\begin{align*}
p(Z|X) = \frac{p(X|Z)p(Z)}{\int p(X|Z)p(Z) dZ}.
\end{align*}
$$

We also showed that with the introduction of a variational distribution $$q(Z)$$, we can convert the problem of finding $$p(Z\|X)$$ as an optimization problem below

$$
\begin{align*}
p(Z|X) &= \arg\max_{q}\mathcal{L}(q).
\end{align*}
$$

However, this optimization problem above is still very hard and it does not lend itself to any easy solution. Here, the objective $$\mathcal{L}(q)$$, called the variational lower-bound, is a functional as it maps a function into a scalar value, whose domain is the space of all functions. 

Since it is hard to optimize the variational lower bound as is, one may wonder, how about constraining the search space of $$q$$ from all potential functions to within a limited function space? Could that make the problem simpler? Even though we may not find the optimal solution after restricting the set of functions we could search from, the hope is that by doing so we can device practical algorithms with solutions that are reasonably close to the true posterior. This is exactly the idea behind mean field approximation. 

In the mean field method, we add a constraint to the domain of the optimization: instead of allowing $$q(Z)$$ to be in arbitrary form, we only look at cases when it can be factorized into a product form with disjoint latent variables in each multiplicative factor. More specifically, we divide the dimension of latent variables into $$K$$ groups $$Z=[Z_1, Z_2, \ldots, Z_K]$$ and enforce $$q$$ to have the form of $$q(Z)=q_1(Z_1)q_2(Z_2)\ldots q_K(Z_K)$$. Put it in precise math, we have

$$
\begin{align*}
q^* = \underset{q=q_1 q_2 \ldots q_K}{\arg\max}\mathcal{L}(q)
\end{align*}
$$

Referring back to the equation on the decomposition of observed data log-likelihood

$$
\begin{align*}
\ln p(X) = \text{KL}(q || p(Z|X)) + \mathcal{L}(q),
\end{align*}
$$

we know that by maximizing $$\mathcal{L}(q)$$ with respect to functions with form $$q(Z)=\prod_{k=1}^K q_k(Z_k)$$, we are trying to find one function in the confined function space (defined as the set of functions that can be factorized as such) that is closest to the true posterior $$p(Z\|X)$$ measured in KL divergence. It is worth emphasizing that we are merely constraining $$q(Z)$$ to have this factorization form, and do not make any assumption on what each individual factor would look like. 

Let us plug in the factorized form of $$q(Z) = \prod_{k=1}^K q_k(Z_k)$$ in the expression of the variational lower bound, yielding

$$
\begin{align*}
\mathcal{L}(q) =& \int q(Z)\ln\frac{p(X, Z)}{q(Z)}dZ\\
=& \int q(Z)\ln p(X, Z)dZ +  \int q(Z)\ln\frac{1}{q(Z)}dZ\\
=& \int \prod_{k=1}^K q_k(Z_k) \ln p(X, Z)dZ +  \sum_{k=1}^K\int q_k(Z_k)\ln\frac{1}{q_k(Z_k)}dZ_k.
\end{align*}
$$

The second term is just the entropy of $$q$$, which, given the assumption that it can be decomposed into independent factors, becomes the summation of the entropy for each individual $$q_k$$.

It may not be immediately apparent why this modified formulation is any easier to solve. Nevertheless, let us proceed by making the temporary assumption that among the $$K$$ factors, all are known except for one factor $$q_j$$. Then, we just need to maximize $$\mathcal{L}$$ with respect to $$q_j$$ with all the other factors $$q_{k}, k\not=j$$ as given. The variational lower bound can be rewritten as

$$
\begin{align*}
&\int q_j(Z_j) \mathbb{E}_{q_k, k\not=j}[\ln p(X,Z)]dZ_j + \int q_j(Z_j)\ln \frac{1}{q_j(Z_j)}dZ_j + \text{constant}\\
=&\int q_j(Z_j)\ln\frac{\exp\left( \mathbb{E}_{q_k, k\not=j}[\ln p(X,Z)]\right)}{q_j(Z_j)}dZ_j + \text{constant}
\end{align*}
$$

Since any term that does not involve $$q_j$$ would not affect the solution to $$\arg\max_{q_j}\mathcal{L}(q)$$, we just mark those as constant. Here it comes a key observation: notice how the non-constant term resembles the definition of a negative KL divergence between $$q_j(Z_j)$$ and $$\mathbb{E}\_{q_k, k\not=j}$$ $$\exp\left( \mathbb{E}\_{q_k, k\not=j}[\ln p(X,Z)]\right)$$. The only issue is that $$\exp\left( \mathbb{E}\_{q_k, k\not=j}[\ln p(X,Z)]\right)$$ may not be a proper probability measure that sum/integrate to $$1$$. Luckily, since scaling $$\exp\left( \mathbb{E}\_{q_k, k\not=j}[\ln p(X,Z)]\right)$$ only amounts to adding/subtracting a constant term, we know that $$\mathcal{L}(q)$$ is maximized when

$$
\begin{align*}
q_j(Z_j)  \propto \exp\left( \mathbb{E}_{q_k, k\not=j}[\ln p(X,Z)]\right),
\end{align*}
$$

or more accurately,

$$
\begin{align*}
q_j(Z_j)  = \frac{\exp\left( \mathbb{E}_{q_k, k\not=j}[\ln p(X,Z)]\right)}{\int \exp\left( \mathbb{E}_{q_k, k\not=j}[\ln p(X,Z)]\right) dZ_j}.
\end{align*}
$$

This result tells us that, among the $$K$$ factors, if we have all but one factor fixed, then the optimal solution of that left out function that maximize the variation lower bound (or equivalently, minimizes the KL divergence to the true posterior distribution) can be written in the above form as a function of all the other factors. 

This leads to a nice iterative solution that iteratively visits each factor, and maximize the variational lower bound with respect to the target factor treating all the other factors as known. In special cases, the normalization constant term in the dominator of the above equation could be directly inferred if the numerator term already suggests certain type of known distribution. 

It is interesting to note that, to apply this mean-field-approximation method, one only need to make the assumption on how to partition the latent variable dimensions into disjoint groups, one for each factor, without making any assumption on the detailed function form of any factor. The detail form of the factorized distribution would be obtained as a result of the iterative procedure. 

There is one caveat that we should mention. Looking at the equation above, to find the optimal factor $$q_j$$, assuming all the other are know, we still need to make sure that the expectation $$\mathbb{E}_{q_k, k\not=j}[\ln p(X,Z)]$$ results in tractable form. Given that the expectation itself is a multi-dimensional integral/summation, in general it is hard to guarantee a closed form expression. The expectation may be tractable with specific models and specific ways on which the latent variables are partitioned, which limits the domain where mean-field-approximation could be applied. 

Here we introduced mean field approximation from the perspective of Bayesian inference. As a final remark, it is straightforward to show that it also provide a way to evaluate observed data likelihood and thus can be useful with model-selection as well. According to the identity below, we know that as we maximize $$\mathcal{L}$$, not only do we obtain a variational distribution that is close to the true posterior in the KL divergence sense, we also obtained a surrogate for the log-likelihood, as the lower bound $$\mathcal{L}$$ is a lower bound which gets tighter as it becomes larger.  

$$
\begin{align*}
\ln p(X) = \text{KL}(q || p(Z|X)) + \mathcal{L}(q),
\end{align*}
$$

If we are given $$M$$ models, then one can conduct mean field method on each of them, obtain the corresponding optimized variational lower-bound, and use it as the surrogate to rate the likelihood of each model. We can even combine the prior distribution of the $$M$$ models, if there is any, to obtain a maximum a posterior (MAP) selection of the model.




