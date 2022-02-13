---
layout: post
title: "Markov Chain Monte Carlo: Gibbs, Metropolis-Hasting, and Hamiltonian"
date: 2019-07-06
comments: true
description: a primer on Gibbs sampling, Metropolis-Hasting, and Hamiltonian MC
---

A fundamental problem in statistical learning is to compute the expectation with respect to some target probability distribution $$\pi$$

$$
\begin{align*}
\mathbb{E}_\pi\left[f\right] \triangleq \int \pi(x) f(x) dx.
\end{align*}
$$

There are two difficulties in the evaluation of the above (1) often $$\pi(\cdot)$$ is available to us only as a form of unnormalized probability, i.e., it can be evaluated only up to a normalizing constant (2) even if $$\pi(\cdot)$$ can be evaluated exactly, it is often hard to directly generate samples from it (e.g., for high-dimensional space). 

One example application is Bayesian inference, where the posterior probability of the latent $$\pi(x\|D)$$ is available only in the form of prior $$\pi(x)$$ times likelihood $$\pi(D\|x)$$ up to the unknown normalizing constant of $$\pi(D)$$, and we would like to either sample or obtain the expectation with respect to the posterior probability. 

The idea of Markov Chain Monte Carlo (MCMC) is to construct a Markov chain whose stationary distribution is exactly the target distribution with easy-to-sample transition kernels. One could then start with a random initial state, and yield samples by simply running the transitions and use the generated samples after the chain reaches steady state for the Monte Carlo evaluation of the expectation.

For the design of such Markov chain, all methods that I encountered utilize the following theorem
>An irreducible and aperiodic Markov chain with transition probability $$P$$ has stationary distribution of $$\pi$$ if it satisfies \begin{align} 
\pi(x)P(x'\|x) = \pi(x')P(x\|x') \notag
\end{align}

The game, then, is to design $$P$$ for which the above equality holds. In this article, we will go through three MCMC methods with different ways in the design of $$P$$, namely **Gibbs sampling**, **Metropolis-Hastings**, and **Hamiltonian Monte Carlo** (HMC). 

As a side note, it is worth pointing out that the above equation, referred to as _detailed balance equation_, is a sufficient but not necessary condition for a Markov chain to have stationary distribution $$\pi$$. It defines a special case of Markov chain called reversible Markov chain. The detailed balance equation should be contrasted with _global balance equation_ below, which all Markov chains with stationary distribution $$\pi$$ satisfy. Then it shouldn't be surprising that global balance equation can be easily derived from detailed balance equation (by summing over $$x'$$ on both sides of the above equation) but not the other way around. 

$$
\begin{align*}
\pi(x) = \sum_{x'} \pi(x')P(x'|x).
\end{align*}
$$

## Gibbs sampling

In Gibbs sampling, the transition probability $$P$$ is defined as the following

$$
\begin{align*}
P\left(x'|x\right)=\left\{
\begin{array}{ll}
\frac{1}{d}\pi\left(x'_j|x_1, \ldots, x_{j-1}, x_{j+1}, \ldots, x_d\right) & \text{if there exits }j\text{ such that }x_i'=x_i\text{ for }i\not=j.\\
0&\text{otherwise.}
\end{array}
\right.
\end{align*}
$$

The state $$x$$ is a vector of dimension $$d$$, and the transition probability from state $$x$$ to state $$x'$$ is non-zero when they differ by only one dimension, say dimension $$j$$, and the transition probability is designed to be the conditional probability of $$x'_j$$, given all the other dimensions fixed, scaled by $$1/d$$. This corresponds to a transition scheme where we uniformly pick a dimension $$j$$, and then randomly sample a value in dimension $$j$$ following the conditional distribution. Detailed balance equation holds with such design

$$
\begin{align*}
&\pi(x)P(x'|x)\\
=&\frac{1}{d}\pi(x)\pi(x'_j|x_1, \ldots, x_{j-1}, x_{j+1}, \ldots, x_d)\\
=& \frac{1}{d}\pi(x)\pi(x')/\sum_{z}\pi(x_1, \ldots, x_{j-1}, z, x_{j+1}, \ldots, x_d)\\
=&\frac{1}{d}\pi(x)\pi(x')/\sum_{z}\pi(x'_1, \ldots, x'_{j-1}, z, x'_{j+1}, \ldots, x'_d)\\
=&\frac{1}{d}\pi(x')\pi(x_j|x'_1, \ldots, x'_{j-1}, x'_{j+1}, \ldots, x'_d)\\
=&\pi(x')P(x|x').
\end{align*}
$$

The premise of Gibbs sampling is that the conditional distribution of one dimension given the rest is much easier to normalize and sample from. It is quite limited though, in the sense that the transition can never go very far in each step -- only one dimension can be changed at a time. As a consequence, the transition matrix is quite sparse and the Markov chain may suffer from very large mixing time (time to stationary distribution) and it may not scale well with large dimensional space. 

## Metropolis Hastings
 
Metropolis Hastings algorithm is a much more general version of Gibbs; in fact it encompasses both Gibbs sampling and Hamiltonian MC as special realizations. The basic idea is to construct the transition distribution from a flexible form of proposal distribution $$g(x'\|x)$$, corrected by a _acceptance ratio_ term $$A(x',x)$$  to guarantee reversibility in time. Specifically, the acceptance ratio is chosen to enforce the detailed balance equation

$$
\begin{align*}
\pi(x) g(x'|x) A(x', x) = \pi(x') g(x|x') A(x, x').
\end{align*}
$$

The actual transition probability is then $$P(x'\|x) \triangleq g(x'\|x) A(x', x)$$, corresponding to a sampling scheme where we first sample from $$g(x'\|x)$$ to have a candidate next state $$x'$$, and then accept this candidate with probability $$A(x', x)$$. If the candidate state is rejected, the next state will remain the same as the current state. For an arbitrary proposal distribution $$g$$, from the above equation, we have

$$
\begin{align*}
\frac{A(x', x)}{A(x, x')} = \frac{\pi(x')g(x|x')}{\pi(x)g(x'|x)}.
\end{align*}
$$

To reduce the mixing time of the Markov chain, it is desirable to maximize the acceptance ratio $$A$$. This means that we want to set either $$A(x',x)$$ or $$A(x, x')$$ to be $$1$$ for any pair of $$x$$ and $$x'$$, resulting in the expression below

$$
\begin{align*}
A(x', x) = \min\left\{1, \frac{\pi(x')g(x|x')}{\pi(x)g(x'|x)}\right\}.
\end{align*}
$$

In the above equation, since $$\pi$$ appear in both numerator and denominator, we can easily work with unnormalized probability distribution, as long as it can be evaluated efficiently for each data point.

Metropolis-Hasting algorithm itself is just a MCMC framework; it still relies on a good choice of proposal distribution to perform well. The design of $$g$$ can be problem specific and is the _art_. The clear optimal choice of is $$g(x'\|x)=\pi(x)$$, which degenerates to the direct sampling of $$\pi$$ with acceptance ratio of $$1$$.

## Hamiltonian Monte Carlo   

Let's now image a high dimensional surface for which the potential energy at each point $$x$$ is defined as $$V(x)\triangleq -\log\pi(x)$$. Here we introduce an auxiliary variable $$p$$ with the same dimension as $$x$$, and interpret the pair of variable $$(x, p)$$ as describing the position and momentum of an object on the high dimensional space. 

The kinetic energy of the object with mass $$m$$ and momentum $$p$$ is known as $$K(p)=\frac{p^2}{2m}$$ (e.g., $$\frac{1}{2}mv^2 = (mv)^2/2m$$). We now construct a joint probability distribution of $$(x,p)$$ as 

$$
\begin{align*}
\pi(x, p) = \frac{1}{Z}e^{-V(x)-K(p)} = \frac{1}{Z} e^{\log\pi(x)}e^{p^2/2m} = \frac{1}{Z}\pi(x)\mathcal{N}\left(p|0, \sqrt{m}\right).
\end{align*}
$$

Two remarks here: (1) The joint probability defined above is a function of the total energy $$V(x) + K(p)$$ (potential energy plus kinetic energy) of the imaginary object. (2) Since the marginal distribution of $$\pi(x, p)$$ with respect to $$x$$ is $$\pi(x)$$, if we can construct an effective MCMC algorithm for $$(x, p)$$, we then obtain an MCMC algorithm for $$x$$ by discarding $$p$$. 

The key in Hamiltonian MC is to use Hamiltonian mechanism as a way to obtain new candidate state (corresponding to proposal $$g$$ in Metropolis-Hastings).  Hamiltonian mechanics is an alternative reformation of the classic Newtonian mechanics describing Newton's second law of motion. It characterizes the time evolution of the system in terms of location $$x$$ and momentum $$p$$, with the conservation of the sum of potential energy $$V(x)$$ and Kinetic energy of $$K(p)$$, a.k.a. Hamiltonian $$\mathcal{H}(x, p) \triangleq V(x) + K(p)$$, through the following differential equations

$$
\begin{align*}
\frac{d p}{dt} =& -\frac{\partial \mathcal{H}}{\partial x} &\text{force equals to negative gradient of potential energy}\\
\frac{d x}{dt} =& \frac{\partial \mathcal{H}}{\partial p} &\text{velocity equals to derivative of kinetic energy w.r.t. momentum}
\end{align*}
$$

By solving the path of $$(x, p)$$ according to Hamiltonian mechanics, we are essentially traversing along the contour for which $$\pi (x, p)$$ is fixed. This provide a very nice way of coming up with a proposal function $$g(x', p'\| x, p)$$ without having to reject any candidate. In other words, if we start with the point $$(x, p)$$ and derive the system state $$(x_\tau, p_\tau)$$ after a period of time $$\tau$$ , we then know that $$\pi(x, p) = \pi(x_\tau, p_\tau)$$. If we further apply a negation in the momentum, then the proposal function is reversible. 

$$
\begin{align*}
x, p \xrightarrow[]{\substack{\text{Hamiltonian}\\\text{mechanics}}} x_\tau, p_\tau \xrightarrow[]{\substack{\text{negate}\\\text{momentum}}} x_\tau, -p_\tau  = x', p'\\
x', p' \xrightarrow[]{\substack{\text{Hamiltonian}\\\text{mechanics}}} x'_\tau, p'_\tau \xrightarrow[]{\substack{\text{negate}\\\text{momentum}}}x'_\tau, -p'_\tau  = x, p
\end{align*}
$$

If we have perfect solver for the differential equation, then according to Equation (2) there is no need to reject any transition proposal. However, in reality the differential equation can only be solved in approximation with error, and thus $$\pi(x, p)\not=\pi(x', p')$$, meaning that the acceptance ratio is not strictly $$1$$ and certain fraction of the transition proposal would be rejected. It is worth noting that the method for computing the solution to the differential equation should still be reversible to respect the detailed balance equation. One hidden condition for such transition to be feasible is that the potential energy $$V(\cdot)$$ has to be differentiable, implying that the target distribution $$\pi(\cdot)$$ should be differentiable. 

So now we have defined a proposal function according to Hamiltonian mechanics, which leads to large acceptance ratio. Are we done here? Not yet. If we stop here, then the Markov chain we defined is reducible, i.e., not every state is accessible from an initial state. In fact, we only have pairwise transition in the Markov chain. To ensure the sampling of the entire space, another proposal distribution $$g_2$$ is introduced, taking advantage of the fact that $$\pi(x, p)$$ has factorized form for which $$p$$ follows a zero-mean normal distribution -- the proposal distribution $$g_2$$ simply samples the momentum value $$p$$ from the corresponding marginal distribution. For such proposal, the corresponding acceptance ratio is $$1$$

$$
\begin{align*}
A((x, p'), (x, p)) = \min\left\{1, \frac{\pi(x, p')g_2(p|p')}{\pi(x, p)g_2(p'|p)}\right\} = \min\left\{1, \frac{\pi(x)}{\pi(x)}\right\}=1 .
\end{align*}
$$

Now we concatenate the above two proposals to have the final form of Hamiltonian MC sampling

$$
\begin{align*}
x, p_0 \xrightarrow[]{\substack{\text{resample}\\\text{momentum}}} x, p 
\xrightarrow[]{\substack{\text{Hamiltonian}\\\text{mechanics}}} x_\tau, p_\tau 
\xrightarrow[]{\substack{\text{negate}\\\text{momentum}}} x_\tau, -p_\tau  = x', p'.
\end{align*}
$$

Since every time after applying the Hamiltonian mechanics the momentum is resampled, we can ignore the momentum negation operation, leading to the following

$$
\begin{align*}
x \xrightarrow[]{\substack{\text{resample}\\\text{momentum}}} x, p 
\xrightarrow[]{\substack{\text{Hamiltonian}\\\text{mechanics}}} x_\tau, p_\tau 
\xrightarrow[]{\substack{\text{discard}\\\text{momentum}}} x_\tau = x',
\end{align*}
$$

and the corresponding acceptance ratio is

$$
\begin{align*}
A((x_\tau, p_\tau), (x, p)) = \min\left\{1, \frac{\pi(x_\tau, p_\tau)}{\pi(x, p)}\right\} =  \min\left\{1, e^{\mathcal{H}(x, p) - \mathcal{H}(x_\tau, p_\tau)}\right\}.
\end{align*}
$$


