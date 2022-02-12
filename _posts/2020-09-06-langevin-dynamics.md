---
layout: post
title: An Introduction of Langevin Dynamics for Bayesian Inference
date: 2020-09-06
comments: true
description: Stochastic differential equation, Fokker Plank equation, and their connections to Bayesian inference
---

In this post we visit some technical details centered around Langevin Dynamics in the context of stochastic Bayesian learning, assuming minimal background on conventional calculus and Brownian motion. Starting with quadratic variation, we gradually show how Ito's Lemma and Fokker-Planck equation can be derived. Using Fokker-Planck equation, it is revealed that an Langevian dynamic can be used as a MCMC method to generate samples from an un-normalized distribution. Lastly, [stochastic gradient Langevin dynamics](https://www.ics.uci.edu/~welling/publications/papers/stoclangevin_v6.pdf) method is discussed.  

The following materials are taken as references:

* [UC-Davis Lecture Notes on Applied Mathematics](https://www.math.ucdavis.edu/~hunter/m280_09/ch5.pdf)
* [MIT Topics in Mathematics with Applications in Finance Lecture 17: Stochastic Processes II](https://www.youtube.com/watch?v=PPl-7_RL0Ko)
* [MIT Topics in Mathematics with Applications in Finance Lecture 18: Itō Calculus](https://www.youtube.com/watch?v=Z5yRMMVUC5w)

## Quadratic Variation
For Brownian motion $$B_t$$, 
we know that $$B\left(\frac{i+1}{N}T\right) - B\left(\frac{i}{N}T\right)$$ for different index $$i$$ are i.i.d. with distribution $$\mathcal{N}\left(0, \frac{T}{N}\right)$$. The following holds by strong law of large numbers.

$$
\begin{align*}
\lim_{N\to\infty}\sum_{i=1}^N \left(B\left(\frac{i+1}{N}T\right) - B\left(\frac{i}{N}T\right)\right)^2&=T \text{ a.s.} \\
\end{align*}
$$

The above can be written in differential form as 

$$
\begin{align*}
\int (dB)^2 &= \int dt\\
(dB)^2 &= dt\\
\end{align*}
$$

which is known as quadratic variation. This means that the second order term of Taylor expansion involving $$B_t$$ scales as $$O(t)$$ instead of $$o(t)$$, the implication of which is detailed in Ito's Lemma below.

## Ito's Lemma

Suppose we want to compute $$f(B_t)$$ for some smooth function $$f$$. By Taylor expansion, the infinitesimal difference can be expressed as

$$
\begin{align*}
f(B_{t+\Delta t}) - f(B_t) &= f'(B_t) (B_{t+\Delta t} - B_t) + \frac{f''(B_t)}{2}\left(B_{t+\Delta t}-B_t\right)^2 \\
                  \text{(differential form) }      df &= f'(B_t) dB_t + \frac{f''(B_t)}{2}\left(dB_t\right)^2 \\
                  \text{(quadratic variation) }      df &= f'(B_t) dB_t + \frac{f''(B_t)}{2} dt\\
                       \frac{df}{dt} &= f'(B_t) \frac{dB_t}{dt} \color{red}{+ \frac{f''(B_t)}{2}}
\end{align*}
$$

The above equation is a naive version of Ito's Lemma, the basis of Ito's calculus. Note how it differs from conventional calculus by having the second term in red, as a direct consequence of quadratic variation. 

Let us now look at a more advanced version of Ito's Lemma, with the goal of obtaining the differential form of $$f(x_t, t)$$ where $$x_t$$ is a stochastic process defined with the following stochastic differential equation

$$
\begin{align*}
dx_t = \mu(x_t)dt + \sigma dB_t
\end{align*}
$$

Similarly as before, let's apply Taylor expansion on the infinitesimal difference of $$f$$

$$
\begin{align*}
f(x+\Delta x, t+\Delta t) - f(x, t) &= \frac{\partial f}{\partial t} \Delta t + \frac{\partial f}{\partial x} \Delta x + \frac{1}{2}\left[
\frac{\partial^2 f}{\partial t^2}\Delta t^2 + 2\frac{\partial^2 f}{\partial t \partial x} \Delta t \Delta x + \frac{\partial^2 f}{\partial x^2}(\Delta x)^2
\right] \\
\text{(differential form) } d f &= \frac{\partial f}{\partial t}dt + \frac{\partial f}{\partial x} dx_t +\frac{1}{2}\left[
o(dt) + o(dt) + \frac{\partial^2 f}{\partial x^2}(dx_t)^2
\right] \\
\text{(substitute $dx_t$) } d f &= \frac{\partial f}{\partial t}dt + \frac{\partial f}{\partial x} (\mu(x_t)dt +\sigma dB_t) +\frac{1}{2}
\frac{\partial^2 f}{\partial x^2}(\mu(x_t)^2(dt)^2 + 2\mu(x_t)\sigma dt dB_t + \sigma^2 (dB_t)^2)
\\
\text{(quadratic variation) } d f &= \left(\frac{\partial f}{\partial t} + \mu(x_t)\frac{\partial f}{\partial x} + \color{red}{\frac{1}{2}\sigma^2\frac{\partial^2 f}{\partial x^2}}\right)dt + \sigma\frac{\partial f}{\partial x} dB_t
\end{align*}
$$

Again, the red term highlights the difference to conventional calculus. In the special when $$f$$ is not a function of $$t$$, the above can be reduced to 

$$
\begin{align*}
 d f &= \left(\mu(x_t)f'(x) + \color{red}{\frac{1}{2}\sigma^2 f''(x)}\right)dt + \sigma f'(x) dB_t,
\end{align*}
$$

which is used in the derivation of Fokker-Planck equation in the next section.

## Fokker-Planck equation 

For a stochastic process $$x$$ that is defined as $$dx_t = \mu(x_t) dt + \sigma dB_t$$, we are interested in how the distribution $$p_t$$ of $$x_t$$ evolves over time. For an arbitrary smooth function $$f$$, the following holds

$$
\begin{align*}
\frac{d}{dt}\mathbb{E}\left[f(x_t)\right] = \left\{
\begin{array}{ll}
\int f(x) \frac{d}{dt}p_t(x) dx & \text{ $f(x)$ viewed as a function of $x$ sampled from $p_t$} \\
\mathbb{E}\left[\frac{d}{dt}f(x_t)\right] & \text{ $f(x_t)$ viewed as a function of stochastic process $x_t$}\\
\end{array}
\right.
\end{align*}
$$

The second expression can be evaluated with Ito's Lemma. 

$$
\begin{align*}
&\mathbb{E}\left[\frac{d}{dt}f(x_t)\right]\\
\text{(Ito's Lemma) }=&\mathbb{E}\left[\mu(x_t)f'(x_t)+\frac{1}{2}\sigma^2f''(x_t) + \sigma f'(x_t) \frac{dB_t}{dt}\right] \\
\text{($dB_t$ has mean $0$) }=&\mathbb{E}\left[\mu(x_t)f'(x_t)+\frac{1}{2}\sigma^2f''(x_t) \right] \\
\text{(using $x_t\sim p_t$) }=&\int\left[\mu(x)f'(x)+\frac{1}{2}\sigma^2f''(x) \right]p_t(x)dx \\
\text{(integration by part) }=&-\int f(x)\frac{\partial (\mu(x)p_t(x)) }{\partial x}dx+\frac{1}{2}\sigma^2\int f(x)\frac{\partial^2 p_t(x)}{\partial x^2} p_t(x)dx
\end{align*}
$$

Combining the above two and cancelling out the arbitrary function $$f$$, we obtain Fokker-Planck equation below.

$$
\begin{align*}
\frac{d}{dt}p_t = -\frac{\partial}{\partial x}\left(\mu(x)p_t(x)\right)+\frac{1}{2}\sigma^2\frac{\partial^2}{\partial x^2}p_t(x)
\end{align*}
$$

## Langevin Dynamics

let $$\mu(x) = -u'(x)$$ for some function $$u(x)$$, then the corresponding stochastic process is defined as $$dx_t = -u'(x_t) dt + \sigma dB_t$$, often referred as over-damped Langevin process. Using Fokker-Planck equation, we know that 

$$
\begin{align*}
p(x) \propto e^{-2/\sigma^2 u(x)}
\end{align*}
$$

is the stationary distribution of $$x_t$$. 

$$
\begin{align*}
&\frac{\partial}{\partial x}\left(u'(x)p(x)\right)+\frac{1}{2}\sigma^2\frac{\partial^2}{\partial x^2}p(x) \\
=&u''(x)p(x)-\frac{2}{\sigma^2}(u'(x))^2 p(x) + \frac{1}{2}\sigma^2\left(-\frac{2}{\sigma^2}u''(x)p(x) + \frac{4}{\sigma^4}(u'(x))^2 p(x)\right)=0
\end{align*}
$$

### Langevin MCMC

The fact that Langevin process $$dx_t = -u'(x_t) dt + \sigma dB_t$$ converges to a stationary distribution $$p(x) \propto e^{-2/\sigma^2 u(x)}$$ lends itself as a suitable Markov chain Monte Carlo method. Specifically, to obtain samples from a un-normalized density function $$\bar{p}(x)$$, we just need to run the following Langevin process from a random starting point till it reaches steady state distribution

$$
\begin{align*}
dx_t = \nabla_x \log \bar{p}(x) dt + \sqrt{2} dB_t
\end{align*}
$$

Discretized sample path of Langevin process can be generated with Euler method 

$$
\begin{align*}
x_{k+1}  = x_k  + \nabla_x \log \bar{p}(x_k) \epsilon + \sqrt{2\epsilon}\xi_k
\end{align*}
$$

Since the discretization is only an approximation to the original continuous stochastic process, it does not in itself lead to desired stationary distribution (unless $$\epsilon$$ becomes infinitesimal) and thus should be corrected by Metropolis-Hastings to enforce detailed balance condition. 

One lingering question is: does the discretization of Langevin dynamics satisfy detailed balance equation in $$\epsilon\to0$$ asymptote? The fact that it converges to a desirable distribution does not indicate that it is a time-reversible Markov chain. Even thought it is claimed by some source that the asymptotic acceptance ratio approaches 1, I was not able to show that it is the case and is stuck at the following derivation.

$$
\begin{align*}
&\frac{\bar{p}(x)P(x\to x')}{\bar{p}(x')P(x'\to x)} = \frac{
\bar{p}(x)\mathcal{N}\left(x'-x-\nabla_x \bar{p}(x)\tau|0, 2\tau\right)
}{
\bar{p}(x')\mathcal{N}\left(x-x'-\nabla_x \bar{p}(x')\tau|0, 2\tau\right)
}\\
=& 
\frac{
\bar{p}(x)e^{(x'-x)\nabla_x \bar{p}(x)/2 + o(\tau)}
}{
\bar{p}(x')e^{(x-x')\nabla_x \bar{p}(x')/2 + o(\tau)} 
}
=
\frac{
\bar{p}(x)e^{(x'-x)\nabla_x \frac{\bar{p}(x)+\bar{p}(x')}{2} + o(\tau)}
}{
\bar{p}(x') 
}
\end{align*}
$$

### Relevance to Bayesian Inference

In Bayesian inference we deal with a prior distribution $$p_\text{prior}(\theta)$$ for some latent parameter $$\theta$$ and a likelihood term $$p_\text{likelihood}(\mathcal{D}\|\theta)$$ of the dataset $$\mathcal{D}$$ given the latent parameter, and the goal is to obtain samples according to the posterior probability $$p_\text{post}(\theta\|\mathcal{D}) = p_\text{prior}(\theta)p_\text{likelihood}(\mathcal{D}\|\theta)/p(\mathcal{D})$$. Since the constant marginal likelihood term $$p(\mathcal{D})=\int p_\text{likelihood}(\mathcal{D}\|\theta)p_\text{prior}(\theta)d\theta$$ is often intractable, we are left with a un-normalized poster probability $$p_\text{post}\propto p_\text{prior}p_\text{likelihood}$$. To sample from it, we can simply construct and run the following stochastic process

$$
\begin{align*}
d\theta_t = \left(\nabla_\theta \log p_\text{prior}(\theta) + \nabla_\theta \log p_\text{likelihood}(\mathcal{D}|\theta)\right) dt + \sqrt{2} dB_t
\end{align*}
$$

Hereafter we use the notation of $$x$$ to indicate elements in the dataset $$x\in\mathcal{D}$$, $$\theta$$ to denote the hidden parameter for which we want to conduct Bayesian inference, and drop the subscript to different $$p$$ as they can be differentiated by their arguments.


## Stochastic Gradient Langevin Dynamics (SGLD)

Discretizing Langevin dynamics with step size of $$\epsilon_t$$ leads to the following update rule

$$
\begin{align*}
\Delta \theta = \epsilon_t \left(\nabla_\theta \log p(\theta) + \nabla_\theta \log p(\mathcal{D}|\theta)\right) + \sqrt{2 \epsilon_t} \xi_t, \text{ where }\xi_t\sim\mathcal{N}(0, 1)
\end{align*}
$$

If we have $$\sum_t\epsilon_t = \infty$$ and $$\sum_t\epsilon^2 <\infty$$ then asymptotically the discretization error will become negligible and the update rule approaches the corresponding Langevin dyanmics, resulting in a sequence of $$\theta_t$$ that converges to the posterior distribution $$p(\theta\|\mathcal{D})$$.

An interesting and clever observation made by [stochastic gradient Langevin dynamics](https://www.ics.uci.edu/~welling/publications/papers/stoclangevin_v6.pdf) paper is that the convergence will hold even if we use mini-batches of the data to estimate the gradient of $$\nabla_\theta \log p(\mathcal{D}\|\theta)$$. 

$$
\begin{align*}
\nabla_\theta \log p(\mathcal{D}|\theta) \approx \frac{N}{n}\sum_{i=1}^n\nabla_\theta \log p(x_{t,i}|\theta)
\end{align*}
$$

The insight is that the stochastic error introduced from using mini-batches instead of the whole dataset dies out much faster than the added Gaussian noise as the $$\epsilon_t$$ decreases, so it does not change the asymptotical behavior of the update rule. Specifically, the randomness coming from the stochastic estimate of $$\nabla_\theta \log p(\mathcal{D}\|\theta)$$ has a variance that scales as $$\epsilon_t^2$$ since it is multiplied with $$\epsilon_t$$. In comparison, the variance of the added Gaussian noise scales linearly as $$\epsilon_t$$.

$$
\begin{align*}
\Delta \theta =\underbrace{ \underbrace{ \underbrace{\epsilon_t \frac{N}{n}\sum_{i=1}^n\nabla_\theta \log p(x_{t, i}|\theta)}_{\text{gradient step towards ML target}} +\epsilon_t \nabla_\theta \log p(\theta)}_{\text{gradient step towards MAP target}} + \sqrt{2 \epsilon_t} \xi_t}_{\text{stochastic gradient Langevin dynamics for posterior sampling}} , \text{ where }\xi_t\sim\mathcal{N}(0, 1)
\end{align*}
$$

Given that stochastic Langevin dynamics converges to the desired distribution as $$\epsilon_t\to0$$, we do not need to carry out Metropolis-Hastings to reject samples. This is crucial in simplifying the algorithm, since evaluation of rejection/acceptance rate is computed at every step and it depends on the evaluation of $$p(\theta)p(\mathcal{D}\|\theta)$$ which can only be computed after traversing the whole dataset.

As a closing remark, if we use the posterior sampling for the estimation of the expectation of some function $$f$$, it is recommended in [stochastic gradient Langevin dynamics](https://www.ics.uci.edu/~welling/publications/papers/stoclangevin_v6.pdf) that the following equation be used. 

$$
\begin{align*}
\mathbb{E}[f(\theta)] = \frac{\sum_t \epsilon_t f(\theta_t)}{\sum_t \epsilon_t}
\end{align*}
$$

with the intuition that each $$\theta_t$$ will contribute an effective sample size proportional to $$\epsilon_t$$.



