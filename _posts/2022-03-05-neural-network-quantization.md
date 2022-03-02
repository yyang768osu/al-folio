---
layout: post
title: A Primer on Neural Network Quantization 
date: 2022-03-01
comments: true
description: how to properly quantize neural networks for efficient hardware inference?
---

Table of content

* TOC
{:toc}

## Quantization basics 

---
#### Conversion of floating point and fixed point
---

<div class="row mt-3">
    <div class="col-sm-1 mt-3 mt-md-0">
    </div>
    <div class="col-sm-10 mt-3 mt-md-0">
        {% include figure.html path="assets/img/blog_img/quantization/quantization.png" class="img-fluid rounded z-depth-1" zoomable=false%}
    </div>
    <div class="col-sm-1 mt-3 mt-md-0">
    </div>
</div>
<div class="caption">
    Quantization
</div>

$$
\begin{align*}
\hat{x} =& s \left(x_\text{int} - z\right)\\
        =& s \left(\texttt{clamp}\left(\left\lfloor x/s\right\rceil +z; n, p\right)-z\right) \\
        =& \texttt{clamp}\left(s\left\lfloor x/s \right\rceil; s(n-z), s(p-z)\right) \\
        =& \texttt{clamp}\left(s\left\lfloor x/s \right\rceil; q_\text{min}, q_\text{max}\right)
\end{align*}
$$


$$
\begin{align*}
\hat{x} =& \texttt{clamp}{\bigg(}\underbrace{s\left\lfloor x/s \right\rceil}_{\substack{\text{rounding}\\\text{error}}}; \underbrace{\overbrace{s(n-z)}^{q_\text{min}}, \overbrace{s(p-z)}^{q_\text{max}}}_{\text{clipping error}}{\bigg)}
= \left\{
  \begin{array}{ll}
  s\left\lfloor x/s \right\rceil & \text{if } q_\text{min} \leq x \leq q_\text{max} \\
  s(n-z) & \text{if } x < q_\text{min} \\
  s(p-z) & \text{if } x > q_\text{max}
  \end{array}
  \right.
\end{align*}
$$


---
#### Fixed point arithmetic
---

$$
\begin{align*}
y =& w x\\
\widetilde{y} =& \hat{w}\hat{x}\\
=& s_w(w_\text{int}-z_w)s_x(x_\text{int}-z_x)\\
=& s_w s_x \left[ (w_\text{int}-z_w)(x_\text{int}-z_x)\right] \\
=& \underbrace{s_w s_x}_{\substack{\text{scale of}\\\text{product}}} {\big[} w_\text{int} x_\text{int} - \underbrace{z_wx_\text{int}}_{\substack{\text{input }x\\\text{dependent}}} - \underbrace{w_\text{int}z_x + z_wz_x}_{\substack{\text{can be}\\\text{pre-computed}}} {\big]}
\end{align*}
$$

<div class="row mt-3">
    <div class="col-sm-2 mt-3 mt-md-0">
    </div>
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.html path="assets/img/blog_img/quantization/requantization.png" class="img-fluid rounded z-depth-1" zoomable=false%}
    </div>
    <div class="col-sm-2 mt-3 mt-md-0">
    </div>
</div>
<div class="caption">
    ReQuantization
</div>

$$
\begin{align*}
\boldsymbol{y} =& \boldsymbol{W} \boldsymbol{x} + \boldsymbol{b}\\
y_i =& \sum_{j}W_{ij}x_j + b_i\\
\tilde{y}_i = & \sum_{j}\hat{W}_{ij}\hat{x}_j + \hat{b}_i\\
=& \sum_{j} s_i^w \left(W_{ij}^\text{int} - z_i^w\right) s^x \left(x_j^\text{int}- z^x\right) + s_i^ws^x b_i^\text{int}\\
=& s_i^wz_i^w {\bigg[}
  \sum_{j}{\big(}
    W_{ij}^\text{int}x_j^\text{int} - z_i^wx_j^\text{int} - W_{ij}^\text{int}z^x + z_i^wz^x
  {\big)}
  +
  b_i^\text{int}
{\bigg]}\\
=& s_i^wz_i^w {\color{blue}\bigg[}
    \underbrace{\sum_{j}{\big(}
    W_{ij}^\text{int}x_j^\text{int}{\big)}}_{\substack{\text{Multiply-Accumulate}\\\text{(MAC)}}} - 
    \underbrace{\sum_{j}{\big(}z_i^wx_j^\text{int}{\big)}}_{\substack{\text{No-op if}\\\text{ weigth offset is }0}} + 
    \underbrace{\sum_{j}{\big(}- W_{ij}^\text{int}z^x + z_i^wz^x
  {\big)}
  +
  b_i^\text{int}}_{
    \substack{\text{Can be pre-computed and }\\\text{pre-loaded to the accumulator}}}
{\color{blue}\bigg]}

\end{align*}
$$


---
#### Quantization simulation and gradient computation
---

<div class="row mt-3">
    <div class="col-sm-2 mt-3 mt-md-0">
    </div>
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.html path="assets/img/blog_img/quantization/quantization_sim.png" class="img-fluid rounded z-depth-1" zoomable=false %}
    </div>
    <div class="col-sm-2 mt-3 mt-md-0">
    </div>
</div>
<div class="caption">
    Quantization simulation
</div>

$$
\begin{align*}
\frac{\partial \hat{x}}{\partial x} =& \left\{
  \begin{array}{ll}
  s \frac{\partial}{\partial x}\left\lfloor x/s\right\rceil \overset{\text{st}}{\approx} 1 &\text{if } q_\text{min} \leq x \leq q_\text{max} \\
  0 & \text{if } x < q_\text{min} \\
  0 & \text{if } x > q_\text{max}
  \end{array}
  \right.\\

\frac{\partial \hat{x}}{\partial s} =& \left\{
  \begin{array}{ll}
  \left\lfloor x/s \right\rceil + s \frac{\partial}{\partial s} \left\lfloor x/s \right\rceil \overset{\text{st}}{\approx}  \left\lfloor x/s \right\rceil - x/s  &\text{if } q_\text{min} \leq x \leq q_\text{max} \\
  n - z  & \text{if } x < q_\text{min} \\
  p - z & \text{if } x > q_\text{max}
  \end{array}
  \right. \\
  
\frac{\partial \hat{x}}{\partial z} =& \left\{
  \begin{array}{ll}
  0 &\text{if } q_\text{min} \leq x \leq q_\text{max} \\
  -s & \text{if } x < q_\text{min} \\
  -s & \text{if } x > q_\text{max}
  \end{array}
  \right.
\end{align*}
$$

---
#### Post-Training-Quantization (PTQ) vs Quantization-Aware-Training (QAT)
---

---
## Quantization techniques 
---
#### Range Estimation Methods
---

* MSE 
  * Solve $$\underset{z,s}{\arg\min} \|x-\hat{x}\|_F^2$$ 
* Cross-Entropy
  * Solve $$\underset{z,s}{\arg\min}\text{ }\texttt{CrossEntropy}(\texttt{softmax}(x),\texttt{softmax}(\hat{x}))$$ 
* Min-Max
  * Find $$z$$ and $$s$$ such that $$q_\text{min} \approx \min x$$ and $$q_\text{max} \approx \max x$$
* BatchNorm (For activation quantization only)
  * Use batch norm statistics to approximate min and max as $$\min x\approx \beta - \alpha\gamma$$ and $$\max x\approx \beta + \alpha\gamma$$. ($$\alpha=6$$ is suggested)

---
#### Cross Layer Equalization
---

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/blog_img/quantization/cross_layer_equalization.png" class="img-fluid rounded z-depth-1" zoomable=false %}
    </div>
</div>
<div class="caption">
    Cross layer equalization
</div>

---
#### Bias Correction 
---

---
#### Adaptive Rounding
---

---
## PTQ and QAT best practices
---
#### PTQ development and debugging pipeline
---

---
#### QAT pipeline
---

---
## Special treatment
---
#### Handling of layers other than Conv and FC
---

---
#### Quantization of Transformers
---
