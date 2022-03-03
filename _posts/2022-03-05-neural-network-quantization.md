---
layout: post
title: A Primer on Neural Network Quantization 
date: 2022-03-01
comments: true
description: how to properly quantize neural networks for efficient hardware inference?
---


> **Disclaimer**: this post is a reading note of the paper [A White Paper on Neural Network Quantization](https://arxiv.org/abs/2106.08295) [[Nagel et al. 2021]](#Nagel_et_al_2021) with additional illustrations and explanations that help my understanding (hopefully can help yours too). The same materials can be found in the paper and the references therein. You are highly encouraged to read the paper first and the references if you want to go into depth for certain topics.

Neural network model quantization enables fast network inference using efficient fixed point arithmetics in typical AI hardware accelerators. Compared with floating point inference, it allows less storage space, smaller memory footprint, lower power consumption, and faster inference speed, all of which are essential for practical edge deployment of deep learning solutions. It is thus a critical step in the model efficient pipeline. In this post, based off [[Nagel et al. 2021]](#Nagel_et_al_2021), we provide a detailed guide to network quantization.  

---

We will start off with the basics in the first section by first going through the conversion of floating point and fixed point and introduce necessary notations. Followed by showing how fixed point arithmetic works in a typical hardware accelerator and the way to simulate it using floating point operation. And lastly define terminology of Post-Training-Quantization (PTQ) and Quantization-Aware-Training (QAT).

The second section covers four techniques for improved quantization performance, which are mostly designed targeting PTQ, with some also serving as a necessary step to get good parameter initialization for QAT.

In the last part, we go over the recommended pipeline and practices of PTQ and QAT as suggested by [the white paper](https://arxiv.org/abs/2106.08295) [[Nagel et al. 2021]](#Nagel_et_al_2021) and discuss how special layers other than Conv/FC can be handled. Here's an outline.

* TOC
{:toc}

---
# Quantization basics 

---
#### Conversion of floating point and fixed point representation
---

Fixed point and floating point are different ways of representing numbers in computing devices. Floating point representation is designed to capture fine precisions (by dividing the bit field into mantissa and exponent) with high bit-width (typically 32 bits or 64 bits), whereas fixed point numbers lives on a fixed width grid that are typically much coarser (typically 4, 8, 16 bits). Due to its simplicity, the circuitry for fixed point arithmetic can be much cheaper, simpler, and more efficient than its floating point counterpart. The term *network quantization* refers to the process of converting neural network models with floating point weight and operations into one with fixed point weight and operations for better efficiency. 

The figure below describes how the conversion from floating point number to fixed point integer can be done. Let us label the start and the end of the fixed point integer number as $$n$$ and $$p$$. For signed integer with bit-width of $$b$$, we have $$[n, p]=[-2^{b-1}, 2^{b-1}-1]$$; for unsigned integer $$[n, p]=[0, 2^{b}-1]$$. In this simple example $$b=4$$, which gives us 16 points on the grid.

[//]: #  The process snaps floating point numbers with fine precisions into a fixed coarse grid with limited range. 

Assume that we have a conversion scheme that maps a floating point $$0$$ to an integer $$z$$, and $$s$$ to $$z+1$$. The mapping of floating point number $$x$$ to its fixed point representation $$x_\text{int}$$ can be described as 

$$x_\text{int} = \texttt{clamp}(\lfloor x/s \rceil; n, p),$$

where $$\lfloor\cdot\rceil$$ denotes rounding-to-nearest operation and $$\texttt{clamp}(\cdot: n, p)$$ clips its input to within $$[n, p]$$

<div class="row mt-3">
    <div class="col-sm-1 mt-3 mt-md-0">
    </div>
    <div class="col-sm-10 mt-3 mt-md-0">
        {% include figure.html path="assets/img/blog_img/quantization/quantization.png" class="img-fluid rounded z-depth-1" zoomable=false%}
    </div>
    <div class="col-sm-1 mt-3 mt-md-0">
    </div>
</div>

The integer $$x_\text{int}$$, when mapped back to the floating point axis, corresponds to 

$$\hat{x} = s\left(x_\text{int} - z\right).$$

By combining the two equations, we can express the quantized floating point value $$\hat{x}$$ as 

$$
\begin{align}
\hat{x} =& s \left(\texttt{clamp}\left(\left\lfloor x/s\right\rceil +z; n, p\right)-z\right) \notag\\
        =& \texttt{clamp}\left(s\left\lfloor x/s \right\rceil; s(n-z), s(p-z)\right) \notag\\
        =& \texttt{clamp}{\Big(}\underbrace{s\left\lfloor x/s \right\rceil}_{\substack{\text{rounding}\\\text{error}}}; \underbrace{q_\text{min}, q_\text{max}}_{\substack{\text{clipping}\\\text{error}}}{\Big)}.
\end{align}
$$

Here we denote $$q_\text{min}\triangleq s(n-z)$$ and $$q_\text{max}\triangleq s(p-z)$$ as the floating point values corresponding to two limits on the fixed point grid. The quantization error of $$x-\hat{x}$$ comes from two competing factors: clipping and rounding. Increasing $$s$$ reduces the clipping error at the cost of increased rounding error. Decreasing $$s$$ reduces rounding error with higher chance for clipping.

To avoid getting "lost in notation" with the many we introduced ($$n$$, $$p$$, $$b$$, $$z$$, $$s$$, $$q_\text{min}$$, $$q_\text{max}$$, $$x$$, $$\hat{x}$$, $$x_\text{int}$$), it is helpful to keep the following in mind:
* $$n$$, $$p$$, and $$b$$ are determined by integer type that is available to us. They describe the hardware constraints and are not something we can control (nor is there a need to control them)
  * For 8-bit signed integer we have $$n=-128$$ and $$p=127$$
  * For 8-bit unsigned integer we have $$n=0$$ and $$p=255$$
  * For bit-width of $$b$$, we have $$p-n+1 = 2^b$$
* Either $$(s, z)$$ or $$(q_\text{min}, q_\text{max})$$ uniquely describes the quantization scheme. Together they are redundant in that one can derive the other. Therefore, when we talk about **quantization parameters**, we refer to either $$(s, z)$$, or equivalently $$(q_\text{min}, q_\text{max})$$. In other words, there are only **two degrees of freedom** when it comes to **quantization parameters**.
  * Derive $$(q_\text{min}, q_\text{max})$$ from $$(s, z)$$:
    * $$q_\text{min} = s(n-z)$$.
    * $$q_\text{max} = s(p-z)$$.
  * Derive $$(s, z)$$ from $$(q_\text{min}, q_\text{max})$$:
    * $$s = (q_\text{max} - q_\text{min}) / 2^b $$.
    * $$z = n - q_\text{min}/s = p - q_\text{max}/s$$.
  * **Quantization parameter** is either $$(s, z)$$ or $$(q_\text{min}, q_\text{max})$$
    * In the context of QAT, we often use $$(s, z)$$ as they are directly trainable
    * In the context of PTQ, we often think about $$(q_\text{min}, q_\text{max})$$
    * **Range estimation** basically refers to estimate of $$(q_\text{min}, q_\text{max})$$
* Among the different variables, $$z$$ and $$x_\text{int}$$ are fixed point number whereas $$x$$, $$\hat{x}$$, $$s$$ are floating point numbers
  * We want to carry out all arithmetic operations in the fixed point domain, so ideally only offset $$z$$ and $$x_\text{int}$$ are involved in hardware multiplication and summation.
* The fact that floating point $$0$$ maps to an integer $$z$$ ensures that there is no quantization error in representing floating point $$0$$
  * This guarantees that zero-padding and ReLU do not introduce quantization error


---
#### Fixed point arithmetic
---

$$
\begin{align*}
y =& w x\\
y^\text{acc} =& \hat{w}\hat{x}\\
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
y_i^\text{acc} = & \sum_{j}\hat{W}_{ij}\hat{x}_j + \hat{b}_i\\
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

Equation (5) in [[Jacob et al. 2018]](#Jacob_et_al_2018)



---
#### Per-tensor vs per-channel weight quantization
---

---
#### Quantization simulation and gradient computation
---


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
# Quantization techniques 
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

$$f(sx) = sf(x)$$

$$
\begin{align*}
&W_{c_3\times c_2}^{(2)} f\left( W_{c_2\times c_1}^{(1)}x\right)\\
=&\left[W_{\cdot 1}^{(2)},\ldots,W_{\cdot c_2}^{(2)}\right] f\left(\left[
  \begin{array}{c}
  W_{1\cdot}^{(1)} \\
  \vdots \\
  W_{c_2\cdot}^{(1)} \\
  \end{array}
  \right]x\right)\\
=&\left[W_{\cdot 1}^{(2)},\ldots,W_{\cdot c_2}^{(2)}\right] \left[
  \begin{array}{c}
  f(W_{1\cdot}^{(1)}x) \\
  \vdots \\
  f(W_{c_2\cdot}^{(1)}x) \\
  \end{array}
  \right]\\
=&\sum_{i:1\to c_2} W_{\cdot i}^{(2)}f\left(W_{i\cdot}^{(1)}x\right)\\
=&\sum_{i:1\to c_2} W_{\cdot i}^{(2)} s_i f\left(\frac{W_{i\cdot}^{(1)}}{s_i}x\right)
\end{align*}
$$

$$
\begin{align*}
                &\max\left| W_{\cdot i}^{(2)} s_i \right| = \max\left|\frac{W_{i\cdot}^{(1)}}{s_i}\right| \\
\Longrightarrow & \max\left| W_{\cdot i}^{(2)}\right| s_i = \max\left|W_{i\cdot}^{(1)}\right|/s_i \\
\Longrightarrow & s_i = \sqrt{\frac{
\max\left|W_{i\cdot}^{(1)}\right|
}{
\max\left| W_{\cdot i}^{(2)}\right|
}}
\end{align*}
$$



<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/blog_img/quantization/cross_layer_equalization.png" class="img-fluid rounded z-depth-1" zoomable=false %}
    </div>
</div>
<div class="caption">
    Cross layer equalization
</div>

[[Nagel et al. 2019]](#Nagel_et_al_2019)
[[Meller et al. 2019]](#meller_et_al_2019)

---
#### Bias Correction 
---

Quantization of weights can lead to a shift in mean value of the output distribution. Specifically, for a linear layer with weight $$W$$ and input of $$x$$, the gap between the mean output of quantized weight $$\hat{W}$$ and its floating point counterpart $$W$$  can be expressed as $$\mathbb{E}[\hat{W}x] - \mathbb{E}[Wx] = (W-\hat{W})\mathbb{E}[x]$$. Given that $$x$$ is the activation of the previous layer, $$\mathbb{E}[x]$$ is often non-zero (e.g., if $$x$$ is the output of the ReLU activation), and thus the gap can be non-zero. 

Luckily, this shift in mean can be easily corrected by absorbing $$(W-\hat{W})\mathbb{E}[x]$$ into the bias (subtract $$(W-\hat{W})\mathbb{E}[x]$$ from bias). Since $$W$$ and $$\hat{W}$$ are known after the quantization, we only need to estimate $$\mathbb{E}[x]$$, which can come from two sources

* If there is a small amount of input data, it can be used to get an empirical estimate of $$\mathbb{E}[x]$$
* If $$x$$ is the output of a BatchNorm + ReLU layer, we can use the batch norm statistics to derive $$\mathbb{E}[x]$$

[[Nagel et al. 2019]](#Nagel_et_al_2019)

---
#### Adaptive Rounding
---

In PTQ, after the quantization range $$[q_\text{min}, q_\text{max}]$$ (or equivalently, step size $$s$$ and offset $$z$$) of a weight tensor $$W$$ is determined, the weight will be **round** to its **nearest** value on the fixed point grid. Rounding to the **nearest** quantized value is such an apparently right operation that we don't think twice about it. However, there is a valid reason why we may consider otherwise. 

Let us first define a more flexible form of quantization $$\widetilde{W}$$ where we can control to round up or down with a binary auxillary variable $$V$$:

$$
\begin{align*}
\text{Round to nearest } \widehat{W} =& s \left\lfloor W/s\right\rceil, \\
\text{Round up or down } \widetilde{W}(V) =& s \left(\left\lfloor W/s\right\rfloor + V\right).
\end{align*}
$$

Note that we changed $$\lfloor\cdot\rceil$$ into $$\lfloor\cdot\rfloor$$ and ignored clamping for notational clarity. Rounding-to-nearest minimizes the mean-square-error (MSE) of the quantized value with its floating point value, i.e., 

$$
\widehat{W} = \min_{V\in [0, 1]^{|V|}} \left\| W - \widetilde{W}(V) \right\|_F^2.
$$

Instead of minimizing MSE of the quantized weight, a better target is to minimize the MSE of the activation, which reduces the effect of quantization from an input-output stand point:

$$
\min_{V\in [0, 1]^{|V|}} \left\|f(Wx) - f\left(\widetilde{W}(V)\hat{x}\right)\right\|_F^2.
$$

The method of determined whether to round up or round down by optimizing the above objective is called Adaptive Rounding or AdaRound proposed by [[Nagel et al. 2020]](#Nagel_et_al_2020). Note that optimization of the above objective only requires a small amount of representative input data. Please refer to it for details regarding how this integer optimization problem can be solved with relaxation and annealed regularization term that encourage $$V$$ to converge to 0/1. 

Alternatively, one can use straight-through estimator (STE) to directly optimize for the quantized weight, which allows more flexible quantization beyond just rounding up or down. In the table below, we can see that AdaRound outperforms this STE approach, likely due to biased gradient of STE. 

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/blog_img/quantization/adaround.png" class="img-fluid rounded z-depth-1" zoomable=false %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
 
    </div>
</div>
<div class="caption">
    Figure 5 from <a href="#Nagel_et_al_2020">[Nagel et al. 2020]</a>: ImageNet validation on ResNet18. 
</div>

To summarize, AdaRound is an effective PTQ weight quantization technique, with the following characteristics/limitations:
* It requires the access of a small amount of representative input data (no label needed)
* It is only a weight quantization technique
* It is applied after the range ($$q_\text{min}$$ and $$q_\text{max}$$, or equivalently $$s$$ and $$z$$) is determined. 
* When QAT is applied, AdaRound becomes irrelevant. 


---
# PTQ and QAT best practices
---
#### PTQ development and debugging pipeline
---

---
#### QAT pipeline
---

---
# Treatment of special layers
---
#### Handling of layers other than Conv and FC
---

---
#### Quantization of Transformers
---


---
# References
---
- <a name="Nagel_et_al_2021"></a> **\[Nagel et al. 2021\]**  Markus Nagel, Marios Fournarakis, Rana Ali Amjad, Yelysei Bondarenko, Mart van Baalen, Tijmen Blankevoort
 "*[A White Paper on Neural Network Quantization](https://arxiv.org/abs/2106.08295)*", Arxiv June 2021
- <a name="Nagel_et_al_2020"></a> **\[Nagel et al. 2020\]** Markus Nagel, Rana Ali Amjad, Mart Van Baalen, Christos Louizos, Tijmen Blankevoort "*[Up or Down? Adaptive Rounding for Post-Training Quantization](http://proceedings.mlr.press/v119/nagel20a.html)*", ICML 2020
- <a name="Nagel_et_al_2019"></a> **\[Nagel et al. 2019\]** Markus Nagel, Mart van Baalen, Tijmen Blankevoort, Max Welling, "*[Data-Free Quantization Through Weight Equalization and Bias Correction](https://openaccess.thecvf.com/content_ICCV_2019/html/Nagel_Data-Free_Quantization_Through_Weight_Equalization_and_Bias_Correction_ICCV_2019_paper.html)*", ICCV 2019
- <a name="meller_et_al_2019"></a> **\[Meller et al. 2019\]** Eldad Meller, Alexander Finkelstein, Uri Almog, Mark Grobman, "*[Same, Same But Different: Recovering Neural Network Quantization Error Through Weight Factorization](http://proceedings.mlr.press/v97/meller19a.html)*", ICML 2019
- <a name="Jacob_et_al_2018"></a> **\[Jacob et al. 2018\]** Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Hartwig Adam, Dmitry Kalenichenko, *"[Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://openaccess.thecvf.com/content_cvpr_2018/html/Jacob_Quantization_and_Training_CVPR_2018_paper.html)"*, CVPR 2018