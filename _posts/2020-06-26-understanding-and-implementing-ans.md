---
layout: post
title: Understanding and Implementing Asymmetric Numeral System (ANS)
date: 2020-06-26
comments: true
description: an introduction of ANS and its implementation
---

Denote an alphabet with $$N$$ different symbols as $$\mathcal{A}=\{0, 1, \ldots, N-1\}$$. Let us consider a source coding algorithm where a sequence of these symbols are encoded into a sequence of bits, which is represented by an integer $$s$$, and assume that we can decode each symbol sequentially by breaking down $$s$$ into one symbol $$x\in\mathcal{A}$$ and a new integer $$s'\in\mathbb{N}$$ capturing information of the remaining symbols. The encoding and decoding operation can be represented by the following `push` and `pop` operation:

$$
\begin{align*}
\text{encode/push  }e:& \mathbb{N}\times\mathcal{A} \to \mathbb{N}\\
\text{decode/pop   }d:& \mathbb{N} \to \mathbb{N}\times\mathcal{A}
\end{align*}
$$

There are two design goals for such a codec:
1. validity: for valid encoding and decoding, we want to make sure that $$e$$ and $$d$$ are bijections and inverse of each other ($$e=d^{-1}$$).
2. efficiency: for coding efficiency, we want the final codeword length to approach the entropy of the data source

Let us focus on the decoding process $$d$$ and denote the mapping from an integer $$s$$ using $$d$$ as $$d(s) = s', x$$ where $$x\in\mathcal{A}$$. From information theory we know that, the information content (the amount of surprise) of the event of encountering $$x$$ with probability $$P(x)$$ can be expressed as $$-\log P(x)$$. Then, to optimal performance we expect that the codeword length used to represent $$x$$ to be roughly $$1/\log(P(x))$$. In other words, in an efficient encoding algorithm, the number of bits in $$s$$ ($$\log(s)$$) to be $$1/\log(P(x))$$ more than that of $$s'$$:

$$
\begin{align*}
\log(s) - \log(s') \approx \frac{1}{\log P(x)}\text{ for }d(s) = x, s'.
\end{align*}
$$

Expressed in another way

$$
\begin{align*}
\frac{s'}{s} = P(x)\text{ for }d(s) = x, s'.
\end{align*}
$$

The question is: how can we design a bijective mapping from $$\mathbb{N}$$ and $$\mathbb{N}\times\mathcal{A}$$ satisfying the above goal? Here's the core idea of asymmetric numerical system: let us assume that we have access to a function that maps each of the number in $$\mathbb{N}$$ into one of the symbols in $$\mathcal{A}$$, denoted as $$h:\mathbb{N}\to\mathcal{A}$$ with the property that for any integer $$s$$ and symbol $$x$$, there are roughly $$P(x)\times s$$ numbers below $$s$$ with symbol $$x$$. Put more precisely, 

$$
\begin{align*}
\frac{|\{n\in\mathbb{N}, n<s, h(n) = x\}|}{s} \approx P(x) \text{ for any }s\in\mathbb{N}\text{ and }x\in\mathcal{A}.
\end{align*}
$$

With such a mapping $$h$$ available, we can define the bijective decoder mapping $$d$$ to be

$$
\begin{align*}
d(s)=&s',x\text{ where} \\
s'=&\left|\left\{n\in\mathbb{N},n<s,h(n)=h(s)\right\}\right|,\\
x=&h(s).
\end{align*}
$$

and it is easy to check that our two design goals are satisfied, and now we have shifted our task to finding such a labeling function $$h$$ such that it leads to easy computation of $$d$$ and $$e$$.

## Mapping of natural numbers to symbols ($$h$$)

In r-ANS (range-ANS) design, the pmf $$P:\mathcal{A}\to[0, 1]$$ is quantized into integers $$p(x)$$ where 

$$
\begin{align*}
&\sum_{x\in\mathcal{A}}p(x)=2^r\\
&p(x)/2^r\approx P(x). 
\end{align*}
$$

The mapping $$h$$ is design as the following: we divide the natural numbers into chunks with length $$2^r$$. Within each chunk, start with symbol $$0\in\mathcal{A}$$, we map the first $$p(0)$$ numbers to $$0$$; then the subsequent $$p(1)$$ numbers are mapped to $$1$$, so on and so forth. 

Let us define $$c:\mathcal{A} \to \mathcal{N}$$ with $$c(x)=\sum_{a\in\mathcal{A}, a<x}p(a)$$. Then the number from $$c(x)$$ to $$c(x)+p(x)$$ within a length $$2^r$$ chunk is labeled as $$x$$.

With this mapping, can express $$d$$ and $$e$$ into the following arithmetics that are easy to compute:

$$
\begin{align*}
d(s) =& p(x)\times(s//2^r) + (s\text{ mod }2^r-c(x)), x\triangleq h(s) \\
e(s', x) =& 2^r \times (s'//p(x)) + (s'\text{ mod }p(x) + c(x))
\end{align*}
$$

with this comes the first implementation of ANS

## ANS without rescaling (flawed version)

### Encoder

```python
def ans_encoder(symbols, p, c, r):
    s = 0
    for x in symbols:
        s = 2 ** r * (s // p[x]) + \
            s % p[x] + c[x]
    return s
```

### Decoder
```python
def ans_decoder(s, p, c, r):
    def h(s):
        s = s % 2 ** r
        # this loop can be improved by binary search
        for a in reversed(range(len(c))):
            if s >= c[a]:
                return a
    decoded_symbols = []
    while s:
        x = h(s)
        decoded_symbols.append(x)
        s = p[x] * (s // 2 ** r) + \
            s % (2 ** r) - c[x]
    return list(reversed(decoded_symbols)
```

Running this encoder decoder pair through tests, one will realize that there is a small issue with the encoder and decoder function $$e$$ and $$d$$. Specifically, if both $$s'$$ and $$x$$ are $$0$$, then $$s=e(0, 0)=0$$. This means that any front-loaded $$0$$-sequence will be just be coded into $$0$$, and decoder has no ways of knowing how many $$0$$ symbols are there in the front of the sequence, if any! To solve this issue, we need to additionally guarantee that $$e$$ results in strictly increasing integer. A fixed version is provided in the next section.

## ANS without rescaling (correct version)

### Encoder
```python
def ans_encoder(symbols, p, c, r):
    """ ANS encoder (no rescaling)

    Parameters
    ----------
    symbols : list of int
        list of input symbols represented by index
        value should not be larger than len(p)
    p : list of int
        quantized pmf of all input alphabet, sum(p) == 2 ** r
    c : list of int
        quantized cdf of all input alphabet, len(c) = len(p)
        c[0] = 0, and c[-1] = sum(p[:-1])
    r : int
        bit-width precision of the quantized pmf
        sum(p) == 2 ** r

    Warnings
    --------
    int type for all the input arguments should be python int type,
    which has arbitrary precision

    Returns
    -------
    s : integer representation of the encoded message

    """
    s = 0
    for x in symbols:
        if s < c[1]:
            s += 1
        s = 2 ** r * (s // p[x]) + \
            s % p[x] + c[x]
    return s
```

### Decoder
```python
def ans_decoder(s, p, c, r):
    """ ANS encoder (no rescaling)

    Parameters
    ----------
    s : int
        integer representation of the encoded message
    p : list of int
        quantized pmf of all input alphabet, sum(p) == 2 ** r
    c : list of int
        quantized cdf of all input alphabet, len(c) = len(p)
        c[0] = 0, and c[-1] = sum(p[:-1])
    r : int
        bit-width precision of the quantized pmf
        sum(p) == 2 ** r

    Warnings
    --------
    int type for all the input arguments should be python int type,
    which has arbitrary precision

    Returns
    -------
    decoded_symbols : list of int
        list of decoded symbols

    """

    def h(s):
        s = s % 2 ** r
        # this loop can be improved by binary search
        for a in reversed(range(len(c))):
            if s >= c[a]:
                return a

    decoded_symbols = []
    while s:
        x = h(s)
        decoded_symbols.append(x)
        s = p[x] * (s // 2 ** r) + \
            s % (2 ** r) - c[x]
        if s < c[1]:
            s -= 1
    return list(reversed(decoded_symbols))
```
### Test

```python
import random
import math

# initialize data distribution and input length
p = [20, 50, 80, 106]
c = [0, 20, 70, 150]
r = 8
sequence_length = 100

# randomly sample input
random.seed(1)
symbols = random.choices(range(len(p)), weights=p, k=sequence_length)

# encode
s = ans_encoder(symbols, p, c, r)

# decode
decoded_symbols = ans_decoder(s, p, c, r)

# statistics
average_bps = math.log2(s) / sequence_length
entropy = sum(-i / 2 ** r * math.log2(i / 2 ** r) for i in p)

# sanity check
assert all(x == y for x, y in zip(decoded_symbols, symbols))

# display results
print(f"encoded integer        : {s}")
print(f"average bits per symbol: {average_bps:.5f} bits/symbol")
print(f"data source entropy    : {entropy:.5f} bits/symbol")
```

Test output

```python
encoded integer        : 125621967822099623663819958660494947377946858741001513589
average bits per symbol: 1.86357 bits/symbol
data source entropy    : 1.79865 bits/symbol
```

## ANS with rescaling

The above ANS implementation takes advantage of the fact that python integer has arbitrary precision, which allows us to encode a sequence that is arbitrarily long without overflowing. This poses a complexity issue: the encoding operation gets increasingly hard to compute as integer $$s$$ gets larger. Without resolving this reliance on infinite precision integer arithmetic, we cannot implement it using lower-level language with more hardware friendly instructions. 

One idea is to limit the range of $$s$$, say with a maximum bit-width of $$r_s$$. Since the encoding process will necessary increase the value of $$s$$, we then need to scale down its value before additional encoding, to a point where we can avoid overflow. In other words, before carrying out the encoding operation os $$e(s', x)$$, $$s'$$ should satisfy 

$$
\begin{align*}
&&2^r\times (s'//p(x)) + (s'\text{ mod }p(x) + c(x)) &< 2^{r_s}\\
\Longleftrightarrow&& s'//p(x) + \underbrace{(s'\text{ mod }p(x) + c(x)) / 2^r}_{<1}&< 2^{r_s-r}\\
\Longleftrightarrow&& s'//p(x) &< 2^{r_s-r}\\
\Longleftrightarrow&& s' &< (2^{r_s-r}+1)\times p(x)
\end{align*}
$$

In the rescaling implementation, scaling down is achieved by extracting $$r_t$$ least significant bits, packing these bits into an integer $$t$$, and saving this integer to a stack $$t_\text{stack}$$, achieved through the following logic
```python
while s >= (2 ** (r_s - r) + 1) * p[x]:
    t = (s % 2 ** r_t)
    s >>= r_t
    t_stack.append(t)
```

Now we have guaranteed that encoder output will be an integer $$s<2^{r_s}$$ accompanied by a stack of integers $$t<2^{r_t}$$, the next question is, how to perform decoding? An easy answer would be to do the exact inverse of encoding, but to do that we need to know exactly when to perform up-scaling. The key is to realize that after the above loop is performed, it is guaranteed that $$e(s', x)$$ is always larger than or equal to $$2^{r_s-r_t}$$ (assuming $$r_t > r$$), and thus during decoding we just need to upscale $$s$$ whenever it falls below $$2^{r_s-r_t}$$. 

After the while loop terminates, we have

$$
\begin{align*}
&& 2^{r_t} s' + (2^{r_t}-1) &\geq (2^{r_s-r}+1) p(x)\\
\Longleftrightarrow && 2^{r_t}s' + 2^{r_t} &> (2^{r_s-r}+1)p(x)\\
\Longleftrightarrow && 2^{r_t}s'&> 2^{r_s-r}p(x) + p(x) - 2^{r_t}\\
\Longleftrightarrow &&        s'&> 2^{r_s-r-r_t}p(x) + \underbrace{p(x)/2^{r_t}}_{<1} - 1\\
\Longleftrightarrow &&        s'&\geq 2^{r_s-r-r_t}p(x)
\end{align*}
$$

Plugging in the above inequality to $$e(s', x)$$, we have 

$$
\begin{align*}
e(s', x) \geq 2^{r_s -r_t}.
\end{align*}
$$

Now we are ready to implement ANS with rescaling.

```python
def ans_encoder(symbols, p, c, r, r_s, r_t):
    """ ANS encoder

    Parameters
    ----------
    symbols : list of int
        list of input symbols represented by index
        value should not be larger than len(p)
    p : list of int
        quantized pmf of all input alphabet, sum(p) == 2 ** r
    c : list of int
        quantized cdf of all input alphabet, len(c) = len(p)
        c[0] = 0, and c[-1] = sum(p[:-1])
    r : int
        bit-width precision of the quantized pmf
        sum(p) == 2 ** r
    r_s : int
        bit-width precision of encoded integer s
    r_t : int
        bit-width precision of integers in stack t_stack

    Returns
    -------
    s : int
        s < 2 ** r_s
    t_stack : list of int
        each int < 2 ** r_t

    """
    s = 0
    t_stack = []
    for x in symbols:
        if s < c[1]:
            s += 1
        while s >= (2 ** (r_s - r) + 1) * p[x]:
            t = s % (2 ** r_t)
            s >>= r_t
            t_stack.append(t)
        s = 2 ** r * (s // p[x]) + \
            s % p[x] + c[x]
    return s, t_stack


def ans_decoder(s, t_stack, p, c, r, r_s, r_t):
    """ ANS encoder

    Parameters
    ----------
    s : int
        (s, t_stack) together represent the encoded message; s < 2 ** r_s
    t_stack : list of int
        (s, t_stack) together represent the encoded message; t < 2 ** r_t
    p : list of int
        quantized pmf of all input alphabet, sum(p) == 2 ** r
    c : list of int
        quantized cdf of all input alphabet, len(c) = len(p)
        c[0] = 0, and c[-1] = sum(p[:-1])
    r : int
        bit-width precision of the quantized pmf
        sum(p) == 2 ** r
    r_s : int
        bit-width precision of encoded integer s
    r_t : int
        bit-width precision of integers in stack t_stack

    Returns
    -------
    decoded_symbols : list of int
        list of decoded symbols

    """

    def h(s):
        s = s % 2 ** r
        # this loop can be improved by binary search
        for a in reversed(range(len(c))):
            if s >= c[a]:
                return a

    decoded_symbols = []
    while s:
        x = h(s)
        decoded_symbols.append(x)
        s = p[x] * (s // 2 ** r) + \
            s % (2 ** r) - c[x]
        if s < c[1]:
            s -= 1
        while s < 2 ** (r_s - r_t) and len(t_stack):
            t = t_stack.pop()
            s = (s << r_t) + t
    return list(reversed(decoded_symbols))


## Test code
import random
import math

# initialize data distribution and input length
p = [20, 50, 80, 106]
c = [0, 20, 70, 150]
r = 8
r_s = 32
r_t = 16
sequence_length = 10000

# randomly sample input
random.seed(1)
symbols = random.choices(range(len(p)), weights=p, k=sequence_length)

# encode
s, t_stack = ans_encoder(symbols, p, c, r, r_s, r_t)

# decode
decoded_symbols = ans_decoder(s, t_stack[:], p, c, r, r_s, r_t)

# statistics
average_bps = (r_s + len(t_stack) * r_t) / sequence_length
entropy = sum(-i / 2 ** r * math.log2(i / 2 ** r) for i in p)

# sanity check
assert all(x == y for x, y in zip(decoded_symbols, symbols))

# display results
print(f"average bits per symbol: {average_bps:.5f} bits/symbol")
print(f"data source entropy    : {entropy:.5f} bits/symbol")
```

### Test results

```python
average bits per symbol: 1.80960 bits/symbol
data source entropy    : 1.79865 bits/symbol
```
