---
layout: post
title: Arithmetic Coding (AC) Implementation
date: 2020-06-24
comments: true
description: python implementation of arithmetic coding
---

## Arithmetic Encoder (infinite precision)

```python
a = 0.
b = 1.
s = 0
for symbol in symbols:
    width = b - a
    a, b = a + width * c[symbol], a + width * d[symbol]
while b <= 1 / 2 or a >= 1 / 2:
    if b <= 1 / 2:  # case 0
        bits.append(0)
        a *= 2
        b *= 2
    else:  # case 1
        bits.append(1)
        a = 2 * (a - 1 / 2)
        b = 2 * (b - 1 / 2)
# a < 1/2 and b > 1/2
while a > 1 / 4 and b < 3 / 4:
    s += 1
    a = 2 * (a - 1 / 4)
    b = 2 * (b - 1 / 4)
s += 1
# a <= 1/4 or b >= 3/4
if a <= 1 / 4:  # case 2a
    bits.append(0)
    bits += [1] * s
else:  # case 2b
    bits.append(1)
    bits += [0] * s
```

## Arithmetic Decoder (infinite precision)

```python
decoded_symbols = []
z = 0.0
a = 0.0
b = 1.0
for bit_index, bit in enumerate(bits):
    binary_block_size = 2 ** (-bit_index - 1)
    if bit == 1:
        z += binary_block_size
    symbol = decode_one_symbol(z, z + binary_block_size, a, b, c, d)
    while symbol is not None:
        decoded_symbols.append(symbol)
        a, b = a + (b - a) * c[symbol], a + (b - a) * d[symbol]
        symbol = decode_one_symbol(z, z + binary_block_size, a, b, c, d)

def decode_one_symbol(z_0, z_1, a, b, c, d):
    """
    Parameters
    ----------
    z_0: lower end of the current binary block
    z_1: higher end of the current binary block
    a: lower end of the current sub-interval
    b: higher end of the current sub-interval
    c: CDF starts with a 0.0
    d: CDF that ends with 1.0

    Returns
    -------
    if [z_0, z_1] is not contained in any of the symbols inside [a, b]:
        return None
    else:
        return the decoded index

    """
    for index, (low, high) in enumerate(zip(c, d)):
        low = a + (b - a) * low
        high = a + (b - a) * high
        if low <= z_0 and z_1 <= high:
            return index

```

## Arithmetic Encoder with Rescaling (infinite precision)

```python
a = 0.
b = 1.
s = 0
for symbol in symbols:
    width = b - a
    a, b = a + width * c[symbol], a + width * d[symbol]
    while b <= 1 / 2 or a >= 1 / 2:
        if b <= 1 / 2:  # case 0
            bits.append(0)
            bits += [1] * s
            s = 0
            a *= 2
            b *= 2
        else:  # case 1
            bits.append(1)
            bits += [0] * s
            s = 0
            a = 2 * (a - 1 / 2)
            b = 2 * (b - 1 / 2)
    # a < 1/2 and b > 1/2
    while a > 1 / 4 and b < 3 / 4:
        s += 1
        a = 2 * (a - 1 / 4)
        b = 2 * (b - 1 / 4)
s += 1
# a <= 1/4 or b >= 3/4
if a <= 1 / 4:  # case 2a
    bits.append(0)
    bits += [1] * s
else:  # case 2b
    bits.append(1)
    bits += [0] * s
```

## Arithmetic Encoder with Rescaling (finite precision)

```python
a = 0.
b = 2 ** range_precision
s = 0
for symbol in symbols:
    width = b - a
    a, b = a + width * c[symbol] // 2 ** pmf_precision, a + width * d[symbol] // 2 ** pmf_precision
    while b <= range_half or a >= range_half:
        if b <= range_half:  # case 0
            bits.append(0)
            bits += [1] * s
            s = 0
            a *= 2
            b *= 2
        else:  # case 1
            bits.append(1)
            bits += [0] * s
            s = 0
            a = 2 * (a - range_half)
            b = 2 * (b - range_half)
    # a < 1/2 and b > 1/2
    while a > range_quarter and b < 3 * range_quarter:
        s += 1
        a = 2 * (a - range_quarter)
        b = 2 * (b - range_quarter)
s += 1
# a <= 1/4 or b >= 3/4
if a < range_quarter:  # case 2a
    bits.append(0)
    bits += [1] * s
else:  # case 2b
    bits.append(1)
    bits += [0] * s
```

## Arithmetic Decoder with Rescaling (finite precision)

```python
z = 0
for i in range(range_precision):
    z = (z << 1)
    if i < len(bits):
        z += bits[i]
next_bit_index = min(len(bits), range_precision)
z_gap = 1 << max(0, range_precision - len(bits))

decoded_symbols = []
a = 0
b = 2 ** range_precision
while True:
    for index, (low, high) in enumerate(zip(c, d)):
        low = a + (b - a) * low // 2 ** pmf_precision
        high = a + (b - a) * high // 2 ** pmf_precision
        if low <= z and high >= z + z_gap:
            a = low
            b = high
            decoded_symbols.append(index)
            break
    else:
        break
    while b <= range_half or a >= range_half:
        if b <= range_half:
            b = 2 * b
            a = 2 * a
            z = 2 * z
            if next_bit_index < len(bits):
                z += bits[next_bit_index]
                next_bit_index += 1
            else:
                z_gap <<= 1
        else:
            b = 2 * (b - range_half)
            a = 2 * (a - range_half)
            z = 2 * (z - range_half)
            if next_bit_index < len(bits):
                z += bits[next_bit_index]
                next_bit_index += 1
            else:
                z_gap <<= 1
    while a > range_quarter and b < 3 * range_quarter:
        a = 2 * (a - range_quarter)
        b = 2 * (b - range_quarter)
        z = 2 * (z - range_quarter)
        if next_bit_index < len(bits):
            z += bits[next_bit_index]
            next_bit_index += 1
        else:
            z_gap <<= 1
```

