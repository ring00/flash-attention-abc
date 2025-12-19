# Flash Attention Speedrun

This document provides a mathematical walkthrough of **Flash Attention**, an algorithm that significantly speeds up attention mechanisms in Transformers by minimizing memory access. We start from the basics of **Online Softmax** and build up to the full Flash Attention algorithm.

## Symbols and Notation

Here are the mathematical symbols used throughout this guide:

**Inputs & Outputs:**

- $Q \in \mathbb{R}^{n \times d}$: **Query** matrix (input)
- $K \in \mathbb{R}^{n \times d}$: **Key** matrix (input)
- $V \in \mathbb{R}^{n \times d}$: **Value** matrix (input)
- $O \in \mathbb{R}^{n \times d}$: **Output** matrix (result of attention)

**Intermediate Variables:**

- $S \in \mathbb{R}^{n \times n}$: **Attention Scores** (pre-softmax, $Q K^T$)
- $P \in \mathbb{R}^{n \times n}$: **Attention Weights** (post-softmax probabilities)
- $m$: **Running Maximum** (used for numerical stability)
- $l$: **Running Sum** (normalization factor for softmax)

**Dimensions:**

- $n$: Sequence length (number of tokens)
- $d$: Head dimension (feature size per head)

**Gradients (for Backward Pass):**

- $L$: Scalar loss function value
- $dO = \frac{\partial L}{\partial O}$: Gradient of the loss with respect to the output

## Online Softmax

Standard Softmax computes the exponential of the entire input vector at once. **Online Softmax** computes it incrementally, which is crucial for processing large sequences without keeping everything in memory.

Given an input vector $`x = \{x_i\}_{i = 1}^N`$, the standard softmax operation generates a new vector $y$:

```math
\begin{aligned}
y &= \{y_i\}_{i = 1}^N \\
y_i &= \frac{e^{x_i}}{\sum_{j = 1}^N e^{x_j}}
\end{aligned}
```

### Numerical Stability (Safe Softmax)

Directly computing $e^{x_i}$ can lead to **numerical overflow** if $x_i$ is large (e.g., $e^{100}$ is huge). To prevent this, we subtract the maximum value $m = \max(x)$ from each element. This shifts the largest exponent to 0 ($e^0=1$), keeping values manageable without changing the result:

$$
\begin{aligned}
m &= \max(x) \\
y_i &= \frac{e^{x_i - m}}{\sum_{j = 1}^N e^{x_j - m}}
\end{aligned}
$$

### Naive Implementation (3-Pass)

A straightforward implementation requires **three passes** over the data:

1. Find the max $m$.
2. Compute the sum of exponentials (denominator) $l$.
3. Compute the final values $y_i$.

Let's trace this naive approach:

---

**1. Initialization**

$$
\begin{aligned}
m_0 &= - \infty \\
l_0 &= 0
\end{aligned}
$$

---

**2. Pass 1: Find Maximum**

```
for i = 1 to N
```

$$
m_i = \max(m_{i - 1}, x_i)
$$

```
end
```

---

**3. Pass 2: Compute Denominator**

```
for i = 1 to N
```

$$
l_i = l_{i - 1} + e^{x_i - m_N}
$$

```
end
```

---

**4. Pass 3: Compute Output**

```
for i = 1 to N
```

$$
y_i = \frac{e^{x_i - m_N}}{l_N}
$$

```
end
```

---

This approach is inefficient because it reads the input $x$ multiple times.

### The Online Algorithm (2-Pass)

The 3-pass algorithm generates two intermediate sequences, one for the running maximum of the input vector

```math
\begin{aligned}
m &= \{m_i\}_{i = 1}^N \\
m_i &= \max( \{x_j\}_{j = 1}^i )
\end{aligned}
```

The other for the accumulated denominator in the softmax equation

```math
\begin{aligned}
l &= \{l_i\}_{i = 1}^N \\
l_i &= \sum_{j = 1}^ie^{x_j - m_N}
\end{aligned}
```

The challenge is that the denominator $l$ depends on the _global_ max $m_N$, which we don't know until the end.
However, we can maintain a "running" denominator $l'$ that uses the _current_ max $m_i$.

```math
\begin{aligned}
l' &= \{l_i'\}_{i = 1}^N \\
l_i' &= \sum_{j = 1}^i e^{x_j - m_j}
\end{aligned}
```

Notice that $l_N' = l_N$, which is exactly the softmax denominator. Therefore, we could use the following recurrence relation:

$$
\begin{aligned}
m_i &= \max({m_{i - 1}, x_i}) \\
l_i' &= \sum_{j = 1}^i e^{x_j - m_i} \\
&= \sum_{j = 1}^{i - 1} e^{x_j - m_i} + e^{x_i - m_i} \\
&= \sum_{j = 1}^{i - 1} e^{x_j - m_{i-1} + m_{i-1} - m_i} + e^{x_i - m_i} \\
&= e^{m_{i-1} - m_i} \underbrace{\sum_{j = 1}^{i - 1} e^{x_j - m_{i-1}}}_{l'_{i-1}} + e^{x_i - m_i} \\
&= l_{i - 1}' \cdot e^{m_{i - 1} - m_i} + e^{x_i - m_i}
\end{aligned}
$$

Based on the formula, $l_i'$ depends only on $l_{i - 1}'$, $m_{i - 1}$, $m_i$ and $x_i$. This allows us to compute both the max and the sum in one go:

---

1. Initialization

$$
\begin{aligned}
m_0 &= -\infty \\
l_0 &= 0
\end{aligned}
$$

---

2. Pass-1: Online computation

```
for i = 1 to N
```

$$
\begin{aligned}
m_i &= max(m_{i - 1}, x_i) \\
l_i &= l_{i - 1} e^{m_{i - 1} - m_i} + e^{x_i - m_i}
\end{aligned}
$$

```
end
```

---

3. Pass-2: Computing softmax results

```
for i = 1 to N
```

$$
y_i = \frac{e^{x_i - m_N}}{l_N}
$$

```
end
```

---

## Flash Attention

### Forward

Given $Q, K, V \in \mathbb{R}^{n \times d}$, where $n \in \mathbb{N^+}$ is the sequence length and $d \in \mathbb{N^+}$ is the head dimension, the attention operation computes the following

$$
\begin{aligned}
S &= QK^T \\
P &= \text{softmax}(S) \\
O &= PV
\end{aligned}
$$

### Standard Attention (row-wise)

Let's look at the attention computation for a single query row $q = Q[k, :]$.
The standard algorithm would be:

1.  Compute scores $x = q K^T$.
2.  Compute softmax statistics $m, l$.
3.  Compute output $o = \text{softmax}(x) V$.

Using the "naive" online softmax logic derived above, this looks like:

---

**Pass 1: Statistics**

```
for i = 1 to N
```

$$
\begin{aligned}
x_i &= qK[i, :]^T \\
m_i &= \max(m_{i - 1}, x_i) \\
l_i &= l_{i - 1} e^{m_{i - 1} - m_i} + e^{x_i - m_i}
\end{aligned}
$$

```
end
```

---

**Pass 2: Output Accumulation**

```
for i = 1 to N
```

$$
\begin{aligned}
y_i &= \frac{e^{x_i - m_N}}{l_N} \\
o_i &= o_{i - 1} + y_i V[i, :]
\end{aligned}
$$

```
end
```

---

$$ O[k, :] = o_N $$

### Fusing the Loops

Let's take a closer look at the second loop, clearly we have

$$
\begin{aligned}
o_i &= \sum_{j = 1}^i y_j V[j, :] \\
    &= \sum_{j = 1}^i \frac{e^{x_j - m_N}}{l_N} V[j, :]
\end{aligned}
$$

We want to compute $o_i$ in the _same_ loop as $m_i$ and $l_i$.
However, $y_i$ depends on the final $m_N$ and $l_N$.

Let's recall the online softmax trick and define the running output $o'_i$ using the _current_ max $m_i$ and sum $l_i$:

$$
o_i' = \sum_{j = 1}^i \frac{e^{x_j - m_i}}{l_i} V[j, :]
$$

Now, let's derive the update rule for $`o'_i`$.
We want to express $`o'_i`$ using the previous value $`o'_{i-1}`$.

$$
\begin{aligned}
o_i' &= \frac{1}{l_i} \left( \sum_{j = 1}^{i - 1} e^{x_j - m_i} V[j, :] + e^{x_i - m_i} V[i, :] \right) \\
&= \frac{1}{l_i} \left( \sum_{j = 1}^{i - 1} e^{x_j - m_{i-1} + m_{i-1} - m_i} V[j, :] + e^{x_i - m_i} V[i, :] \right) \\
&= \frac{1}{l_i} \left( e^{m_{i-1} - m_i} \underbrace{\sum_{j = 1}^{i - 1} e^{x_j - m_{i-1}} V[j, :]}_{o'_{i-1} \cdot l_{i-1}} + e^{x_i - m_i} V[i, :] \right) \\
&= \frac{1}{l_i} \left(l_{i-1} \cdot e^{m_{i-1} - m_i} \cdot o'_{i-1} + e^{x_i - m_i} V[i, :] \right)
\end{aligned}
$$

This gives us the update rule:

1.  **Rescale** the old output $`o'_{i-1}`$ by $`(l_{i-1} / l_i) \cdot e^{m_{i-1} - m_i}`$.
2.  **Add** the new term, scaled by $e^{x_i - m_i} / l_i$.

We can see that $o_i'$ only depends on $o_{i - 1}'$, $l_{i - 1}$, $l_i$, $m_{i - 1}$, $m_i$ and $x_i$. This allows us to fuse everything into one single loop:

```
for i = 1 to N
```

$$
\begin{aligned}
x_i &= qK[i, :]^T \\
m_i &= \max(m_{i - 1}, x_i) \\
l_i &= l_{i - 1} e^{m_{i - 1} - m_i} + e^{x_i - m_i} \\
o_i' &= \frac{1}{l_i}(l_{i - 1} e^{m_{i - 1} - m_i} o_{i - 1}' + e^{x_i - m_i} V[i, :])
\end{aligned}
$$

```
end
```

### Backward

Recall that attention operation calculates

$$
\begin{aligned}
S &= QK^T \\
P &= \text{softmax}(S) \\
O &= PV
\end{aligned}
$$

Given a scalar loss $L$, assume we already have $dO = \frac{\partial L}{\partial O}$, we would like to compute the following gradients

$$
\begin{aligned}
dP &= \frac{\partial L}{\partial P} \\
dV &= \frac{\partial L}{\partial V} \\
dS &= \frac{\partial L}{\partial S} \\
dQ &= \frac{\partial L}{\partial Q} \\
dK &= \frac{\partial L}{\partial K}
\end{aligned}
$$

Before we dive into the calculation, let's first examine the derivative of Softmax operation.

#### Backward of Softmax

Let's recall that softmax gives

```math
\begin{aligned}
y &= \{y_i\}_{i = 1}^N  \\
y_i &= \frac{e^{x_i}}{\sum_{k = 1}^N e^{x_k}}
\end{aligned}
```

The partial derivative $\frac{\partial y_i}{\partial x_j}$ is hard to compute directly, however, we can take logarithm to simplify softmax formula first

$$
\begin{aligned}
log(y_i) &= x_i - log(\sum_{k = 1}^N e^{x_k}) \\
\frac{1}{y_i}\frac{\partial y_i}{\partial x_j} &= \frac{\partial x_i}{\partial x_j} - \frac{\frac{\partial \sum_{k = 1}^N e^{x_k}}{\partial x_j}}{\sum_{k = 1}^N e^{x_k}} \\
                                               &= \mathbb{I}[i = j] - y_j \\
\frac{\partial y_i}{\partial x_j} &= y_i (\mathbb{I}[i = j] - y_j)
\end{aligned}
$$

Consider an input $[x_1, x_2, x_3]$ and its output $[y_1, y_2, y_3]$, the derivative matrix (aka jacobian matrix $J$) is

$$
J = \begin{bmatrix}
y_1 (1 - y_1) & -y_1 y_2 & -y_1 y_3 \\
-y_2 y_1 & y_2 (1 - y_2) & -y_2 y_3 \\
-y_3 y_1 & -y_3 y_2 & y_3 (1 - y_3)
\end{bmatrix}
$$

By vectorizing the formula we get

$$
J = \text{diag}(y) - y^Ty \in \mathbb{R}^{d \times d}
$$

For backward computation, assume that we have a scalar loss $L$, applying the chain-rule we get

$$
\begin{aligned}
\frac{\partial L}{\partial x_i} &= \sum_{j = 1}^N \frac{\partial L}{\partial y_j} \frac{\partial y_j}{\partial x_i} \\
                                &= \sum_{j = 1}^N \frac{\partial L}{\partial y_j} y_j (\mathbb{I}[i = j] - y_i) \\
                                &= dy \cdot J[:, i] \\
dx &= dy \cdot J \\
   &= dy (\text{diag}(y) - y^Ty)
\end{aligned}
$$

#### Backward of Score Matrix

Let's focus on a vector at the $i$-th row in $dS$

$$
\begin{aligned}
p &= P[i, :] \in \mathbb{R}^n \\
s &= S[i, :] \in \mathbb{R}^n \\
p &= \text{softmax}(s)
\end{aligned}
$$

$$
\begin{aligned}
ds &= dp (\text{diag}(p) - p^Tp) \\
   &= dp \odot p - (dp \cdot p^T) p
\end{aligned}
$$

We know that $dP = dO \cdot V^T$, which gives us

$$
\begin{aligned}
dp &= dP[i, :] \\
   &= \sum_{j = 1}^N dO[i, :] V[:, j] \\
dp \cdot p^T &= \sum_{j = 1}^N dO[i, :] V[:, j] P[i, :]^T \\
             &= dO[i, :] \sum_{j = 1}^N V[:, j] P[i, :]^T \\
             &= dO[i, :] O[i, :]^T \\
             &= do \cdot o^T \in \mathbb{R}
\end{aligned}
$$

Let $d = do \cdot o^T = dO[i, :] O[i, :]^T$, we have

```math
\begin{aligned}
D &= \{d_i\}_{i = 1}^N \\
  &= \text{rowsum}(dO \odot O)
\end{aligned}
```

Plugging in the previous formula, we have

$$
\begin{aligned}
ds &= dp \odot p - \text{broadcast}(d) \odot p \\
   &= p \odot (dp - \text{broadcast}(d))
\end{aligned}
$$

which gives us a single row in the final $dS$ matrix, vectorizing it yields

$$
dS = P \odot (dP - \text{broadcast}(D))
$$

Finally, we can compute the gradients for $Q, K, V$:

$$
\begin{aligned}
dV &= P^T dO \\
dQ &= dS K \\
dK &= dS^T Q
\end{aligned}
$$

## Flash Attention 1

The primary trick is to rewrite the forward iteration process in [Fusing the Loops](#fusing-the-loops) in its block form.

Let's first review the input of the attention operation

- $Q \in \mathbb{R}^{n \times d}$: Query matrix
- $K \in \mathbb{R}^{n \times d}$: Key matrix
- $V \in \mathbb{R}^{n \times d}$: Value matrix

For outputs, we pre-allocate some extra tensors on HBM

- $O = (0)_{n \times d} \in \mathbb{R}^{n \times d}$: Output matrix
- $m = (-\infty)_n \in \mathbb{R}^n$: Running maximum
- $l = (0)_n \in \mathbb{R}^n$: Running softmax denominator

Let's assume that $Q$ is divided into blocks of shape $\mathbb{R}^{B_r \times d}$, and $K, V$ are divided into blocks of shape $\mathbb{R}^{B_c \times d}$.

There will be $T_r = \lceil \frac{n}{B_r} \rceil$ $Q$ blocks and $T_c = \lceil \frac{n}{B_c} \rceil$ $K, V$ blocks respectively.

The forward computation process can be implemented as

---

```
for j = 1 to T_c
```

Outer loop:

Load $K_j, V_j \in \mathbb{R}^{B_c \times d}$ from HBM into SRAM

```
    for i = 1 to T_r
```

Inner loop:

Load $Q_i, O_i \in \mathbb{R}^{B_r \times d}$ and $m_i, l_i \in \mathbb{R}^{B_r}$ from HBM into SRAM

1. Computation

$$
\begin{aligned}
S_{ij} &= Q_i K_j^T \in \mathbb{R}^{B_r \times B_c} \\
\hat{m}_{ij} &= \text{rowmax}(S_{ij}) \in \mathbb{R}^{B_r} \\
\hat{P}_{ij} &= e^{S_{ij} - \hat{m}_{ij}} \in \mathbb{R}^{B_r \times B_c} \\
\hat{l}_{ij} &= \text{rowsum}(\hat{P}_{ij}) \in \mathbb{R}^{B_r}
\end{aligned}
$$

2. Online update

$$
\begin{aligned}
m_i' &= \max(m_i, \hat{m}_{ij}) \in \mathbb{R}^{B_r} \\
l_i' &= e^{m_i - m_i'} l_i + e^{\hat{m}_{ij} - m_i'} \hat{l}_{ij} \in \mathbb{R}^{B_r} \\
O_i' &= \text{diag}(l_i')^{-1} (\text{diag}(l_i) e^{m_i - m_i'} O_i + e^{\hat{m}_{ij} - m_i'} \hat{P}_{ij} V_j) \in \mathbb{R}^{B_r \times d}
\end{aligned}
$$

3. Write back HBM

$$
\begin{aligned}
O_i &= O_i' \\
l_i &= l_i' \\
m_i &= m_i'
\end{aligned}
$$

```
    end
```

```
end
```

---

Now, let's take a look at the memory access and flops pattern assuming that we're using bfloat16.

On HBM, we have $Q, K, V, O$ consume $8nd$ bytes and $m, l$ consume $4n$ bytes. Therefore the total HBM consumption is $8nd + 4n$.

In the outer loop, $K_j, V_j \in \mathbb{R}^{B_c \times d}$ are read into SRAM, there are $4 B_c d$ bytes HBM read. Since the outer loop repeats $T_c$ times, the total read volume is $4 B_c d T_c = 4nd$.

In the inner loop, $Q_i, O_i \in \mathbb{R}^{B_r \times d}$ and $m_i, l_i \in \mathbb{R}^{B_r}$ are read from HBM, creating $4 B_r d + 4 B_r$ bytes read. Since the inner loop repeats $T_c T_r$ times, the total read volume is $(4 B_r d + 4 B_r)T_r T_c = n(4d + 4)T_c$.

As for the write back, $O_i \in \mathbb{R}^{B_r \times d}$ and $m_i, l_i \in \mathbb{R}^{B_r}$ are written back to HBM, creating $2 B_r d + 4 B_r$ bytes write. Since the inner loop repeats $T_c T_r$ times, the total write volume is $(2 B_r d + 4 B_r)T_r T_c = n(2d + 4)T_c$.

Therefore, the total HBM access volume is roughly $2nd (2 + 3T_c) = 2nd (2 + 3\lceil\frac{n}{B_c}\rceil)$ bytes.

Apparently, by having a larger $B_c$, we significantly reduce the HBM access volume. $K_j, V_j \in \mathbb{R}^{B_c \times d}$ and $Q_i, O_i \in \mathbb{R}^{B_r \times d}$ takes $4 (B_c + B_r) d$ bytes on SRAM, for a SRAM of size $M$, we have

$$
\begin{aligned}
4 (B_c + B_r) d &\leq M \\
B_c + B_r &\leq \frac{M}{4d}
\end{aligned}
$$

Thus we set $B_c = \lceil\frac{M}{8d}\rceil, B_r = \min(\lceil\frac{M}{8d}\rceil, d)$. The former helps reduce the HBM access while the latter ensures GEMM efficiency in the inner loop.

The primary flops are spent on GEMM operations

$$
\begin{aligned}
S_{ij} &= Q_i K_j^T \in \mathbb{R}^{B_r \times B_c} \\
O_i' &= \text{diag}(l_i')^{-1} (\text{diag}(l_i) e^{m_i - m_i'} O_i + e^{\hat{m}_{ij} - m_i'} \hat{P}_{ij} V_j) \in \mathbb{R}^{B_r \times d} \\
\hat{P}_{ij} &= e^{S_{ij} - \hat{m}_{ij}} \in \mathbb{R}^{B_r \times B_c} \\
\end{aligned}
$$

$Q_i K_j^T$ and $\hat{P}_{ij} V_j$ both require $2B_r B_c d$ flops. Therefore the total flops is $4 T_c T_r B_r B_c d = 4n^2d$ flops.

It will also be interesting to take a look at the `exp` operations, in the inner loop we have the following

$$
\begin{aligned}
\hat{P}_{ij} &= e^{S_{ij} - \hat{m}_{ij}} \in \mathbb{R}^{B_r \times B_c} \\
l_i' &= e^{m_i - m_i'} l_i + e^{\hat{m}_{ij} - m_i'} \hat{l}_{ij} \in \mathbb{R}^{B_r} \\
O_i' &= \text{diag}(l_i')^{-1} (\text{diag}(l_i) e^{m_i - m_i'} O_i + e^{\hat{m}_{ij} - m_i'} \hat{P}_{ij} V_j) \in \mathbb{R}^{B_r \times d}
\end{aligned}
$$

$`e^{S_{ij} - \hat{m}_{ij}}`$ requires $B_r B_c$ `exp` operations, $`e^{m_i - m_i'}`$ and $`e^{\hat{m}_{ij} - m_i'}`$ each requires $B_r$ `exp` operations. So the total number of `exp` operations is $T_r T_c (B_r B_c + 2 B_r) = n^2 (1 + \frac{2}{B_c})$.

For the backward pass, the trick is that we can always reconstruct the attention scores using $m$ and $l$, because $y_i = \frac{e^{x_i - m_N}}{l_N}$.

Let's first review the values we have on HBM

- $Q \in \mathbb{R}^{n \times d}$: Query matrix
- $K \in \mathbb{R}^{n \times d}$: Key matrix
- $V \in \mathbb{R}^{n \times d}$: Value matrix
- $O \in \mathbb{R}^{n \times d}$: Output matrix
- $dO \in \mathbb{R}^{n \times d}$: Output gradient
- $m \in \mathbb{R}^n$: Maximum attention weights
- $l \in \mathbb{R}^n$: Softmax denominator

For outputs, we pre-allocate some extra tensors on HBM

- $dQ \in \mathbb{R}^{n \times d}$: Query matrix gradient
- $dK \in \mathbb{R}^{n \times d}$: Key matrix
- $dV \in \mathbb{R}^{n \times d}$: Value matrix

Let's assume that $Q, dQ, O, dO$ are divided into blocks of shape $\mathbb{R}^{B_r \times d}$, and $K, dK, V, dV$ are divided into blocks of shape $\mathbb{R}^{B_c \times d}$.

There will be $T_r = \lceil \frac{n}{B_r} \rceil$ $Q, dQ, O, dO$ blocks and $T_c = \lceil \frac{n}{B_c} \rceil$ $K, dK, V, dV$ blocks respectively.

The backward computation process can be implemented as

---

```
for j = 1 to T_c
```

Outer loop:

Load $K_j, V_j \in \mathbb{R}^{B_c \times d}$ from HBM into SRAM. Initialize $dK_j, dV_j = (0)_{B_c \times d} \in \mathbb{R}^{B_c \times d}$ on SRAM.

```
    for i = 1 to T_r
```

Inner loop:

Load $Q_i, dQ_i, O_i, dO_i \in \mathbb{R}^{B_r \times d}$ and $m_i, l_i \in \mathbb{R}^{B_r}$ from HBM into SRAM

1. Computation

$$
\begin{aligned}
S_{ij} &= Q_i K_j^T \in \mathbb{R}^{B_r \times B_c} \\
P_{ij} &= \text{diag}(l_i)^{-1} e^{S_{ij} - m_i} \in \mathbb{R}^{B_r \times B_c} \\
dV_j &\leftarrow dV_j + P_{ij}^T dO_i \in \mathbb{R}^{B_c \times d} \\
dP_{ij} &= dO_i V_j^T \in \mathbb{R}^{B_r \times B_c} \\
D_i &= \text{rowsum}(dO_i \odot O_i) \in \mathbb{R}^{B_r} \\
dS_{ij} &= P_{ij} \odot (dP_{ij} - D_i) \in \mathbb{R}^{B_r \times B_c} \\
dQ_i' &\leftarrow dQ_i + dS_{ij} K_j \in \mathbb{R}^{B_r \times d} \\
dK_j &\leftarrow dK_j + dS_{ij}^T Q_i \in \mathbb{R}^{B_c \times d}
\end{aligned}
$$

2. Write back HBM

$$
dQ_i = dQ_i'
$$

```
    end
```

Write back $dK_j, dV_j \in \mathbb{R}^{B_c \times d}$ to HBM.

```
end
```

---

Now, let's take a look at the memory access and flops pattern assuming that we're using bfloat16.

On HBM, we have $Q, dQ, K, dK, V, dV, O, dO$ consume $16nd$ bytes and $m, l$ consume $4n$ bytes. Therefore the total HBM consumption is $16nd + 4n$.

In the outer loop, $K_j, V_j \in \mathbb{R}^{B_c \times d}$ are read into SRAM, there are $4 B_c d$ bytes HBM read. Since the outer loop repeats $T_c$ times, the total read volume is $4 B_c d T_c = 4nd$. On the other hand, $dK_j, dV_j \in \mathbb{R}^{B_c \times d}$ are written back to HBM, accounting for $4B_c d$ bytes HBM write, and the total write volume is $4B_c d T_c = 4nd$.

In the inner loop, $Q_i, dQ_i, O_i, dO_i \in \mathbb{R}^{B_r \times d}$ and $m_i, l_i \in \mathbb{R}^{B_r}$ are read from HBM, creating $8 B_r d + 4 B_r$ bytes read. Since the inner loop repeats $T_c T_r$ times, the total read volume is $(8 B_r d + 4 B_r)T_r T_c = n(8d + 4)T_c$. As for the write back, $dQ_i \in \mathbb{R}^{B_r \times d}$ is written back to HBM, creating $2 B_r d$ bytes write. Since the inner loop repeats $T_c T_r$ times, the total write volume is $2 B_r d T_r T_c = 2ndT_c$.

Therefore, the total HBM access volume is roughly $4nd + 4nd + n(8d + 4)T_c + 2ndT_c \approx 2nd (4 + 5\lceil\frac{n}{B_c}\rceil)$ bytes.

Apparently, by having a larger $B_c$, we significantly reduce the HBM access volume. $K_j, dK_j, V_j, dV_j \in \mathbb{R}^{B_c \times d}$ and $Q_i, dQ_i, O_i, dO_i \in \mathbb{R}^{B_r \times d}$ takes $8 (B_c + B_r) d$ bytes on SRAM, for a SRAM of size $M$, we have

$$
\begin{aligned}
8 (B_c + B_r) d &\leq M \\
B_c + B_r &\leq \frac{M}{8d}
\end{aligned}
$$

Thus we set $B_c = \lceil\frac{M}{16d}\rceil, B_r = \min(\lceil\frac{M}{16d}\rceil, d)$. The former helps reduce the HBM access while the latter ensures GEMM efficiency in the inner loop.

The primary flops are spent on GEMM operations

$$
\begin{aligned}
S_{ij} &= Q_i K_j^T \in \mathbb{R}^{B_r \times B_c} \\
dV_j &\leftarrow dV_j + P_{ij}^T dO_i \in \mathbb{R}^{B_c \times d} \\
dP_{ij} &= dO_i V_j^T \in \mathbb{R}^{B_r \times B_c} \\
dQ_i' &\leftarrow dQ_i + dS_{ij} K_j \in \mathbb{R}^{B_r \times d} \\
dK_j &\leftarrow dK_j + dS_{ij}^T Q_i \in \mathbb{R}^{B_c \times d}
\end{aligned}
$$

$Q_i K_j^T$ and $dO_i V_j^T$ each requires $2B_r B_c d$ flops. $P_{ij}^T dO_i$ and $dS_{ij}^T Q_i$ each requires $2 B_r d B_c$ flops. $dS_{ij} K_j$ requires $2 B_r B_c d$ flops. Therefore the total flops is $10 T_c T_r B_r B_c d = 10n^2d$ flops, which is $2.5\times$ that of forward passes. Compare with a regular backward pass of a linear layer, the extra $0.5\times$ comes from the fact that we need to recompute attention scores on the fly.

It will also be interesting to take a look at the `exp` operations, in the inner loop we have the following

$$
P_{ij} = \text{diag}(l_i)^{-1} e^{S_{ij} - m_j} \in \mathbb{R}^{B_r \times B_c}
$$

$e^{S_{ij} - m_j}$ requires $B_r B_c$ `exp` operations, the total flops is $T_r T_c B_r B_c = n^2$.

## Flash Attention 2

Flash Attention 2 improves upon the original algorithm by addressing inefficiencies in work partitioning and parallelism. The key changes are **inverting the loop order** and **reducing non-GEMM operations**.

### Update to the Recurrence Formula

In the naive 2-pass attention algorithm, we have

$$
\begin{aligned}
o_i &= \sum_{j = 1}^i y_j V[j, :] \\
    &= \sum_{j = 1}^i \frac{e^{x_j - m_N}}{l_N} V[j, :]
\end{aligned}
$$

If we define $o_i' = \sum_{j = 1}^i e^{x_j - m_i} V[j, :]$, we have

$$
\begin{aligned}
o_i' &= \sum_{j = 1}^i e^{x_j - m_i} V[j, :] \\
&= \sum_{j = 1}^{i - 1} e^{x_j - m_i} V[j, :] + e^{x_i - m_i} V[i, :] \\
&= e^{m_{i - 1} - m_i} o_{i - 1}' + e^{x_i - m_i} V[i, :] \\
o_N &= \frac{o_N'}{l_N}
\end{aligned}
$$

### Inverting the Loop Order

In Flash Attention 1, the outer loop iterates over $K, V$ blocks (columns), and the inner loop iterates over $Q$ blocks (rows). This requires writing partial results of $O$ to HBM and reading them back in the next iteration, leading to excessive memory traffic.

Flash Attention 2 flips this: the outer loop iterates over $Q$ blocks, and the inner loop iterates over $K, V$ blocks.

- Since we process a block of queries $Q_i$ entirely in the inner loop, we can keep the output accumulator $O_i$ in on-chip SRAM (or registers) throughout the computation.
- $O_i$ is written to HBM only once at the end.
- This avoids the read-modify-write overhead for $O$.

### Forward

Given the same block setup:

- $Q$ partitioned into blocks of size $B_r \times d$ ($T_r$ blocks).
- $K, V$ partitioned into blocks of size $B_c \times d$ ($T_c$ blocks).

The algorithm proceeds as follows:

---

```
for i = 1 to T_r
```

Outer loop (parallelized over sequence length):

Load $Q_i \in \mathbb{R}^{B_r \times d}$ from HBM to SRAM.
Initialize $`O_i = (0)_{B_r \times d}`$ and statistics $`m_i = (-\infty)_{B_r}`$, $`\ell_i = (0)_{B_r}`$ in SRAM.

```
    for j = 1 to T_c
```

Inner loop:

Load $K_j, V_j \in \mathbb{R}^{B_c \times d}$ from HBM to SRAM.

1. Computation

$$
\begin{aligned}
S_{ij} &= Q_i K_j^T \in \mathbb{R}^{B_r \times B_c} \\
\hat{m}_{ij} &= \text{rowmax}(S_{ij}) \in \mathbb{R}^{B_r} \\
m_i^{new} &= \max(m_i, \hat{m}_{ij}) \in \mathbb{R}^{B_r} \\
\hat{P}_{ij} &= e^{S_{ij} - m_i^{new}} \in \mathbb{R}^{B_r \times B_c} \\
\ell_i^{new} &= e^{m_i - m_i^{new}} \ell_i + \text{rowsum}(\hat{P}_{ij}) \in \mathbb{R}^{B_r}
\end{aligned}
$$

2. Update Output (Unnormalized)

$$
O_i \leftarrow \text{diag}(e^{m_i - m_i^{new}}) O_i + \hat{P}_{ij} V_j
$$

3. Update Statistics

$$
\begin{aligned}
m_i &= m_i^{new} \\
\ell_i &= \ell_i^{new}
\end{aligned}
$$

```
    end
```

Finalize and Write Back:

$$
\begin{aligned}
O_i &= \text{diag}(\ell_i)^{-1} O_i \\
L_i &= m_i + \log(\ell_i)
\end{aligned}
$$

Write $O_i$ to HBM.
Write $L_i$ to HBM (needed for backward pass).

```
end
```

---

### Analysis

- **HBM Access:** $O$ is now written only once per block $Q_i$, eliminating the $O(N \cdot T_c)$ read/write traffic of FA1.
- **Parallelism:** The outer loop over $Q$ blocks allows parallelizing across the sequence length dimension. Each thread block handles a distinct chunk of queries and writes to a disjoint part of the output matrix, removing synchronization needs.

### Backward

In the backward pass of Flash Attention 1, the term $D_i = \text{rowsum}(dO_i \odot O_i)$ was computed inside the inner loop. Since $dO_i$ and $O_i$ reside in HBM, this required repeated reads of $O_i$ inside the loop.

Flash Attention 2 optimizes this by **precomputing** $D$ before entering the attention loop.

$$
D = \text{rowsum}(dO \odot O) \in \mathbb{R}^n
$$

This vector $D$ is small ($O(n)$) compared to the matrices ($O(n \times d)$) and can be computed in a single pass over $dO$ and $O$.

By computing $D$ upfront:

1.  We avoid reading the large matrix $O$ multiple times inside the backward loop.
2.  We remove the $O(N \cdot d)$ computation from the critical path of the inner loop.
3.  Inside the backward kernel, we only need to load the scalar values $D_i$ corresponding to the current query block, which is very cheap.

The backward process maintains the loop order from FA1 (outer loop over $K, V$) but leverages the precomputed $D$ and $L$ to simplify computation:

---

1. Precomputation

$$
D = \text{rowsum}(dO \odot O) \in \mathbb{R}^n
$$

2. Backward Loop

```
for j = 1 to T_c
```

Outer loop (parallelized over sequence length $T_c$):

Load $K_j, V_j \in \mathbb{R}^{B_c \times d}$ from HBM to SRAM.
Initialize $dK_j, dV_j = (0)_{B_c \times d}$ in SRAM.

```
    for i = 1 to T_r
```

Inner loop:

Load $Q_i, dO_i \in \mathbb{R}^{B_r \times d}$ and $L_i, D_i \in \mathbb{R}^{B_r}$ from HBM to SRAM.

$$
\begin{aligned}
S_{ij} &= Q_i K_j^T \in \mathbb{R}^{B_r \times B_c} \\
P_{ij} &= e^{S_{ij} - L_i} \in \mathbb{R}^{B_r \times B_c} \\
dV_j &\leftarrow dV_j + P_{ij}^T dO_i \\
dP_{ij} &= dO_i V_j^T \in \mathbb{R}^{B_r \times B_c} \\
dS_{ij} &= P_{ij} \odot (dP_{ij} - D_i) \in \mathbb{R}^{B_r \times B_c} \\
dQ_i &\leftarrow_{\text{atomic}}^{\text{HBM}} dQ_i + dS_{ij} K_j \\
dK_j &\leftarrow dK_j + dS_{ij}^T Q_i
\end{aligned}
$$

```
    end
```

Write $dK_j, dV_j$ to HBM.

```
end
```

---

## Flash Attention 3

## References

- [From Online Softmax to FlashAttention](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf)
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/pdf/2205.14135)
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/pdf/2307.08691)
- [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/pdf/2407.08608)
