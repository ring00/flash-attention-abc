# Flash Attention Speedrun

## Symbols

- $Q \in \mathbb{R}^{n \times d}$: Query matrix
- $K \in \mathbb{R}^{n \times d}$: Key matrix
- $V \in \mathbb{R}^{n \times d}$: Value matrix
- $S \in \mathbb{R}^{n \times n}$: Attention scores (pre-softmax)
- $P \in \mathbb{R}^{n \times n}$: Attention weights (post-softmax)
- $O \in \mathbb{R}^{n \times d}$: Output matrix
- $n$: Sequence length
- $d$: Head dimension
- $L$: Scalar loss
- $dO = \frac{\partial L}{\partial O}$: Gradient of the loss with respect to the output


## Online Softmax

Given an input vector $`x = \{x_i\}_{i = 1}^N`$, the softmax operation generates a new vector

```math
\begin{aligned}
y &= \{y_i\}_{i = 1}^N \\
y_i &= \frac{e^{x_i}}{\sum_{j = 1}^N e^{x_j}}
\end{aligned}
```

To avoid overflow during the exponential operations, we usually rewrite the formula as

$$
\begin{aligned}
m &= \max(x) \\
y_i &= \frac{e^{x_i - m}}{\sum_{j = 1}^N e^{x_j - m}}
\end{aligned}
$$

It's very clear that in its current form, a naive way to compute $y$ is:

---
1. Initialization

$$
\begin{aligned}
m_0 &= -\infty \\
l_0 &= 0
\end{aligned}
$$

---
2. Geting maximum

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
3. Computing softmax denominator

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
4. Computing softmax results

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

This algorithm generates two intermediate sequences, one for the running maximum of the input vector

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

It's clear that $l_i$ relies on $m_N$, which is the only thing that's preventing us from computing both $m_i$ and $l_i$ together.

Let's look at an alternative variable

```math
\begin{aligned}
l' &= \{l_i'\}_{i = 1}^N \\
l_i' &= \sum_{j = 1}^i e^{x_j - m_j}
\end{aligned}
```

Notice that $l_N' = l_N$, which is exactly the softmax denominator. Therefore, we could use the following recurrence relation

$$
\begin{aligned}
m_i &= \max({m_{i - 1}, x_i}) \\
l_i' &= \sum_{j = 1}^i e^{x_j - m_j} = \sum_{j = 1}^{i - 1} e^{x_j - m_j} + e^{x_i - m_i} = l_{i - 1}' e^{m_{i - 1} - m_i} + e^{x_i - m_i}
\end{aligned}
$$

Based on the formula, $l_i'$ depends only on $l_{i - 1}'$, $m_{i - 1}$, $m_i$ and $x_i$, which means we can compute its value in one go.

---
1. Initialization

$$
\begin{aligned}
m_0 &= -\infty \\
l_0 &= 0
\end{aligned}
$$

---
2. Online computation

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
3. Computing softmax results

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

Given $Q, K, V \in \mathbb{R}^{n \times d}$, where $n \in \mathbb{N^+}$ is the sequence length and $d \in \mathbb{N^+}$ is the head dimension, attention operation computes the following

$$
\begin{aligned}
S &= QK^T \\
P &= \text{softmax}(S) \\
O &= PV
\end{aligned}
$$

Let's take a closer look at the attention computation over the $k$-th row of the query matrix $q = Q[k, :]$

---
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
$$
O[K, :] = o_N
$$

Let's take a closer look at the second loop, clearly we have

$$
o_i = \sum_{j = 1}^i y_j V[j, :] = \sum_{j = 1}^i \frac{e^{x_j - m_N}}{l_N} V[j, :]
$$

We can use the same trick which we applied in online softmax, we define a new variable

$$
o_i' = \sum_{j = 1}^i \frac{e^{x_j - m_i}}{l_i} V[j, :]
$$

which gives the following recurrence relation

$$
\begin{aligned}
o_i' &= \sum_{j = 1}^{i - 1} \frac{e^{x_j - m_i}}{l_i} V[j, :] + \frac{e^{x_i - m_i}}{l_i} V[i, :] \\
     &= o_{i - 1}' \frac{l_{i - 1} e^{m_{i - 1} - m_i}}{l_i} + \frac{e^{x_i - m_i}}{l_i} V[i, :]
\end{aligned}
$$

We can see that $o_i'$ only depends on $o_{i - 1}'$, $l_{i - 1}$, $l_i$, $m_{i - 1}$, $m_i$ and $x_i$, therefore we can fuse all calculations in one single loop

```
for i = 1 to N
```
$$
\begin{aligned}
x_i &= qK[i, :]^T \\
m_i &= \max(m_{i - 1}, x_i) \\
l_i &= l_{i - 1} e^{m_{i - 1} - m_i} + e^{x_i - m_i} \\
o_i' &= o_{i - 1}' \frac{l_{i - 1} e^{m_{i - 1} - m_i}}{l_i} + \frac{e^{x_i - m_i}}{l_i} V[i, :]
\end{aligned}
$$
```
end
```

## Backward

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

### Backward of Softmax

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

Let's focus on an vector at $i$-th row in $dS$

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
D = \{d_i\}_{i = 1}^N = \text{rowsum}(dO \odot O)
```

Plugging in the previous formula, we have

$$
ds = dp \odot p - \text{broadcast}(d) \odot p = p \odot (dp - \text{broadcast}(d))
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

## References

* [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/pdf/2205.14135)
* [From Online Softmax to FlashAttention](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf)
