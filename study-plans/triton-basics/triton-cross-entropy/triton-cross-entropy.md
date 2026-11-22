# Cross Entropy Loss (Mean Reduction)

This kernel introduces **cross-program reduction**: each Triton program computes one row's negative log-likelihood, and all programs combine those scalars into a single loss value.

The key numerical technique is **log-sum-exp**. Subtracting the row maximum before exponentiation keeps the largest exponent exactly $e^0=1$ and prevents overflow.

---

## Operation

Given logits $X \in \mathbb{R}^{B\times C}$ and integer targets $t \in \{0,\ldots,C-1\}^{B}$, mean cross entropy is

$$
L = \frac{1}{B}\sum_{i=0}^{B-1}
\left(
\log\left(\sum_{c=0}^{C-1}e^{X_{i,c}}\right)-X_{i,t_i}
\right).
$$

For row $i$, let

$$
m_i=\max_{0\leq c<C}X_{i,c}.
$$

The stable per-row expression is

$$
L_i=m_i+
\log\left(\sum_{c=0}^{C-1}e^{X_{i,c}-m_i}\right)
-X_{i,t_i}.
$$

Since at least one shifted exponent is exactly zero,

$$
1\leq\sum_c e^{X_{i,c}-m_i}\leq C.
$$

Consequently,

$$
0\leq\log\left(\sum_c e^{X_{i,c}-m_i}\right)\leq\log C.
$$

This keeps exponentiation bounded regardless of the absolute logit values, provided those values are finite.

> Directly evaluating $e^{X_{i,c}}$ in fp32 can overflow around $88$, even if the final cross-entropy value is finite.

---

## Kernel Contract

The implementation in [`triton-cross-entropy.py`](./triton-cross-entropy.py) uses:

| Argument | Type and shape | Purpose |
|---|---|---|
| `logits` | 2-D contiguous fp32 tensor $(B,C)$ | Row-wise logit values |
| `target` | 1-D integer tensor $(B,)$ | Target class for each row |
| `loss_out` | 1-D fp32 tensor $(1,)$ | Global fp32 accumulator |
| `stride_logits_row` | Python/int argument | Row stride of `logits` |
| `B`, `C` | Python/int arguments | Batch and class dimensions |
| `BLOCK_SIZE` | `tl.constexpr` | Tile width; normally `next_power_of_2(C)` |

The wrapper should preallocate and zero the accumulator:

```python
loss_out = torch.zeros((1,), dtype=logits.dtype, device=logits.device)
loss_out.zero_()
```

It should then launch one Triton program per row and divide once on the host:

```python
cross_entropy_kernel[(B,)](
    logits,
    target,
    loss_out,
    logits.stride(0),
    B,
    C,
    BLOCK_SIZE=triton.next_power_of_2(C),
)
loss = loss_out[0] / B
```

The kernel's contract assumes:

- $B>0$
- $C>0$
- `logits` is 2-D and contiguous
- `target` is 1-D with length $B$
- $0\leq t_i<C$ for every row
- `target` and `logits` are on the same device

The standalone implementation does not currently perform shape or bounds validation; validate these values in the host wrapper or a test harness.

---

## Program Decomposition

The launch grid is $(B,)$, with exactly one Triton program for each row.

Each program:

1. Gets `row = tl.program_id(0)`.
2. Creates a one-dimensional class tile:

   ```python
   cols = tl.arange(0, BLOCK_SIZE)
   mask = cols < C
   ```

3. Loads the row:

   ```python
   row_ptr = logits_ptr + row * stride_logits_row + cols
   logits = tl.load(row_ptr, mask=mask, other=-float("inf"))
   ```

4. Computes `row_max = tl.max(logits, axis=0)`.
5. Computes the stable sum:

   ```python
   shifted = logits - row_max
   sum_exp = tl.sum(tl.exp(shifted), axis=0)
   lse = tl.log(sum_exp) + row_max
   ```

6. Loads the target and target logit.
7. Atomically adds the row loss:

   ```python
   row_loss = lse - target_logit
   tl.atomic_add(loss_out_ptr, row_loss)
   ```

Each row is independent until the final global accumulator. Atomic addition avoids a temporary length-$B$ per-row loss buffer.

An alternative two-stage implementation writes every `row_loss` to an intermediate buffer and launches a separate reduction kernel. It avoids atomic contention but adds launch overhead and memory traffic.

---

## Tile Shape and Masking

The block width is

```python
BLOCK_SIZE = triton.next_power_of_2(C)
```

and must be passed to the kernel as `tl.constexpr`.

For $C=3$, the tile width is $4$:

```text
valid lanes:  [0, 1, 2]
tail lane:    [3]
```

The tail mask disables the extra lane. The padded lane must contain `-inf`:

```python
other=-float("inf")
```

`-inf` is the correct neutral element for this computation:

- `tl.max(...)` ignores it in favor of a finite logits value.
- `tl.exp(-inf) == 0`, so it contributes nothing to `tl.sum(...)`.
- Masked lanes therefore never appear in the row result.

Using `0.0` is incorrect:

- It can become the maximum of an all-negative row.
- It contributes $e^0=1$ to the exponent sum.

---

## Target Loading

The target tensor is not tiled:

```python
target_id = tl.load(target_ptr + row)
```

The corresponding logit is loaded by scalar pointer arithmetic:

```python
target_logit = tl.load(
    logits_ptr
    + row * stride_logits_row
    + target_id
)
```

This is one scalar HBM or cache read for the target logit. Valid target indices must be guaranteed; Triton does not provide Python-level bounds checks for this pointer load.

---

## Memory and Reuse

The logits tile is loaded once and reused for:

- Row maximum
- Shift-and-exponent calculation
- Exponent sum
- Target logit loading if the target scalar is not gathered from the row tile

There is no shared-memory stage in this standalone kernel. `tl.max` and `tl.sum` are compiler-generated tree reductions, with staging and synchronization chosen by the Triton/target compiler.

For fp32 inputs, approximate per-row traffic is:

| Operation | Approximate amount |
|---|---:|
| Logits tile load | $4C$ bytes |
| Target load | $4$ bytes |
| Target-logit load | Scalar; cache dependent |
| Atomic accumulator update | $4$ bytes of logical work per program |

The exact HBM traffic depends on caching, stride, vectorization, coalescing, and atomic implementation. A formula such as $4C+12$ bytes should be treated only as an idealized accounting, not an exact hardware measurement.

---

## Performance Characterization

Per fp32 logit element, the kernel performs approximately:

- One comparison contribution for the maximum
- One subtraction
- One exponential
- One sum contribution

The amortized row operations add the logarithm, target subtraction, and mean division.

An approximate arithmetic intensity is

$$
\frac{\approx5\ \text{FLOPs}}{\approx4\ \text{bytes}}
\approx1.25\ \text{FLOPs/byte}.
$$

This is far below typical accelerator roofline crossover regions, so the kernel is primarily memory-bound for realistic logits rows. Its runtime is expected to scale with the logits read rather than solely with the exponential throughput.

Fused softmax has similar arithmetic, but writes an entire normalized $C$-element row. This cross-entropy kernel instead writes one scalar, reducing output traffic at the cost of atomic cross-row accumulation.

---

## Atomic Accumulation and Precision

`tl.atomic_add` ensures that row additions update the same accumulator without individual updates being overwritten. It does not guarantee a reproducible addition order.

In fp32, final values can differ in the last few digits or ULPs depending on scheduling and atomic order:

- Result differences are usually small.
- Very large batch sizes can amplify drift.
- Tighter tolerance does not improve the implementation; it can make a valid implementation fail.

For fp32 tests, a starting tolerance may be

```python
atol=1e-2
rtol=1e-2
```

Choose tolerances from the observed dtype and batch size. For strict reproducibility, use a two-stage reduction:

1. Write each row loss to a length-$B$ buffer.
2. Sum the buffer in a deterministic or tree-structured reduction.

---

## Author vs. Compiler

### Author responsibilities

- Choose the program grid.
- Choose the block shape.
- Mask tail lanes.
- Use `-inf` for padded logits.
- Apply max-before-exp.
- Load the target and target logit.
- Accumulate row losses and divide once on the host.

### Compiler responsibilities

- Lower `tl.max` and `tl.sum` to reduction instructions or trees.
- Allocate tile and reduction registers.
- Choose vector and load widths.
- Schedule elementwise and reduction operations.
- Lower `tl.atomic_add` to an atomic instruction.

The author decides **what is reduced** and **where reduction boundaries occur**. The compiler decides **how** those reductions are implemented on the target device.

---

## Naive vs. Fused Implementation

A non-fused eager implementation can execute separate operations for:

- Row maximum
- Stable shift and exponentiation
- Exponent sum and logarithm
- Target-logit subtraction
- Batch mean reduction

Each intermediate may be materialized in HBM and read again.

The fused implementation:

- Loads each logits row once.
- Keeps the row tile in registers.
- Produces only one row scalar.
- Atomically combines $B$ scalars.

Further production-level optimizations include:

- **Forward/backward fusion:** compute softmax derivatives and materialize gradients in a fused path.
- **Vocabulary-projection fusion:** combine the unembedding matrix multiply with per-token log-sum-exp and target selection.
- **Deterministic reduction:** replace atomic summation with a per-row buffer and tree reduction.

This file describes the standalone forward kernel, not the gradient or unembedding portions of a language-model training step.

---

## Worked Example

Let $B=2$, $C=3$, and `BLOCK_SIZE=4`. Consider

$$
X=
\begin{bmatrix}
1 & 2 & 3\\
0 & 0 & 5
\end{bmatrix},
\qquad
 t=
\begin{bmatrix}
2\\
0
\end{bmatrix}.
$$

The launch grid is $(2,)$.

### Program 0: $i=0$

Loaded tile:

$$
[1,2,3,-\infty]
$$

The maximum is $m_0=3$. The shifted exponentials are

$$
[e^{-2},e^{-1},e^0,0]
\approx[0.1353,0.3679,1,0],
$$

so

$$
\sum_c e^{X_{0,c}-m_0}\approx1.5032.
$$

Therefore,

$$
L_0
=3+\log(1.5032)-X_{0,2}
=3+0.4076-3
=0.4076.
$$

### Program 1: $i=1$

Loaded tile:

$$
[0,0,5,-\infty]
$$

The maximum is $m_1=5$. The shifted exponentials are

$$
[e^{-5},e^{-5},e^0,0]
\approx[0.0067,0.0067,1,0],
$$

so

$$
\sum_c e^{X_{1,c}-m_1}\approx1.0135.
$$

Thus,

$$
L_1
=5+\log(1.0135)-X_{1,0}
=5+0.0134-0
=5.0134.
$$

After both programs complete,

$$
\texttt{loss\_out}[0]
\approx0.4076+5.0134
=5.4210.
$$

The host computes the mean:

$$
L=\frac{5.4210}{2}\approx2.7105.
$$

The stable calculation avoids overflow. In contrast, direct $e^{100}$ overflows fp32, even though

$$
100+\log\left(\sum_c e^{X_c-100}\right)-X_{t}
$$

may remain finite.

---

## Edge Cases

The standalone implementation should not be called when:

- $B=0$, because the host mean divides by zero.
- $C=0$, because the log-sum-exp is undefined and no valid target exists.
- A target is outside $[0,C)$.
- A row contains `NaN`, unless the caller explicitly defines the required behavior.
- A row contains `Inf`; stable finite-precision arithmetic alone does not define all corner cases involving infinities.

A host wrapper can validate the common cases:

```python
if logits.ndim != 2:
    raise ValueError("logits must be a 2-D tensor")
if logits.dtype != torch.float32:
    raise TypeError("this kernel requires fp32 logits")

B, C = logits.shape
if B == 0:
    raise ValueError("batch size must be positive")
if C == 0:
    raise ValueError("class count must be positive")
if target.shape != (B,):
    raise ValueError("target shape must be (B,)")
if target.dtype not in (torch.int32, torch.int64):
    raise TypeError("target must contain int32 or int64 values")
if target.numel() and (
    target.min().item() < 0 or target.max().item() >= C
):
    raise ValueError("target contains an out-of-range class index")
```

Avoid evaluating `target.min()` on a device or dtype unsupported by the wrapper when an equivalent shape/type check can be performed first.

---

## Testing

Use this test matrix:

- $B=1$, $C=1$
- Powers of two and non-powers of two for $C$
- Equal logits
- Large positive logits
- Large negative logits
- Mixed logits
- Random fp32 logits and valid targets
- Invalid target indices
- Empty batch and empty vocabulary
- Several repeated launches to observe atomic-order drift
- Small and large batch sizes

Compare valid values with PyTorch:

```python
reference = torch.nn.functional.cross_entropy(
    logits,
    target,
    reduction="mean",
)

torch.testing.assert_close(
    result,
    reference,
    atol=1e-2,
    rtol=1e-2,
)
```

For fp32, repeated launches may differ in the final ULPs. The test should verify numerical correctness without assuming a fixed atomic accumulation order.

---

## Common Pitfalls

- **Skipping max subtraction** — Direct exponentiation can overflow for moderate positive logits.
- **Using `0.0` as the padded value** — It is neither the max identity nor the exp-sum identity.
- **Forgetting `loss_out.zero_()`** — Atomic adds accumulate on stale values.
- **Calling this kernel with $B=0$** — The host-side mean is undefined and divides by zero.
- **Passing an out-of-range target** — The target-logit scalar load can access invalid memory.
- **Expecting exact reproducibility** — Atomic order is nondeterministic for floating-point sums.
- **Treating $4C+12$ as exact HBM traffic** — Cache hits and hardware atomics change the real transfer count.
- **Using an incorrectly typed accumulator** — Accumulate in a dtype compatible with the logits and test the wrapper contract.
- **Tightening tolerance below atomic drift** — A correct kernel may differ by a few ULPs across launches.

---
