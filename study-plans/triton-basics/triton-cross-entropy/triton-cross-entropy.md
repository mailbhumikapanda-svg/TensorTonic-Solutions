# Cross Entropy Loss (Mean Reduction)

Cross entropy is the first kernel in the curriculum that needs **cross-program** combination. Each program computes a per-row scalar—the negative log likelihood of that row's target—and every row's scalar is added into a single global loss value before the host divides by the batch size.

The numerical core is log-sum-exp: subtract the row maximum before exponentiating so the largest exponent is exactly zero. This prevents overflow while requiring only one pass over each logits row.

---

## The Operation

Given logits of shape $(B, C)$ and integer targets of shape $(B,)$, mean-reduction cross entropy is

$$
\text{loss} = \frac{1}{B}\sum_{i=0}^{B-1}\left(\log\sum_{c=0}^{C-1}e^{\texttt{logits}[i,c]}-\texttt{logits}[i,t_i]\right).
$$

The per-row term is the gap between the row's log-sum-exp and the logit at the target index. Let

$$
m_i = \max_c \texttt{logits}[i,c].
$$

Factoring out $m_i$ gives the numerically stable form:

$$
\text{loss}_i=m_i+\log\sum_{c=0}^{C-1}e^{\texttt{logits}[i,c]-m_i}-\texttt{logits}[i,t_i].
$$

After the substitution, the largest exponent inside the sum is exactly zero. The sum therefore lies in $[1,C]$, and its logarithm lies in $[0,\log C]$. Without this subtraction, direct $e^{\texttt{logits}}$ can overflow fp32 for logits above roughly $88$.

---

## Kernel Contract

The implementation in `triton-cross-entropy.py` expects:

- `logits`: a contiguous 2-D fp32 tensor of shape $(B,C)$
- `target`: a 1-D integer tensor of shape $(B,)$
- `loss_out`: a preallocated 1-D tensor containing one fp32 accumulator
- Valid target indices satisfying $0 \leq t_i < C$
- $B>0$ and $C>0$, because this implementation does not validate either case

The Triton kernel returns no tensor. It atomically adds each row's loss into `loss_out[0]`, and the host wrapper divides that accumulator by $B$.

The wrapper should also use a loss accumulator with the same dtype and device as the logits:

```python
loss_out = torch.zeros((1,), dtype=logits.dtype, device=logits.device)
loss_out.zero_()
```

The explicit `zero_()` is part of the contract. It prevents stale values from a previous launch or call.

---

## Program Decomposition

Rows are independent up to the final mean. The launch grid is $(B,)$, with one Triton program per row.

Each program:

1. Reads its row index with `tl.program_id(0)`.
2. Loads the entire row's logits tile.
3. Computes the row maximum with `tl.max`.
4. Subtracts the maximum, exponentiates, and sums with `tl.sum`.
5. Loads the target and target logit.
6. Atomically adds the resulting row loss to `loss_out[0]`.

This is the simplest cross-program reduction: one scalar per program and one length-$1$ global accumulator. An alternative is to store one scalar per row in a length-$B$ buffer and launch a separate reduction kernel. That avoids atomic contention but requires another launch and temporary storage.

For small and medium batches, the atomic form generally has lower launch and memory costs. For very large batches, atomic contention and float32 accumulation order can become significant. The host-side `loss_out.div_(B)` is a single in-place operation and avoids adding the batch mean to the kernel's reduction.

---

## Tile Shape and Masking

The tile shape is $(\texttt{BLOCK\_SIZE},)$, where

```python
BLOCK_SIZE = triton.next_power_of_2(C)
```

and `BLOCK_SIZE` is declared `tl.constexpr` in the kernel.

The load mask `cols < C` prevents overshoot for vocabularies or classes that do not fill the tile. The logits load must use `other=-inf`, not `0.0`:

- A zero in `tl.max` would become the maximum for an all-negative row.
- A zero in `tl.exp` would contribute $1$ to the exp-sum.
- `-inf` is the additive/max identity: it does not affect `tl.max` and contributes $e^{-inf}=0$ to `tl.sum`.

There is no mask on the target load. Each program loads one integer and one target logit:

```python
target_id = tl.load(target_ptr + row)
target_logit = tl.load(logits_ptr + row * stride_logits_row + target_id)
```

Target-logit access is the only scalar load outside the full-row tile. The target could instead be gathered from the loaded row, but one scalar reload is simpler and avoids an extra indexed gather in the elementwise pass.

---

## Memory Hierarchy and Reuse

The logits row is loaded once into a $(\texttt{BLOCK\_SIZE},)$ tile and held in registers for:

- The row maximum.
- The shift and exponential.
- The exponent sum.

The target integer is loaded once, and the target logit is gathered from the logits row. There is no shared-memory stage in this kernel. The compiler lowers `tl.max` and `tl.sum` to tree reductions, likely using small staging areas between warps or threads.

For fp32 inputs, the idealized per-row traffic is:

- $4C$ bytes: logits tile load
- $4$ bytes: target load
- One scalar target-logit value: implementation-dependent cache behavior
- $4$ bytes per program: atomic accumulator read-modify-write traffic, amortized globally

The value $4C+12$ from a naïve accounting is not a reliable HBM measurement. Cache hits, the row stride, coalescing, and atomic hardware all change the observed value. Treat memory traffic estimates as bounds or order-of-magnitude comparisons rather than exact performance predictions.

---

## Memory-Bound vs Compute-Bound

Per fp32 logit element, the kernel performs roughly five arithmetic operations:

- One max contribution
- One shift subtraction
- One exponent calculation
- One sum contribution
- An amortized share of the row log, target subtraction, and mean division

With an approximate $4$-byte logits load, arithmetic intensity is

$$
\frac{\approx 5\text{ FLOPs}}{\approx 4\text{ bytes}}\approx1.25\text{ FLOPs/byte}.
$$

That is still memory-bound on modern accelerators. The exponential is expensive, but one logits load of the row is the dominant scaling term as $C$ grows.

Fused softmax has nearly the same intensity, but writes a complete normalized row instead of one scalar. Cross entropy saves the large normalized-output write and therefore generally requires less memory bandwidth. A final run comparison is still needed because the atomic loss accumulation adds a small cross-row cost.

---

## Compiler-Handled vs Author-Handled

**Author chooses:**

- The $(B,)$ launch grid
- The next-power-of-two block size
- The tail mask
- `other=-inf` for masked logits
- Max-before-exp placement
- Scalar target-logit loading
- Atomic accumulation and host-side division

**Compiler handles:**

- Lowering `tl.max` and `tl.sum` to tree reductions
- Selecting loads, vector widths, and register allocation
- Scheduling the row tile operations
- Lowering `tl.atomic_add` to the available atomic instruction
- Inserting intra-row synchronization for reductions

The author remains responsible for the reduction contract at the program boundary. Atomic accumulation is not nondeterministic in the sense of losing updates, but it does not define a stable floating-point addition order. The accumulator order can vary between runs.

For fp32 tests, a tolerance of roughly `atol=1e-2, rtol=1e-2` is reasonable for the atomic form. Tighter absolute tolerances can fail for correct implementations. The appropriate tolerance should be based on the dtype and batch size, not copied blindly from a different kernel.

---

## Naive vs Optimized

A naïve eager PyTorch implementation may launch separate kernels for:

- Computing the row maximum
- Subtracting, exponentiating, and summing
- Taking the logarithm and subtracting target logits
- Performing the global mean reduction

The fused Triton kernel loads and processes the logits row once, writes one scalar per program, and performs one global atomic add.

For the standalone forward kernel, this is a teaching baseline rather than the final production cross-entropy implementation. Further fusion is possible:

- **Forward and backward:** compute softmax or logits derivatives in the same kernel family and materialize gradients directly.
- **Vocabulary projection:** fuse the unembedding matrix multiply with cross entropy, avoiding materialization of the full $B\times C$ logits tensor.
- **Stable two-stage reduction:** write per-row losses to a buffer and reduce them with a tree for reproducible floating-point order.

In this implementation, the standalone forward kernel assumes logits already exist in HBM. It does not fuse the matrix multiplication that produces those logits.

---

## Worked Example

Take $B=2$, $C=3$, `BLOCK_SIZE=4`,

$$
\texttt{logits}=\begin{bmatrix}1&2&3\\0&0&5\end{bmatrix},\qquad
\texttt{target}=[2,0].
$$

The launch grid is $(2,)$.

**Program 0**, with `row=0`:

- Tile: $[1,2,3,-\infty]$
- Row maximum: $m_0=3$
- Shifted exponentials: $[e^{-2},e^{-1},e^0,0]=[0.1353,0.3679,1,0]$
- Sum: $1.5032$
- Stable log-sum-exp: $3+\log(1.5032)=3.4076$
- Target logit: $	exttt{logits}[0,2]=3$
- Row loss: $3.4076-3=0.4076$

**Program 1**, with `row=1`:

- Tile: $[0,0,5,-\infty]$
- Row maximum: $m_1=5$
- Shifted exponentials: $[e^{-5},e^{-5},e^0,0]\approx[0.0067,0.0067,1,0]$
- Sum: $1.0134$
- Stable log-sum-exp: $5+\log(1.0134)
  \approx5.0134$
- Target logit: $	exttt{logits}[1,0]=0$
- Row loss: $5.0134-0=5.0134$

After both programs finish:

$$
\texttt{loss\_out}[0]
=0.4076+5.0134
\approx5.4210.
$$

The host divides by $B=2$:

$$
\text{mean loss}=5.4210/2\approx2.7105.
$$

The max subtraction is what keeps the second row finite and numerically stable. Directly exponentiating $[0,0,5]$ is still safe in this small example, but adding a logit of $100$ would produce $e^{100}$ and overflow fp32.

---

## Edge Cases and Validation

The standalone kernel does not implement all safeguards:

- $B=0$ produces division by zero in `loss_out.div_(B)`.
- $C=0$ is not a valid logits shape and is not a meaningful tile size.
- Target indices outside $[0,C)$ must be rejected or guarded by the caller.
- `NaN` logits require an application-level policy because `tl.max` and `tl.sum` do not define a sensible cross-entropy value.
- `Inf` logits need special handling. They are not silently represented by stable finite-precision arithmetic.
- The implementation is intentionally focused on contiguous fp32 logits and integer targets.

The wrapper can validate shapes and target bounds before launching:

```python
assert logits.ndim == 2
assert target.shape == (logits.shape[0],)
assert logits.shape[0] > 0
assert logits.shape[1] > 0
assert logits.dtype == torch.float32
assert target.dtype in (torch.int32, torch.int64)
assert target.min() >= 0
assert target.max() < logits.shape[1]
```

Use a separate reduction kernel or PyTorch mean reduction if strict reproducibility across scheduling choices is required.

---

## Testing

A useful test matrix is:

- $B=1$, $C=1$
- $C$ as powers of two and non-powers of two
- All equal logits
- Large positive logits
- Large negative logits
- Mixed positive and negative logits
- Random fp32 logits
- Random valid targets
- Invalid target indices
- Empty batch and empty vocabulary
- `torch.float16`, `bfloat16`, and `float32`, if supported by the wrapper

Compare each valid case with:

```python
reference = torch.nn.functional.cross_entropy(
    logits,
    target,
    reduction="mean",
)
```

For fp32 atomic accumulation, tolerate last-ULP differences across repeated runs. Run the random comparison with several batch sizes, and test large batch sizes to expose atomic-order noise.

---

## Pitfalls

* **Skipping the max-subtraction.** Direct $\log\sum e^{\texttt{logits}}$ overflows for sufficiently large logits. Subtraction does not change the mathematical result and is required for a numerically valid implementation.

* **Loading masked lanes as `0.0`.** Zero becomes the maximum for a negative row and contributes $1$ to the exp-sum. Use `other=-inf`.

* **Forgetting to pre-zero the loss buffer.** `tl.atomic_add` accumulates onto the existing value. Call `loss_out.zero_()` before every launch.

* **Assuming atomic reduction is stable.** Atomic updates are not lost, but float additions occur in nondeterministic order. Correct implementations can differ in the last few decimal digits between runs.

* **Using a one-pass target gather without bounds checks.** Target load must be in range. Triton does not necessarily provide the same Python-level shape or bounds validation that PyTorch supplies.

* **Comparing memory traffic by logits size alone.** Cache behavior, dtype, stride, and atomic read-modify-write traffic prevent exact bandwidth formulas from characterizing a real launch.

---
