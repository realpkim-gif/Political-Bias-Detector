# Mixed Precision Training (AMP) Notes

Notes on `torch.amp.autocast`, `torch.amp.GradScaler`, and why we switched from fp16 to bf16, in `main.py`'s `train()` function.

## Update: switched from fp16 to bf16 — GradScaler is gone

After multiple training runs (over several days) where the loss stayed completely flat at `~1.0986` (exactly `ln(3)`, i.e. "the model knows nothing") despite fixing the learning rate and an early-stopping bug, we added a diagnostic to check whether `GradScaler` was silently skipping optimizer steps due to fp16 overflow. Rather than keep chasing fp16-specific instability, we switched to **bf16**, which removes the entire class of problem:

- **What bf16 is**: "Brain Float 16" — a different 16-bit floating point format than fp16 (`float16`). Both use 16 bits total, but they split those bits differently:
  - **fp16**: 5 exponent bits, 10 mantissa bits → small range, more precision within that range.
  - **bf16**: 8 exponent bits (same as fp32!), 7 mantissa bits → fp32's full range, less precision than fp16.
- **Why that matters here**: fp16's narrow range is exactly what caused the underflow/overflow problem `GradScaler` existed to manage — tiny gradients underflowing to `0.0`, or oversized ones overflowing to `inf`/`NaN`. bf16 shares fp32's range (just with coarser precision), so values that would have overflowed or underflowed in fp16 simply don't hit those limits in bf16 — no scaling trick needed to dodge a cliff that no longer exists.
- **The tradeoff**: bf16 has fewer mantissa bits than fp16 (7 vs. 10), so it's *less precise* within its range. For this task that's an acceptable tradeoff — the whole point of `autocast` was already "accept mild rounding error on matmuls," and bf16's extra rounding is in the same spirit, just with no cliff-edge failure mode to worry about.
- **What this means for the code**: `GradScaler` is no longer needed anywhere, since there's no overflow/underflow to scale around. `torch.amp.autocast("cuda", dtype=torch.bfloat16)` replaces `torch.amp.autocast("cuda")`, and the training loop goes back to plain `loss.backward()` / `optimizer.step()` — no `scaler.scale()`/`scaler.step()`/`scaler.update()` dance.
- Confirmed working on this GPU: `torch.cuda.is_bf16_supported()` returns `True` on the RTX 4060 (Ada Lovelace architecture has native bf16 Tensor Core support).

Everything below this point describes the fp16 + `GradScaler` approach we started with — kept for reference/history, since the reasoning (what autocast does, why scaling fixes underflow but not rounding, etc.) is still accurate background even though the code no longer uses `GradScaler`.

## TL;DR (Too Long; Didn't Read)

- **`autocast`** — picks fp16/bf16 vs fp32 per-operation during the forward pass. Rounding error from this is small enough not to meaningfully hurt training.
- **`GradScaler`** (fp16 only, no longer used) — existed because fp16 gradients specifically could shrink to zero during backprop. Not needed with bf16, since bf16 doesn't have that narrow-range problem in the first place.

Everything below is the "why" behind the fp16/GradScaler approach, kept for reference — not something to re-derive day to day.

## Why we added this

Full fine-tuning of BERT-large (all ~340M parameters trainable, not just a small classifier head) needs to store activations across all 24 layers for every example in a batch, in order to compute gradients during `backward()`. This uses a lot of GPU memory — enough that even `batch_size=4` was running the GPU at ~97% memory capacity (7930MiB / 8188MiB), causing the training process to run extremely slowly (30+ hours without finishing even one confirmed epoch), likely due to memory fragmentation/thrashing near the memory ceiling.

Mixed precision training runs the expensive matrix math in fp16 (16-bit floats) instead of fp32 (32-bit floats), roughly halving memory use for those operations and often speeding up compute on GPUs with Tensor Cores (which the RTX 4060 has).

## AMP vs. autocast — these aren't two alternatives, one contains the other

"AMP" (Automatic Mixed Precision) is the name for the overall technique. `autocast` and `GradScaler` are the two tools PyTorch gives you to actually implement it, and they handle two different halves of one training step:

- **`autocast`** — handles the **forward pass**. It's the piece that actually decides, per-operation, whether to run in fp16 or fp32, and does the dtype casting.
- **`GradScaler`** — handles the **backward pass**. It does *not* do any dtype casting itself — it just scales the numeric size of the loss/gradients (multiplies, then later divides back down) so that fp16's small representable range doesn't cause gradients to underflow to zero during `backward()`.

So: `AMP = autocast (forward pass precision) + GradScaler (backward pass safety)`. You need both for training; `predict_in_batches` only uses `autocast`, since there's no `backward()` call during evaluation.

## Common mix-ups, corrected

- **"autocast is only used for prediction/eval"** — no. `autocast` wraps the forward pass in *both* training and eval. In `train()`, it wraps the exact same `output = model(...)`/loss computation that happens during training, not just in `predict_in_batches`.
- **"autocast and GradScaler both scale things"** — no. Only `GradScaler` scales (multiplies/divides numeric values). `autocast` never multiplies anything — it only chooses a *dtype* per operation. Calling both "scales" is a category error; they solve different problems with different mechanisms.
- **"GradScaler casts gradients to a different bit-width"** — no. `GradScaler` never changes any tensor's dtype. It only multiplies the loss up before `backward()` and divides the gradients back down before `optimizer.step()` — pure arithmetic, same dtype throughout. Casting between fp16/fp32 is entirely `autocast`'s job, not `GradScaler`'s.
- **"Sigmoid/softmax are examples of ops that need fp16 casting"** — backwards. Autocast deliberately keeps ops like `sigmoid`, `softmax`, and loss functions in fp32, because they're numerically sensitive. The ops that *do* get cast to fp16 are the expensive, numerically-tolerant ones — matrix multiplications (`Linear`, `matmul`, convolutions).
- **"The scale-up in GradScaler is for the optimizer/learning rate"** — not quite. The scale-**up** (`scaler.scale(loss)`) happens *before* `backward()`, purely to protect the gradient computation itself from underflowing. The scale-**down** (`scaler.step()`'s internal unscale) happens *after* `backward()`, right before `optimizer.step()` — that's the part that ensures the optimizer applies a correctly-sized update with the real learning rate.

## What stays fp32 vs what becomes fp16

**Model weights are always fp32.** AMP never converts your stored parameters. Only specific *operations* running inside an `autocast` block get a temporary fp16 copy of their inputs, used just for that one operation:

```python
with torch.amp.autocast("cuda"):
    output = model(test_input)          # matmuls inside here run in fp16
    loss = loss_function(output, ...)   # but the loss itself stays fp32
```

- Autocast keeps an internal list of "safe" ops (matrix multiplications, convolutions) that get cast to fp16, since these are the expensive operations and are numerically tolerant of lower precision.
- Precision-sensitive ops (softmax, layer norm, loss functions) are automatically kept in fp32, since they're more prone to rounding error.
- This casting is decided per-operation, automatically, every time that operation runs inside the block. Nothing is cast once and reused — it's a fresh, temporary cast each time.
- Gradients computed during `backward()` get accumulated back into each parameter's native fp32 dtype, even for parameters whose forward pass ran through fp16 math.

## Why GradScaler is needed

fp16 has a much smaller representable range than fp32. A gradient that's a very small number can **underflow to exactly zero** in fp16 — silently losing that gradient entirely, which can stall or break training.

`GradScaler` fixes this with a scale-and-unscale trick:

```python
scaler = torch.amp.GradScaler("cuda")   # created once, before the epoch loop

...

scaler.scale(loss).backward()   # multiply loss up (e.g. x1024) before backward,
                                  # so resulting gradients are also scaled up,
                                  # pushing tiny gradients away from zero
scaler.step(optimizer)          # divide the gradients back down to their real
                                  # size, THEN call optimizer.step() — so the
                                  # fp32 weights update by the correct amount
scaler.update()                  # adjust the scale factor for next batch:
                                  # lower it if this step overflowed to inf/NaN,
                                  # otherwise cautiously raise it over time
```

This is purely an algebra trick — since the loss is scaled by one constant factor, and then the gradients are divided by that same factor before the weight update, the actual math result is unaffected. It only exists to keep intermediate fp16 values away from underflowing to zero.

## What exactly is "the scale factor"?

`GradScaler` keeps a single internal number — the scale factor — and uses it in exactly two places:

```python
scaler.scale(loss).backward()   # multiplies the loss BY the scale factor
scaler.step(optimizer)          # divides the gradients BY that same scale factor
```

PyTorch starts this at a default of 65536, though the exact starting value doesn't matter conceptually — it's just "some number big enough to push small gradients away from zero."

**`scaler.update()` is what changes this number, every batch:**

- **If this batch's gradients overflowed** (hit `inf`/`NaN` — a sign the scale factor was too aggressive for this batch), `update()` cuts the scale factor down (halves it by default). That batch's weight update is also skipped entirely — `scaler.step()` silently detects the overflow and does not call the underlying `optimizer.step()` for that batch.
- **If things have been stable for a while** (no overflow for a number of consecutive batches — 2000 by default), `update()` cautiously multiplies the scale factor back up, to keep pushing gradients further from the underflow danger zone.

So over the course of training, this number isn't fixed — it drifts up when things are stable and drops sharply if it ever overshoots. You never set it yourself, and you'd never normally need to look at it, but it's inspectable via `scaler.get_scale()` if you were curious.

**It's one single number, not two separate scales for overflow vs. underflow.** The two directions use different logic because only overflow is actually detectable:
- **Down** = reactive, based on real evidence — `inf`/`NaN` was detected, so the scale was proven too big.
- **Up** = proactive, blind guess — underflow can never be detected after the fact (a gradient that rounded to `0.0` looks identical to one that was supposed to be `0.0`), so `update()` just periodically risks raising the scale after a long stable streak, with no actual evidence underflow was happening.

### Why underflow specifically can't be detected

Say a gradient is mathematically supposed to be `0.0000001`. fp16's smallest nonzero value is roughly `0.00006`, so that gradient rounds down to `0.0`. Now say a *different* gradient is mathematically supposed to be exactly `0.0` — that one is also `0.0`. After the fact, both look identical: an ordinary `0.0`. There's no leftover signal telling you "this one used to be something else." Compare that to overflow, which produces a distinctive, recognizable `inf`/`NaN` — visible, checkable evidence.

So `GradScaler` can't ask "did underflow happen?" — it asks the only answerable question instead: "has this scale factor caused any *overflow* recently?" The reasoning:
1. A bigger scale pushes small gradients away from the "rounds to zero" zone — reduces underflow risk.
2. Too big, and gradients get pushed past fp16's upper limit instead — that's overflow, the opposite failure mode.
3. There's a sweet spot in between, and `GradScaler` doesn't know exactly where it is.
4. So it probes for it: no overflow in 2,000 straight batches → assume it's being too conservative → try a bigger scale.
5. If that guess overshoots, it eventually shows up as overflow on some future batch, and the "cut it down" rule (which has real evidence) corrects it right away.

It's a trial-and-error feedback loop: probe upward blindly and periodically, and use the one failure mode you *can* detect (overflow) as the only signal for when you've gone too far.

## The four-step pattern, per batch

```python
optimizer.zero_grad()

with torch.amp.autocast("cuda"):
    output = model(test_input)
    loss = loss_function(output, test_output.to(device))

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

1. **`autocast` block** — forward pass + loss computed, with safe ops running in fp16.
2. **`scaler.scale(loss).backward()`** — scale the loss up, then backpropagate (computes scaled gradients).
3. **`scaler.step(optimizer)`** — unscale the gradients back to their true size, then update the fp32 weights.
4. **`scaler.update()`** — adjust the scale factor based on whether this step was numerically stable.

## Why ordinary fp16 rounding (in autocast) doesn't get a scaling fix, but underflow (in GradScaler) does

fp16 loses precision two different ways, and only one of them can be fixed by scaling:

- **Ordinary rounding error** — fp16 only stores ~3 decimal digits of precision (10 mantissa bits vs. fp32's 23). Every value computed in fp16 gets rounded to the nearest representable fp16 number. This error is *relative* — roughly the same tiny percentage off, whether the true value is `2.3` or `230,000`.
- **Underflow** — a value smaller than fp16's minimum (~`0.00006`) doesn't just round imprecisely, it collapses to exactly `0.0`. That's a total, absolute loss of the value, not a small percentage error.

**Scaling can fix underflow, but literally cannot fix ordinary rounding — here's the arithmetic showing why:**

Say a value's true magnitude is `x`, and fp16 rounding introduces a relative error of about 0.05%, so the value you actually get is `x × 1.0005` instead of `x`.

Now multiply by a scale factor `S` before the operation, and divide by the same `S` after:
```
rounded(x × S) / S  ≈  (x × S × 1.0005) / S  =  x × 1.0005
```
The `S` cancels out completely — you're left with the exact same 0.0005 (0.05%) relative error you started with. Scaling a value up and back down doesn't change *how many correct digits* fp16 can store; it just shifts where those digits sit numerically. The relative error rides along unchanged either way.

**Underflow is different because it isn't a "percentage" error at all — it's a cliff.** A value like `0.0000001` doesn't round to "a slightly wrong small number," it rounds to `0.0`, a complete loss of information. Scaling *does* help here: multiply `0.0000001` by `1024` first, and you get `0.0001024` — now comfortably above fp16's `~0.00006` floor, so it survives as a real (if still slightly imprecise) number instead of vanishing. Divide back down afterward, and you recover something close to the original value — instead of recovering nothing at all.

**So the reason `autocast` has no scaling mechanism isn't an oversight** — scaling literally cannot reduce relative rounding error (the math cancels out, as shown above). It only rescues values from the underflow cliff. Since ordinary forward-pass activations (kept in a stable range by things like `LayerNorm`) rarely approach that cliff, while gradients — shrunk by repeated chain-rule multiplication across 24 layers — regularly do, scaling is specifically useful for `GradScaler`'s job and simply irrelevant to `autocast`'s.

## Where GradScaler is NOT needed

`GradScaler` is only relevant when there's a `backward()` pass. During evaluation (`predict_in_batches`), we still use `autocast` for the same fp16 speed/memory benefit, but skip `GradScaler` entirely, since there are no gradients to scale:

```python
with torch.amp.autocast("cuda"):
    for batch_texts, batch_labels in loader:
        all_scores.append(model(batch_texts))
        all_labels.append(batch_labels)
```
