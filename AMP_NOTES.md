# Mixed Precision Training (AMP) Notes

Notes on `torch.amp.autocast` and `torch.amp.GradScaler`, and why they're used in `main.py`'s `train()` function.

## Why we added this

Full fine-tuning of BERT-large (all ~340M parameters trainable, not just a small classifier head) needs to store activations across all 24 layers for every example in a batch, in order to compute gradients during `backward()`. This uses a lot of GPU memory — enough that even `batch_size=4` was running the GPU at ~97% memory capacity (7930MiB / 8188MiB), causing the training process to run extremely slowly (30+ hours without finishing even one confirmed epoch), likely due to memory fragmentation/thrashing near the memory ceiling.

Mixed precision training runs the expensive matrix math in fp16 (16-bit floats) instead of fp32 (32-bit floats), roughly halving memory use for those operations and often speeding up compute on GPUs with Tensor Cores (which the RTX 4060 has).

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

## Where GradScaler is NOT needed

`GradScaler` is only relevant when there's a `backward()` pass. During evaluation (`predict_in_batches`), we still use `autocast` for the same fp16 speed/memory benefit, but skip `GradScaler` entirely, since there are no gradients to scale:

```python
with torch.amp.autocast("cuda"):
    for batch_texts, batch_labels in loader:
        all_scores.append(model(batch_texts))
        all_labels.append(batch_labels)
```
