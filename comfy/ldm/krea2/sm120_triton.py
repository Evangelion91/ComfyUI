"""Krea modulation with an FP16 rounding boundary between multiply and add."""

import triton
import triton.language as tl


@triton.jit
def mul_add(x, a, b, out, size: tl.constexpr, channels: tl.constexpr,
            residual: tl.constexpr, block: tl.constexpr):
    i = tl.program_id(0) * block + tl.arange(0, block)
    channel = i % channels
    valid = i < size
    x_value = tl.load(x + i, valid, 0).to(tl.float32)
    multiplier = tl.load(a + channel).to(tl.float32)
    if residual:
        transformed = tl.load(b + i, valid, 0).to(tl.float32)
        values = (multiplier * transformed).to(tl.float16).to(tl.float32) + x_value
    else:
        shift = tl.load(b + channel).to(tl.float32)
        values = (x_value * multiplier).to(tl.float16).to(tl.float32) + shift
    tl.store(out + i, values, valid)
