from types import SimpleNamespace

import pytest
import torch

import comfy.model_base
from comfy.ldm.krea2 import sm120


@pytest.fixture
def square_profile(monkeypatch):
    class Krea2:
        current_weight_patches_uuid = "loaded"
        diffusion_model = SimpleNamespace(blocks=[
            SimpleNamespace(attn=SimpleNamespace(wq=SimpleNamespace(quant_format="float8_e4m3fn")))
        ])

        def get_dtype_inference(self):
            return torch.float16

    patcher = SimpleNamespace(
        model=Krea2(), patches_uuid="loaded", loaded_size=lambda: 1, hook_patches={},
        object_patches={}, injections={}, wrappers={}, callbacks={},
    )
    sampler = SimpleNamespace(
        model=patcher, device="cuda:0", sampler="euler", scheduler="simple", steps=8,
        denoise=1.0, model_options={"transformer_options": {}},
    )
    monkeypatch.setattr(comfy.model_base, "Krea2", Krea2)
    monkeypatch.setattr(sm120, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(sm120, "torch", SimpleNamespace(
        device=torch.device, float16=torch.float16, is_tensor=torch.is_tensor,
        is_inference_mode_enabled=lambda: True,
        cuda=SimpleNamespace(get_device_capability=lambda _: (12, 0),
                             get_device_name=lambda _: "NVIDIA GeForce RTX 5090"),
        backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_fp16_accumulation=True))),
    ))
    monkeypatch.setattr(sm120.args, "use_sage_attention", True)
    noise = torch.empty((1, 16, 1, 256, 256), device="meta")
    positive = [[torch.empty((1, 478, 30720), device="meta"), {}]]
    negative = [[torch.empty_like(positive[0][0]), {}]]
    return sampler, noise, positive, negative, 1, torch.empty(9), None


def test_admits_verified_square_profile(square_profile):
    assert sm120.eligible(*square_profile)


@pytest.mark.parametrize("key", ["optimized_attention_override", "patches_replace"])
def test_standard_attention_extensions_keep_the_stock_path(square_profile, key):
    square_profile[0].model_options["transformer_options"][key] = object()
    assert not sm120.eligible(*square_profile)


def test_alternate_condition_batcher_keeps_the_stock_path(square_profile):
    square_profile[0].model_options["sampler_calc_cond_batch_function"] = object()
    assert not sm120.eligible(*square_profile)


def test_negative_condition_hook_keeps_the_stock_path_at_cfg_one(square_profile):
    square_profile[3][0][1]["hooks"] = object()
    assert not sm120.eligible(*square_profile)
