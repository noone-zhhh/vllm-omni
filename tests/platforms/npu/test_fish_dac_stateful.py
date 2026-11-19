# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise eager Fish DAC history with a small instance of the production codec."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tests.helpers.mark import hardware_marks

pytestmark = [pytest.mark.core_model]


def _codec():
    from vllm_omni.model_executor.models.fish_speech.dac_modules import (
        DAC,
        DownsampleResidualVectorQuantize,
        ModelArgs,
        WindowLimitedTransformer,
    )

    args = ModelArgs(dim=16, n_head=2, head_dim=8, n_layer=1, intermediate_size=32, block_size=64)
    transformer = WindowLimitedTransformer(args, input_dim=16, window_size=4)
    quantizer = DownsampleResidualVectorQuantize(
        input_dim=16, n_codebooks=1, codebook_size=16, semantic_codebook_size=16, post_module=transformer
    )
    return DAC(encoder_dim=8, encoder_rates=[2], decoder_dim=16, decoder_rates=[2], quantizer=quantizer)


def _check_streaming(device, dtype, tmp_path, monkeypatch):
    from vllm_omni.model_executor.models.fish_speech import fish_speech_dac_decoder as dac
    from vllm_omni.platforms.npu.models import fish_dac

    torch.manual_seed(17)
    codec = _codec()
    for layer in codec.quantizer.post_module.layers:
        layer.attention_layer_scale.gamma.data.fill_(1)
        layer.ffn_layer_scale.gamma.data.fill_(1)
    torch.save(codec.state_dict(), tmp_path / "codec.pth")
    monkeypatch.setattr(dac, "build_dac_codec", _codec)
    if device == "cpu":
        monkeypatch.setattr(dac, "current_omni_platform", SimpleNamespace(is_npu=lambda: True, is_cuda=lambda: False))
        monkeypatch.setattr(fish_dac, "supports_stateful_dac", lambda: True)
    extra = {"fish_speech_dac_stateful": True, "fish_speech_dac_dtype": dtype}
    decoder = dac.FishSpeechDACDecoder(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(model=str(tmp_path), stage_connector_config={"extra": extra}),
            device_config=SimpleNamespace(device=device),
            scheduler_config=SimpleNamespace(max_num_seqs=2),
        )
    )
    decoder._num_codebooks = 2
    decoder._ensure_codec_loaded()
    assert decoder._eager_stream is not None
    decode_window = Mock(wraps=decoder._eager_stream.decode_window)
    monkeypatch.setattr(decoder._eager_stream, "decode_window", decode_window)
    codes = torch.randint(16, (2, 2, 9)).to(device)
    expected = decoder._codec.decode(codes, torch.tensor([9, 9], device=device))[0].float()
    hop = decoder._codec.frame_length

    def run(names, windows, sequence, contexts, finished=False):
        return decoder(
            torch.cat([w.flatten() for w in windows]),
            request_ids=names,
            seq_token_counts=[w.numel() for w in windows],
            runtime_additional_information=[
                {"meta": {"chunk_seq": sequence, "left_context_size": c, "finished": finished}} for c in contexts
            ],
        ).multimodal_outputs["model_outputs"]

    first_a, first_b = run(["a", "b"], [codes[0, :, :3], codes[1, :, :5]], 0, [0, 0])
    # Context-only windows produce no audio and consume no new position.
    assert run(["a"], [codes[0, :, :2]], 1, [2])[0].numel() == 0
    with pytest.raises(ValueError, match="left context"):
        run(["b"], [codes[1, :, :2]], 1, [3])
    assert decoder._stream_positions == {"a": 3, "b": 5}
    # No slot: decode the complete window and trim its overlap.
    fallback = run(["other"], [codes[0]], 1, [2])[0]
    fallback_ref = decoder._codec.decode(codes[:1], torch.tensor([9], device=device))[0]
    torch.testing.assert_close(fallback, fallback_ref[0, 0, 2 * hop :].float())
    last_b, last_a = run(["b", "a"], [codes[1, :, 3:], codes[0, :, 1:]], 1, [2, 2], finished=True)
    assert [call.args[0].shape[-1] for call in decode_window.call_args_list] == [5, 6]
    assert not decoder._stream_slots
    # A new request reuses a dirty slot. Its first frame must see zero history.
    restarted, _ = run(["new", "another"], [codes[0, :, :3], codes[1, :, :5]], 0, [0, 0])
    decoder.on_requests_finished({"new", "another"})
    assert not decoder._stream_slots
    torch.testing.assert_close(restarted, first_a, atol=1e-4, rtol=1e-3)
    actual = torch.stack([torch.cat([first_a, last_a]), torch.cat([first_b, last_b])])
    reference = expected[:, 0]
    assert actual.shape == reference.shape
    snr = 10 * torch.log10(reference.square().mean() / (actual - reference).square().mean().clamp_min(1e-20))
    assert snr.item() > 40

    # A failed batch has already claimed slots and may have written layer state.
    decode_window.side_effect = RuntimeError("failed decode")
    with pytest.raises(RuntimeError, match="failed decode"):
        run(["a", "b"], [codes[0], codes[1]], 0, [0, 0])
    assert not decoder._stream_slots and not decoder._stream_positions
    assert len(set(decoder._free_stream_slots)) == 2


@pytest.mark.cpu
@torch.inference_mode()
@pytest.mark.parametrize("dtype", ["fp32", "fp16"])
def test_eager_streaming_reference(dtype, tmp_path, monkeypatch):
    _check_streaming("cpu", dtype, tmp_path, monkeypatch)


@pytest.mark.parametrize(
    "device",
    [
        pytest.param("npu", marks=hardware_marks(res={"npu": "A2"}, num_cards=1), id="A2"),
        pytest.param("npu", marks=hardware_marks(res={"npu": "A3"}, num_cards=1), id="A3"),
    ],
)
@torch.inference_mode()
def test_eager_streaming_npu(device, tmp_path, monkeypatch):
    pytest.importorskip("torch_npu")
    _check_streaming(device, "fp16", tmp_path, monkeypatch)


@pytest.mark.cpu
@pytest.mark.parametrize(("soc", "stateful"), [("A2", True), ("A3", True), ("A2", False), ("A5", True), ("310P", True)])
def test_stateful_device_selection(soc, stateful, tmp_path, monkeypatch):
    from vllm_omni.model_executor.models.fish_speech import fish_speech_dac_decoder as dac

    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.utils",
        SimpleNamespace(AscendDeviceType=SimpleNamespace(A2="A2", A3="A3"), get_ascend_device_type=lambda: soc),
    )
    monkeypatch.setattr(dac, "current_omni_platform", SimpleNamespace(is_npu=lambda: True, is_cuda=lambda: False))
    monkeypatch.setattr(dac, "build_dac_codec", _codec)
    torch.save(_codec().state_dict(), tmp_path / "codec.pth")
    # S2 Pro omits both flags; Drama-3 supplies both even on unsupported NPUs.
    extra = {"fish_speech_dac_stateful": True, "fish_speech_dac_cuda_graphs": True} if stateful else {}
    decoder = dac.FishSpeechDACDecoder(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(model=str(tmp_path), stage_connector_config={"extra": extra}),
            device_config=SimpleNamespace(device="cpu"),
            scheduler_config=SimpleNamespace(max_num_seqs=1),
        )
    )
    decoder._ensure_codec_loaded()
    enabled = stateful and soc in {"A2", "A3"}
    assert decoder._codec is not None and not decoder._cuda_graphs
    assert (decoder._eager_stream is not None) == enabled
    assert decoder._stateful == enabled
