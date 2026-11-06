# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real Fish codec parity at the NPU V1/V2 runner output boundary.

The engine/input path is covered by the online test. Here fixed codes remove
AR sampling differences, while real sample_tokens methods exercise request
partitioning and device-to-host output handling on NPU.
"""

import os
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

pytest.importorskip("vllm_ascend")

from tests.helpers.mark import hardware_marks
from tests.platforms.npu.test_npu_generation_output_copy import make_runner as make_v1_runner
from vllm_omni.model_executor.models.fish_speech.fish_speech_dac_decoder import FishSpeechDACDecoder
from vllm_omni.platforms.npu.worker.npu_omni_v2_model_runner import NPUOmniGenerationModelRunner

pytestmark = [pytest.mark.advanced_model, pytest.mark.omni, *hardware_marks(res={"npu": "A3"}, num_cards=1)]


def _make_v2_runner(output, num_reqs):
    runner = object.__new__(NPUOmniGenerationModelRunner)
    runner.device = torch.device("npu:0")
    runner._gen_model_output = output
    runner._gen_input_batch = SimpleNamespace(
        num_reqs=num_reqs,
        req_ids=[f"req-{i}" for i in range(num_reqs)],
        idx_mapping_np=np.arange(num_reqs),
    )
    runner.execute_model_state = SimpleNamespace(finished_req_ids=set(), ec_connector_output=None)
    runner.aux_output_connector = None
    runner.req_states = SimpleNamespace(
        prompt_len=SimpleNamespace(np=np.ones(num_reqs, dtype=np.int64)),
        num_computed_tokens=Mock(),
    )
    runner.kv_connector = SimpleNamespace(post_forward=lambda _: None)
    runner._pending_generation_req_ids = lambda _: set()
    runner._reserve_native_data_plane_outputs = lambda _: None
    runner._retains_state_across_chunks = False
    runner._release_generation_slots = Mock()
    runner._finalize_native_data_plane_output = lambda result: result
    return runner


@torch.inference_mode()
def test_fixed_codes_match_v1_after_next_decode():
    device = torch.device("npu:0")
    # Only model and connector fields consumed by the codec are needed; this
    # test deliberately avoids loading the AR backbone or creating an engine.
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            model=os.environ.get("FISH_OMNI_NPU_MODEL", "fishaudio/drama-3"),
            stage_connector_config={"extra": {"fish_speech_dac_dtype": "fp16"}},
        ),
        device_config=SimpleNamespace(device=device),
    )
    decoder = FishSpeechDACDecoder(vllm_config=config)
    decoder.capture_auxiliary_graphs()
    assert decoder._codec is not None and decoder._graphs is None
    assert decoder.owns_generation_output_storage

    contexts = [0, 6, 32, 32]
    new_frames = [6, 8, 64, 13]
    generator = torch.Generator().manual_seed(42)
    infos = [
        {
            "codes": {"audio": torch.randint(0, 4096, (10, context + count), generator=generator)},
            "meta": {"left_context_size": context},
        }
        for context, count in zip(contexts, new_frames, strict=True)
    ]
    num_reqs = len(infos)

    def decode(information):
        return decoder.forward(
            input_ids=torch.zeros(num_reqs, device=device, dtype=torch.long),
            seq_token_counts=[1] * num_reqs,
            runtime_additional_information=information,
        )

    v1 = make_v1_runner(decode(infos).multimodal_outputs, num_reqs)
    v2 = _make_v2_runner(decode(infos), num_reqs)
    outputs = [v1.sample_tokens(None), v2.sample_tokens(None)]
    # Complete another differently ordered batch before consuming deferred
    # output. Fish owns fresh waveforms, so the previous payload must survive.
    decode(list(reversed(infos)))
    for index, result in enumerate(outputs):
        if hasattr(result, "get_output"):
            outputs[index] = result.get_output()
    baseline, candidate = outputs
    assert baseline.req_ids == candidate.req_ids == [f"req-{i}" for i in range(num_reqs)]
    assert candidate.pooler_output is None  # audio must not be published twice
    assert candidate.sampled_token_ids == [[] for _ in range(num_reqs)]
    for index, count in enumerate(new_frames):
        expected = baseline.multimodal_outputs[index]["model_outputs"]
        actual = candidate.multimodal_outputs[index]["model_outputs"]
        assert actual.device.type == "cpu" and actual.dtype == torch.float32
        assert actual.shape == expected.shape == (count * 2048,)
        assert torch.isfinite(actual).all()
        assert int(candidate.multimodal_outputs[index]["sr"]) == 44100
        # Same codes, module, dtype, and decode batch: runner selection must
        # not alter samples. A failure requires investigation, not relaxed tolerance.
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
