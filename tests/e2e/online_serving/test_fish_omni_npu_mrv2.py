# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Drama3 DAC MRV2 qualification on two Ascend cards, with a V1 control.

Run with real weights at advanced_model level. HTTP chunks are transport
fragments, not codec windows; assertions use the complete PCM sample count.
"""

import io
import os
import time
from concurrent.futures import ThreadPoolExecutor

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import numpy as np
import pytest
import requests
import soundfile as sf
from prometheus_client.parser import text_string_to_metric_families

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

MODEL = os.environ.get("FISH_OMNI_NPU_MODEL", "fishaudio/drama-3")
SAMPLE_RATE = 44100
HOP = 2048
TIMEOUT = 900
SHORT_TEXT = "The quarterly report is due on Thursday, so please finish it by noon."
LONG_TEXT = " ".join(
    [
        "The weather is pleasant today, and the warm sunlight makes the garden feel peaceful. "
        "We will take a walk after breakfast and talk about our plans for the coming week."
    ]
    * 24
)
SERVER_PARAMS = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=get_deploy_config_path(filename),
            # Keep metrics enabled to check outstanding pipeline requests.
            server_args=[],
            stage_init_timeout=1200,
        ),
        id=runner,
    )
    for runner, filename in (
        ("dac-v1", "fish_omni.yaml"),
        ("dac-v2", "fish_omni_npu_mrv2.yaml"),
    )
]


def _payload(model: str, text: str, *, stream: bool) -> dict:
    return {
        "model": model,
        "input": text,
        "seed": 42,
        "max_new_tokens": 320,
        "response_format": "pcm" if stream else "wav",
        "stream": stream,
        **({"stream_format": "audio"} if stream else {}),
    }


def _assert_samples(samples: np.ndarray) -> None:
    assert samples.ndim == 1 and samples.size >= HOP
    assert samples.size % HOP == 0, "DAC output ended on a partial codec frame"
    assert np.isfinite(samples).all()
    assert np.max(np.abs(samples)) > 0, "DAC returned only silence"


def _wait_for_idle(base_url: str, timeout: float = 120) -> None:
    names = ("vllm_omni:num_requests_running", "vllm_omni:num_requests_waiting")
    latest: dict[str, list[float]] = {name: [] for name in names}
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = requests.get(f"{base_url}/metrics", timeout=10)
        response.raise_for_status()
        latest = {name: [] for name in names}
        for family in text_string_to_metric_families(response.text):
            for sample in family.samples:
                if sample.name in latest:
                    latest[sample.name].append(sample.value)
        if all(values and all(value == 0 for value in values) for values in latest.values()):
            return
        time.sleep(1)
    missing = [name for name, values in latest.items() if not values]
    assert not missing, f"Cannot verify request cleanup: missing metrics {missing}"
    pytest.fail(f"Requests did not drain after cancellation/completion: {latest}")


@pytest.mark.advanced_model
@pytest.mark.tts
@hardware_test(res={"npu": "A3"}, num_cards=2)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
def test_fish_dac_full_and_streaming_audio(omni_server):
    base_url = f"http://{omni_server.host}:{omni_server.port}"
    url = f"{base_url}/v1/audio/speech"
    with requests.post(url, json=_payload(omni_server.model, SHORT_TEXT, stream=False), timeout=TIMEOUT) as response:
        response.raise_for_status()
        samples, rate = sf.read(io.BytesIO(response.content), dtype="float32")
    assert rate == SAMPLE_RATE
    _assert_samples(samples)

    with requests.post(
        url, json=_payload(omni_server.model, LONG_TEXT, stream=True), stream=True, timeout=TIMEOUT
    ) as response:
        response.raise_for_status()
        pcm = b"".join(response.iter_content(chunk_size=None))
    assert len(pcm) % 2 == 0
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
    _assert_samples(samples)
    # Six ramp windows total 98 frames; cross the first steady 64-frame window.
    assert samples.size > (98 + 64) * HOP, "Long request did not exercise the steady DAC window"
    _wait_for_idle(base_url)


@pytest.mark.advanced_model
@pytest.mark.tts
@hardware_test(res={"npu": "A3"}, num_cards=2)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
def test_fish_dac_concurrent_abort_releases_requests(omni_server):
    base_url = f"http://{omni_server.host}:{omni_server.port}"
    url = f"{base_url}/v1/audio/speech"

    def synthesize(index: int) -> bytes:
        text = LONG_TEXT if index < 2 else SHORT_TEXT
        with requests.post(
            url, json=_payload(omni_server.model, text, stream=True), stream=True, timeout=TIMEOUT
        ) as response:
            response.raise_for_status()
            chunks = response.iter_content(chunk_size=None)
            if index < 2:
                # Abort one stream after its first bytes and one after later audio.
                target = 1 if index == 0 else 14 * HOP * 2
                received = 0
                for chunk in chunks:
                    received += len(chunk)
                    if received >= target:
                        return b""  # context manager closes the active stream
                pytest.fail("Stream ended before the requested cancellation point")
            return b"".join(chunks)

    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(synthesize, range(4)))
    for pcm in outputs[2:]:
        assert len(pcm) % 2 == 0
        _assert_samples(np.frombuffer(pcm, dtype="<i2").astype(np.float32))
    _wait_for_idle(base_url)

    # Repeat after cleanup to catch retained terminal state and stale windows.
    for _ in range(2):
        with requests.post(
            url, json=_payload(omni_server.model, SHORT_TEXT, stream=False), timeout=TIMEOUT
        ) as response:
            response.raise_for_status()
            samples, rate = sf.read(io.BytesIO(response.content), dtype="float32")
        assert rate == SAMPLE_RATE
        _assert_samples(samples)
    _wait_for_idle(base_url)
