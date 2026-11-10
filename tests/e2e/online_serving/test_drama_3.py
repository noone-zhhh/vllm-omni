# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""E2E online tests for Fish Audio Drama 3 via /v1/audio/speech.

Real model inference (no mocks): text-only synthesis, streaming, voice cloning
and concurrent requests, plus NPU DAC window and cancellation checks.
The reference audio is inlined as a base64 data URL
because CI hosts cannot fetch external URLs.
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
from tests.helpers.media import get_asset_path
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.platforms import current_omni_platform

MODEL = "fishaudio/drama-3"
DEFAULT_AUDIO_SPEECH_TIMEOUT_S = 300.0
# The DAC codec decodes at 44.1 kHz; raw PCM carries no header.
SAMPLE_RATE = 44100
MAX_CONCURRENT = 4
HOP = 2048
DAC_REQUEST_TIMEOUT_S = 900
LONG_TEXT = " ".join(
    [
        "The weather is pleasant today, and the warm sunlight makes the garden feel peaceful. "
        "We will take a walk after breakfast and talk about our plans for the coming week."
    ]
    * 24
)

# ~0.5 s of 44.1 kHz mono PCM_16 WAV.
_MIN_AUDIO_BYTES = 40_000

REF_AUDIO_URL = get_asset_path("qwen3_tts/clone_2.wav", as_data_url=True)
REF_TEXT = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."


def get_prompt(prompt_type: str = "en") -> str:
    prompts = {
        "en": "The weather is nice today, perfect for a walk in the park.",
        "clone": "The quarterly report is due on Thursday, so please finish it by noon.",
    }
    return prompts.get(prompt_type, prompts["en"])


tts_server_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=get_deploy_config_path("fish_omni.yaml"),
            # No --trust-remote-code: vllm-omni owns the `fish_omni` config.
            # NPU request-cleanup assertions need metrics; keep other platforms unchanged.
            server_args=[] if current_omni_platform.is_npu() else ["--disable-log-stats"],
            stage_init_timeout=900,
        ),
        id="drama_3",
    )
]


@pytest.mark.slow
@pytest.mark.tts
@hardware_test(
    res={"cuda": "H100", "npu": "A3", "rocm": "MI325"},
    num_cards={"cuda": 1, "npu": 2, "rocm": 1},
)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_text_to_audio_001(omni_server, openai_client) -> None:
    """Baseline smoke: default deploy, non-streaming WAV.

    Deploy Setting: fish_omni.yaml
    Input Modal: text
    Output Modal: audio
    Input Setting: stream=False
    """
    request_config = {
        "model": omni_server.model,
        "input": get_prompt(),
        "stream": False,
        "timeout": DEFAULT_AUDIO_SPEECH_TIMEOUT_S,
        "response_format": "wav",
        "min_audio_bytes": _MIN_AUDIO_BYTES,
    }
    openai_client.send_audio_speech_request(request_config)


@pytest.mark.slow
@pytest.mark.tts
@hardware_test(
    res={"cuda": "H100", "npu": "A3", "rocm": "MI325"},
    num_cards={"cuda": 1, "npu": 2, "rocm": 1},
)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
@pytest.mark.parametrize("seed", [0, 42], ids=["seed0", "seed42"])
def test_voice_clone_streaming_002(omni_server, openai_client, seed: int) -> None:
    """Voice cloning over the async_chunk streaming path.

    Deploy Setting: fish_omni.yaml (async_chunk: true)
    Input Modal: text + ref_audio + ref_text
    Input Setting: stream=True, response_format=pcm, seed=0 or 42
    """
    # Pin stochastic synthesis so the HNR assertion checks a reproducible clip.
    request_config = {
        "model": omni_server.model,
        "input": get_prompt("clone"),
        "ref_audio": REF_AUDIO_URL,
        "ref_text": REF_TEXT,
        "seed": seed,
        "stream": True,
        "stream_format": "audio",
        "timeout": DEFAULT_AUDIO_SPEECH_TIMEOUT_S,
        "response_format": "pcm",
        "min_audio_bytes": _MIN_AUDIO_BYTES,
        "expected_sample_rate": SAMPLE_RATE,
        "transcript_pcm_sample_rate": SAMPLE_RATE,
        # The clone_2.wav reference is itself only 1.70 dB on this metric, and the
        # cloned voice keeps its rough delivery: H100 runs have scored 2.61, 1.89
        # and 0.80 dB, with the transcript matching word for word where it was
        # checked. Keep the catastrophic-failure check and leave speech quality to
        # the transcript check.
        "min_hnr_db": -1.0,
    }
    openai_client.send_audio_speech_request(request_config)


@pytest.mark.slow
@pytest.mark.tts
@hardware_test(
    res={"cuda": "H100", "npu": "A3", "rocm": "MI325"},
    num_cards={"cuda": 1, "npu": 2, "rocm": 1},
)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_voice_clone_concurrent_003(omni_server, openai_client) -> None:
    """Concurrent voice-clone requests keep their own codebook-tower state.

    Input Modal: text + ref_audio + ref_text
    Datasets: MAX_CONCURRENT identical requests using the platform deploy config
    """
    request_config = {
        "model": omni_server.model,
        "input": get_prompt("clone"),
        "ref_audio": REF_AUDIO_URL,
        "ref_text": REF_TEXT,
        "stream": False,
        "timeout": DEFAULT_AUDIO_SPEECH_TIMEOUT_S,
        "response_format": "wav",
        "min_audio_bytes": _MIN_AUDIO_BYTES,
    }
    openai_client.send_audio_speech_request(request_config, request_num=MAX_CONCURRENT)


@pytest.mark.slow
@pytest.mark.tts
@hardware_test(
    res={"cuda": "H100", "npu": "A3", "rocm": "MI325"},
    num_cards={"cuda": 1, "npu": 2, "rocm": 1},
)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_voice_clone_requires_ref_text_004(omni_server, openai_client) -> None:
    """``ref_audio`` without ``ref_text`` is rejected."""
    openai_client.send_audio_speech_http_request(
        {
            "json": {
                "model": omni_server.model,
                "input": get_prompt(),
                "ref_audio": REF_AUDIO_URL,
                "response_format": "wav",
            },
            "timeout": DEFAULT_AUDIO_SPEECH_TIMEOUT_S,
        },
        err_code=400,
        err_message="ref_text",
    )


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


@pytest.mark.slow
@pytest.mark.tts
@hardware_test(res={"npu": "A3"}, num_cards=2)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_fish_dac_full_and_streaming_audio(omni_server, openai_client):
    base_url = f"http://{omni_server.host}:{omni_server.port}"
    url = f"{base_url}/v1/audio/speech"
    # The shared advanced-model assertion transcribes the WAV and checks its
    # spoken content against the input text, in addition to checking valid audio.
    responses = openai_client.send_audio_speech_request(
        {
            **_payload(omni_server.model, get_prompt("clone"), stream=False),
            "timeout": DAC_REQUEST_TIMEOUT_S,
            "min_audio_bytes": HOP * 2,
        }
    )
    assert len(responses) == 1 and responses[0].audio_bytes is not None
    samples, rate = sf.read(io.BytesIO(responses[0].audio_bytes), dtype="float32")
    assert rate == SAMPLE_RATE
    _assert_samples(samples)

    with requests.post(
        url, json=_payload(omni_server.model, LONG_TEXT, stream=True), stream=True, timeout=DAC_REQUEST_TIMEOUT_S
    ) as response:
        response.raise_for_status()
        pcm = b"".join(response.iter_content(chunk_size=None))
    assert len(pcm) % 2 == 0
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
    _assert_samples(samples)
    # Six ramp windows total 98 frames; cross the first steady 64-frame window.
    assert samples.size > (98 + 64) * HOP, "Long request did not exercise the steady DAC window"
    _wait_for_idle(base_url)


@pytest.mark.slow
@pytest.mark.tts
@hardware_test(res={"npu": "A3"}, num_cards=2)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_fish_dac_concurrent_abort_releases_requests(omni_server):
    base_url = f"http://{omni_server.host}:{omni_server.port}"
    url = f"{base_url}/v1/audio/speech"

    def synthesize(index: int) -> bytes:
        text = LONG_TEXT if index < 2 else get_prompt("clone")
        with requests.post(
            url, json=_payload(omni_server.model, text, stream=True), stream=True, timeout=DAC_REQUEST_TIMEOUT_S
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
            url, json=_payload(omni_server.model, get_prompt("clone"), stream=False), timeout=DAC_REQUEST_TIMEOUT_S
        ) as response:
            response.raise_for_status()
            samples, rate = sf.read(io.BytesIO(response.content), dtype="float32")
        assert rate == SAMPLE_RATE
        _assert_samples(samples)
    _wait_for_idle(base_url)
