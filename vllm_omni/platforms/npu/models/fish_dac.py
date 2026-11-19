# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stateful eager Fish DAC decoding on Ascend."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from vllm_omni.model_executor.models.fish_speech.dac_channels_last import ChannelsLastDACDecoder, DACStreamState
from vllm_omni.model_executor.models.fish_speech.dac_modules.codec import DAC
from vllm_omni.model_executor.models.fish_speech.dac_modules.layers import Snake1d

if TYPE_CHECKING:
    from vllm_omni.model_executor.models.fish_speech.fish_speech_dac_decoder import FishSpeechDACDecoder


def supports_stateful_dac() -> bool:
    from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

    return get_ascend_device_type() in (AscendDeviceType.A2, AscendDeviceType.A3)


class _EagerStreamState(DACStreamState):
    def fill(self, key: int, padded: torch.Tensor, pad: int) -> None:
        history = self.history(key, pad, padded).index_select(0, self.slots)
        padded[:, :pad] = history.masked_fill(self.fresh[:, None, None].bool(), 0)

    def save(self, key: int, padded: torch.Tensor, pad: int, length: int) -> None:
        # In the upsampled layers each codec frame spans length // span samples.
        # Select the history ending at each row's last real sample, before padding.
        ends = self.frames * (length // self.span)
        indices = ends[:, None] + torch.arange(pad, device=padded.device)
        tail = padded.gather(1, indices[:, :, None].expand(-1, -1, padded.size(2)))
        self.tensors[key].index_copy_(0, self.slots, tail)


class NPUStatefulDACDecoder(ChannelsLastDACDecoder):
    def __init__(self, codec: DAC, num_slots: int) -> None:
        super().__init__(codec)
        self.stream = _EagerStreamState(num_slots)

    @torch.inference_mode()
    def decode_window(self, codes: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        self.stream.bind(rows, codes.shape[-1])
        try:
            return self(codes)
        finally:
            self.stream.unbind()

    @torch.inference_mode()
    def decode_batch(
        self,
        stage: FishSpeechDACDecoder,
        codes: list[torch.Tensor],
        indices: list[int],
        parsed_contexts: list[int],
        infos: list[dict[str, Any]] | None,
        request_ids: list[str],
        group_windows: Callable[[list[tuple[int, torch.Tensor]]], Iterable[list[tuple[int, torch.Tensor]]]],
    ) -> list[torch.Tensor]:
        contexts = [int(parsed_contexts[i]) for i in indices]
        if any(c < 0 or c > window.shape[-1] for c, window in zip(contexts, codes, strict=True)):
            raise ValueError("DAC left context must fit within its codec window")
        lengths = [window.shape[-1] - c for window, c in zip(codes, contexts, strict=True)]
        owners = [request_ids[i] if i < len(request_ids) else None for i in indices]
        slots, fresh, positions, release = stage._claim_stream_slots(
            owners,
            [stage._meta_first_window(infos, i, c) for i, c in zip(indices, contexts, strict=True)],
            [stage._meta_finished(infos, i) for i in indices],
            lengths,
            self.transformer.freqs_cis.shape[0],
        )
        audio = [torch.empty(0, dtype=torch.float32)] * len(codes)
        cached, uncached = [], []
        for j, window in enumerate(codes):
            if lengths[j] == 0:
                owner = owners[j]
                if fresh[j] and owner is not None:
                    release.append(owner)
            elif slots[j] is None:
                uncached.append((j, window))
            else:
                cached.append((j, window[:, contexts[j] :]))

        try:
            for use_history, windows in ((True, cached), (False, uncached)):
                if not windows:
                    continue
                for group in group_windows(windows):
                    members = [j for j, _ in group]
                    frames = max(window.shape[-1] for _, window in group)
                    batch = torch.stack(
                        [
                            F.pad(window, (0, frames - window.shape[-1])) if window.shape[-1] < frames else window
                            for _, window in group
                        ]
                    )
                    with torch.amp.autocast(batch.device.type, enabled=False):
                        if use_history:
                            rows = torch.tensor(
                                [[slots[j], fresh[j], lengths[j], positions[j]] for j in members],
                                dtype=torch.long,
                                device=batch.device,
                            ).T.contiguous()
                            wav = self.decode_window(batch, rows)
                        else:
                            counts = torch.tensor([window.shape[-1] for _, window in group], device=batch.device)
                            wav, _ = stage._decode_codes(batch, counts)
                    wav = wav.float()
                    hop = wav.shape[-1] // frames
                    for row, j in enumerate(members):
                        start = 0 if use_history else contexts[j]
                        audio[j] = wav[row, 0, start * hop : (start + lengths[j]) * hop]
        except Exception:
            # A layer may already have written history or advanced a request's position.
            stage.on_requests_finished({owner for owner in owners if owner is not None})
            raise
        stage.on_requests_finished(release)
        return audio

    def _snake(
        self,
        module: Snake1d,
        hidden: torch.Tensor,
        bias: torch.Tensor | None = None,
        pad: tuple[int, int] = (0, 0),
        history_key: int | None = None,
    ) -> torch.Tensor:
        x = hidden.float()
        if bias is not None:
            x = x + bias
        alpha, inv_alpha = self.snake[id(module)]
        sine_squared = (alpha * x).sin_().square_()
        out = torch.addcmul(x, inv_alpha, sine_squared).to(hidden.dtype)
        if pad != (0, 0):
            out = F.pad(out, (0, 0, *pad))
        if history_key is not None and pad[0]:
            self.stream.fill(history_key, out, pad[0])
        return out
