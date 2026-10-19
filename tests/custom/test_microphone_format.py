"""Microphone bytes must use the format advertised to the EVI session."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

import hume.empathic_voice.chat.audio.microphone as microphone_module
from hume.empathic_voice.chat.audio.microphone import Microphone


@pytest.mark.asyncio
@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("sample_rate", [32000, 48000])
async def test_microphone_explicitly_selects_its_advertised_format(monkeypatch, channels, sample_rate):
    # sounddevice's global defaults can differ from the selected device's
    # capabilities. Omitting these arguments makes it use those globals.
    defaults = SimpleNamespace(device=(7, 8), channels=1, samplerate=16000)
    sounddevice = SimpleNamespace(
        default=defaults,
        query_devices=lambda **kwargs: {"max_input_channels": channels, "default_samplerate": sample_rate},
    )
    opened = {}

    @contextmanager
    def input_stream(**kwargs):
        opened["channels"] = kwargs.get("channels", defaults.channels)
        opened["samplerate"] = kwargs.get("samplerate", defaults.samplerate)
        opened["device"] = kwargs["device"]
        opened["dtype"] = kwargs["dtype"]
        yield

    monkeypatch.setattr(microphone_module, "_FAILED_IMPORTS", [])
    monkeypatch.setattr(microphone_module, "sounddevice", sounddevice, raising=False)
    monkeypatch.setattr(microphone_module, "RawInputStream", input_stream, raising=False)
    with Microphone.context() as microphone:
        assert opened["channels"] == microphone.num_channels
        assert opened["samplerate"] == microphone.sample_rate
        assert opened["device"] == 7
        assert opened["dtype"] == "int16"
