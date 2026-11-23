"""Shared fixtures for tool tests."""

from __future__ import annotations

from typing import Callable

import pytest


@pytest.fixture
def video_hardware(monkeypatch) -> Callable[[bool], None]:
    """Get a setter that pins the render profile's hardware video switch.

    Call the setter with False in a test that compares a tool's output with
    a libx264 oracle or asserts libx264 arguments, so the result does not
    depend on whether the host has VideoToolbox. Call it with True to test
    the VideoToolbox arguments on any host.

    Returns:
        ``set(available)``, which makes
        ``lib.render_profile.hardware_video_available()`` return
        ``available`` until the test ends.
    """

    def set_available(available: bool) -> None:
        monkeypatch.setattr("lib.render_profile.hardware_video_available", lambda: available)

    return set_available
