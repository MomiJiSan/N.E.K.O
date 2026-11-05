"""Exercise the production voice menu in web and desktop-bridge surfaces."""

import pytest
from playwright.sync_api import Page

from tests.frontend.voice_popover_harness import install_voice_popover_harness


@pytest.mark.frontend
@pytest.mark.parametrize("desktop_bridge", [False, True])
def test_local_streaming_choice_is_mutually_exclusive_with_whisper(
    page: Page, desktop_bridge: bool,
) -> None:
    install_voice_popover_harness(page, deferred_permission=False, desktop_bridge=desktop_bridge)
    result = page.evaluate("""async () => {
        const test = window.__voicePopoverTest;
        const state = test.state;
        state.localAsrAvailable = true;
        state.localStreamingAsrConfigured = true;
        state.independentAsrProviderPreference = 'auto';
        state.independentAsrEnabled = true;
        await window.renderFloatingMicList(test.popup());
        test.voiceAction().click();
        await Promise.resolve();
        const panel = test.panel();
        const inputs = panel.querySelectorAll('input[type="checkbox"]');
        const [whisper, streaming] = [inputs[2], inputs[3]];
        function change(input, checked) {
            input.checked = checked;
            input.dispatchEvent(new Event('change', {bubbles: true}));
        }
        change(streaming, true);
        const selected = [state.independentAsrProviderPreference, whisper.checked, streaming.checked];
        change(whisper, true);
        const replaced = [state.independentAsrProviderPreference, whisper.checked, streaming.checked];
        change(whisper, false);
        return {selected, replaced, cleared: state.independentAsrProviderPreference};
    }""")
    assert result == {
        "selected": ["local_streaming", False, True],
        "replaced": ["faster_whisper", True, False],
        "cleared": "auto",
    }


@pytest.mark.frontend
@pytest.mark.parametrize("master,core", [(False, True), (True, False)])
def test_local_streaming_choice_defensively_obeys_route_gates(
    page: Page, master: bool, core: bool,
) -> None:
    install_voice_popover_harness(page, deferred_permission=False)
    result = page.evaluate("""async ([master, core]) => {
        const test = window.__voicePopoverTest;
        const state = test.state;
        state.localAsrAvailable = false;
        state.localStreamingAsrConfigured = true;
        state.independentAsrProviderPreference = 'auto';
        state.independentAsrEnabled = master;
        state.coreApiSupportsIndependentAsr = core;
        await window.renderFloatingMicList(test.popup());
        test.voiceAction().click();
        await Promise.resolve();
        const toggle = test.panel().querySelectorAll('input[type="checkbox"]')[2];
        const disabled = toggle.disabled;
        toggle.checked = true;
        toggle.dispatchEvent(new Event('change', {bubbles: true}));
        return {disabled, checked: toggle.checked, preference: state.independentAsrProviderPreference};
    }""", [master, core])
    assert result == {"disabled": True, "checked": False, "preference": "auto"}


@pytest.mark.frontend
@pytest.mark.parametrize("desktop_bridge", [False, True])
def test_service_configuration_changes_preserve_saved_choice_and_allow_turnoff(
    page: Page, desktop_bridge: bool,
) -> None:
    install_voice_popover_harness(page, deferred_permission=False, desktop_bridge=desktop_bridge)
    result = page.evaluate("""async () => {
        const test = window.__voicePopoverTest;
        const state = test.state;
        state.localAsrAvailable = false;
        state.localStreamingAsrConfigured = null;
        state.independentAsrProviderPreference = 'auto';
        state.independentAsrEnabled = true;
        await window.renderFloatingMicList(test.popup());
        test.voiceAction().click();
        await Promise.resolve();
        const panel = test.panel();
        const count = () => panel.querySelectorAll('input[type="checkbox"]').length;
        const before = count();
        state.localStreamingAsrConfigured = true;
        window.dispatchEvent(new CustomEvent('neko:core-api-capability-changed'));
        const afterConfig = count();
        const toggle = panel.querySelectorAll('input[type="checkbox"]')[2];
        // Even a stale DOM event cannot enable a now-unconfigured service.
        state.localStreamingAsrConfigured = false;
        toggle.checked = true;
        toggle.dispatchEvent(new Event('change', {bubbles: true}));
        const rejected = state.independentAsrProviderPreference;
        state.independentAsrProviderPreference = 'local_streaming';
        state.independentAsrEnabled = false;
        state.coreApiSupportsIndependentAsr = false;
        window.dispatchEvent(new CustomEvent('neko:conversation-settings-hydrated'));
        const saved = panel.querySelectorAll('input[type="checkbox"]')[2];
        const savedState = [saved.checked, saved.disabled, state.independentAsrProviderPreference];
        saved.checked = false;
        saved.dispatchEvent(new Event('change', {bubbles: true}));
        return {before, afterConfig, rejected, savedState, afterOff: count(), preference: state.independentAsrProviderPreference};
    }""")
    assert result == {
        "before": 2,
        "afterConfig": 3,
        "rejected": "auto",
        "savedState": [True, False, "local_streaming"],
        "afterOff": 2,
        "preference": "auto",
    }
