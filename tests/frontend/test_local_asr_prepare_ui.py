"""Real DOM coverage for production prepare/toast functions on web and bridge surfaces."""
import json
import re
from pathlib import Path
from uuid import uuid4

import pytest
from playwright.sync_api import Page, expect

ROOT = Path(__file__).resolve().parents[2]
LOCALES = ("en", "ja", "ko", "zh-CN", "zh-TW", "ru", "pt", "es")
KEYS = ("localAsrServicePreparing", "localAsrPrepareCancel", "localAsrServicePrepareFailed")


def install(page: Page, *, desktop: bool, width: int, ready: bool = False):
    websocket = (ROOT / "static/app/app-websocket.js").read_text(encoding="utf-8")
    begin = websocket.index("    async function prepareLocalStreamingAsr(")
    end = websocket.index("\n    window.prepareLocalStreamingAsr", begin)
    toasts = (ROOT / "static/app/app-ui/bootstrap-goodbye-and-toasts.js").read_text(encoding="utf-8")
    toast_begin = toasts.index("    function normalizeVoiceToastMessage(")
    toast_end = toasts.index("\n    I.mod.showVoicePreparingToast", toast_begin)
    translations = json.loads((ROOT / "static/locales/en.json").read_text(encoding="utf-8"))
    page.set_viewport_size({"width": width, "height": 800})
    page.route("**/static/icons/reminder_blue.png", lambda route: route.fulfill(path=ROOT / "static/icons/reminder_blue.png"))
    page.set_content('<html><head><base href="http://127.0.0.1/"></head><body style="margin:0;background:#162435;font-family:Arial,sans-serif;color:white"><main style="padding:24px">Local voice preparation</main></body></html>')
    page.evaluate("""({desktop, ready, translations, requestId, operationId}) => {
        window.S = {independentAsrEnabled:true, independentAsrProviderPreference:'local_streaming',
            coreApiSupportsIndependentAsr:true, voiceSessionStartEpoch:1};
        window.I = {mod:{}, S:window.S};
        window.__fixture = {desktop, ready, calls:[], starts:0, microphones:0, ipc:0, done:false};
        if (desktop) window.electronAPI = {sendMessage: () => { window.__fixture.ipc++; }};
        Object.defineProperty(window.crypto, 'randomUUID', {value:() => requestId, configurable:true});
        window.t = (key, args) => {
            const value = key.split('.').reduce((value, part) => value && value[part], translations) || key;
            return args ? value.replace('{code}', args.code) : value;
        };
        window.nekoLocalMutationSecurity = {getMutationHeaders:async () => ({'X-CSRF-Token':'fixture'})};
        window.fetch = async (url, options={}) => {
            const fixture = window.__fixture;
            fixture.calls.push({url, method:options.method || 'GET', body:options.body});
            let value;
            if (options.method === 'DELETE') value = {success:true,released:true};
            else value = fixture.ready
                ? {success:true,state:'ready',ready:true,operation_id:operationId}
                : {success:true,state:'loading',ready:false,operation_id:operationId};
            return {ok:true,json:async () => value};
        };
    }""", {"desktop": desktop, "ready": ready, "translations": translations,
             "requestId": str(uuid4()), "operationId": str(uuid4())})
    page.add_script_tag(content=toasts[toast_begin:toast_end] + "\nwindow.showVoicePreparingToast=I.showVoicePreparingToast;")
    page.add_script_tag(content=websocket[begin:end] + "\nwindow.prepareLocalStreamingAsr=prepareLocalStreamingAsr;")
    page.evaluate("""() => {
        // Only the post-prepare external actions are recorders. The production
        // helper, polling, cancellation, toast DOM, and click handler are real.
        window.__flow = (async () => {
            try {
                const result = await window.prepareLocalStreamingAsr(() => true);
                window.__fixture.starts++;
                window.__fixture.microphones++;
                window.__fixture.result = 'ready';
                await result.release();
            } catch (error) {
                window.__fixture.result = error.message;
                window.__fixture.cancelled = !!error.voiceStartCancelled;
            } finally { window.__fixture.done = true; }
        })();
    }""")


@pytest.mark.frontend
@pytest.mark.parametrize("desktop", [False, True], ids=["web", "desktop-bridge"])
@pytest.mark.parametrize("width", [1280, 360], ids=["wide", "narrow"])
def test_loading_cancel_is_clickable_bounded_and_prevents_start(page: Page, desktop: bool, width: int):
    install(page, desktop=desktop, width=width)
    button = page.locator("#voice-preparing-cancel")
    expect(button).to_be_visible()
    expect(button).to_have_text("Cancel preparation")
    expect(page.locator("#voice-preparing-toast")).to_contain_text("Preparing the local ASR service")
    page.wait_for_timeout(350)  # Let the production toast animation finish before layout inspection.
    mode = "desktop" if desktop else "web"
    artifact = ROOT / "artifacts" / f"m1c-ui-{mode}-{width}-loading.png"
    artifact.parent.mkdir(exist_ok=True)
    page.screenshot(path=artifact)
    bounds = page.evaluate("""() => {
        const toast=document.getElementById('voice-preparing-toast');
        const button=document.getElementById('voice-preparing-cancel');
        const t=toast.getBoundingClientRect(), b=button.getBoundingClientRect();
        return {left:b.left,right:b.right,viewport:innerWidth,toastRight:t.right,
            scroll:toast.scrollWidth,client:toast.clientWidth,pointer:getComputedStyle(button).pointerEvents};
    }""")
    assert bounds["pointer"] == "auto"
    assert bounds["left"] >= 0 and bounds["right"] <= bounds["viewport"]
    assert bounds["right"] <= bounds["toastRight"] + 1
    assert bounds["scroll"] <= bounds["client"] + 1
    button.click()
    page.wait_for_function("window.__fixture.done")
    result = page.evaluate("window.__fixture")
    assert result["cancelled"] and result["starts"] == result["microphones"] == result["ipc"] == 0
    posts = [call for call in result["calls"] if call["method"] == "POST"]
    deletes = [call for call in result["calls"] if call["method"] == "DELETE"]
    assert len(posts) == len(deletes) == 1
    assert deletes[0]["url"].endswith(json.loads(posts[0]["body"])["request_id"])
    expect(button).to_have_count(0)


@pytest.mark.frontend
@pytest.mark.parametrize("desktop", [False, True], ids=["web", "desktop-bridge"])
def test_ready_removes_only_prepare_cancel_control(page: Page, desktop: bool):
    install(page, desktop=desktop, width=360)
    expect(page.locator("#voice-preparing-cancel")).to_be_visible()
    page.evaluate("window.__fixture.ready=true")
    page.wait_for_function("window.__fixture.done")
    assert page.evaluate("window.__fixture.result") == "ready"
    assert page.evaluate("window.__fixture.starts") == page.evaluate("window.__fixture.microphones") == 1
    expect(page.locator("#voice-preparing-cancel")).to_have_count(0)
    page.screenshot(path=ROOT / "artifacts" / f"m1c-ui-{'desktop' if desktop else 'web'}-ready.png")


def test_prepare_status_keys_exist_in_every_supported_locale():
    source = (ROOT / "static/app/app-websocket.js").read_text(encoding="utf-8")
    paths = [re.search(r"window\.t\('((?:app|microphone)\." + key + r")'", source).group(1) for key in KEYS]
    for locale in LOCALES:
        values = json.loads((ROOT / "static/locales" / f"{locale}.json").read_text(encoding="utf-8"))
        for path in paths:
            value = values
            for part in path.split("."):
                value = value.get(part) if isinstance(value, dict) else None
            assert isinstance(value, str) and value.strip(), (locale, path)
