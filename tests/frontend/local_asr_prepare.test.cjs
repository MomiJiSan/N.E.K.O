const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');
const { randomUUID } = require('node:crypto');

const root = path.join(__dirname, '../..');
const source = fs.readFileSync(path.join(root, 'static/app/app-websocket.js'), 'utf8');
const start = source.indexOf('    async function prepareLocalStreamingAsr(');
const end = source.indexOf('\n    window.prepareLocalStreamingAsr', start);
assert.ok(start >= 0 && end > start);

function fixture(responder) {
    const calls = [];
    const S = { independentAsrEnabled: true, independentAsrProviderPreference: 'local_streaming',
        coreApiSupportsIndependentAsr: true, voiceSessionStartEpoch: 1 };
    const window = {
        crypto: { randomUUID },
        nekoLocalMutationSecurity: { getMutationHeaders: async () => ({ 'X-CSRF-Token': 'test' }) },
        showVoicePreparingToast: (_message, controls) => { window.controls = controls; },
        fetch: async (url, options = {}) => {
            calls.push({ url, ...options });
            if (options.method === 'DELETE') return response({ success: true });
            return responder(url, options, calls);
        },
    };
    const context = vm.createContext({ S, window, AbortController, setTimeout, clearTimeout, setInterval, clearInterval });
    vm.runInContext(source.slice(start, end), context);
    return { S, window, calls, prepare: context.prepareLocalStreamingAsr };
}
function response(value, ok = true) { return { ok, json: async () => value }; }
function pendingUntilAbort(options) {
    return new Promise((_, reject) => options.signal.addEventListener('abort', () => reject(new Error('aborted')), { once: true }));
}

test('prepare polls outside the audio handshake and releases only its own UUID once', async () => {
    const id = randomUUID();
    const f = fixture(async (_url, options) => options.method === 'POST'
        ? response({ success: true, ready: false, state: 'loading', operation_id: id })
        : response({ success: true, ready: true, state: 'ready', operation_id: id }));
    const ticket = await f.prepare(() => true);
    assert.equal(f.calls.length, 2);
    assert.equal(f.calls[0].method, 'POST');
    const request = JSON.parse(f.calls[0].body);
    assert.equal(request.provider, 'local_streaming');
    assert.equal(request.enabled, true);
    assert.equal(f.calls[0].headers['X-CSRF-Token'], 'test');
    assert.equal(f.calls[1].url, '/api/config/local-asr/operations/' + id);
    await ticket.release();
    await ticket.release();
    assert.equal(f.calls.length, 3);
    assert.equal(f.calls[2].url, '/api/config/local-asr/prepare/' + request.request_id);
});

for (const [name, change] of [['cloud', { independentAsrProviderPreference: 'auto' }],
    ['disabled', { independentAsrEnabled: false }], ['native Core', { coreApiSupportsIndependentAsr: false }]]) {
    test(name + ' does not start a local preparation', async () => {
        const f = fixture(() => { throw new Error('must not fetch'); });
        Object.assign(f.S, change);
        await (await f.prepare(() => true)).release();
        assert.equal(f.calls.length, 0);
    });
}

test('explicit cancel aborts pending HTTP and compensates by request ID', async () => {
    const f = fixture(async (_url, options) => pendingUntilAbort(options));
    const task = f.prepare(() => true);
    while (!f.window.controls) await new Promise(resolve => setTimeout(resolve, 1));
    f.window.controls.onCancel();
    await assert.rejects(task, error => error.voiceStartCancelled && error.microphoneStartCancelled);
    assert.equal(f.calls.at(-1).method, 'DELETE');
});

test('ABA cancellation epoch fences a late ready result', async () => {
    const f = fixture(async (_url, options) => pendingUntilAbort(options));
    const task = f.prepare(() => true);
    while (!f.window.controls) await new Promise(resolve => setTimeout(resolve, 1));
    f.S.voiceSessionStartEpoch += 1;
    await assert.rejects(task, error => error.voiceStartCancelled);
    assert.equal(f.calls.at(-1).method, 'DELETE');
});

test('the total deadline is bounded even before the first ready result', async () => {
    const f = fixture(async (_url, options) => pendingUntilAbort(options));
    await assert.rejects(f.prepare(() => true, { timeoutMs: 25 }), /ASR_LOCAL_PREPARE_TIMEOUT/);
    assert.equal(f.calls.at(-1).method, 'DELETE');
});

test('changing provider during prepare cancels without starting the new provider', async () => {
    const f = fixture(async (_url, options) => pendingUntilAbort(options));
    const task = f.prepare(() => true);
    while (!f.window.controls) await new Promise(resolve => setTimeout(resolve, 1));
    f.S.independentAsrProviderPreference = 'faster_whisper';
    await assert.rejects(task, error => error.voiceStartCancelled);
    assert.equal(f.calls.filter(call => call.method === 'POST').length, 1);
});

test('terminal service failure releases the waiter and preserves a failure result', async () => {
    const f = fixture(async () => response({ success: true, state: 'failed', ready: false, error_code: 'MODEL_HASH_MISMATCH' }));
    await assert.rejects(f.prepare(() => true), /MODEL_HASH_MISMATCH/);
    assert.equal(f.calls.at(-1).method, 'DELETE');
});

test('both production audio start paths prepare before send and clean up in finally', () => {
    const buttons = fs.readFileSync(path.join(root, 'static/app/app-buttons.js'), 'utf8');
    const manual = buttons.indexOf('localAsrPreparation = await window.prepareLocalStreamingAsr');
    const manualSend = buttons.indexOf("action: 'start_session'", manual);
    assert.ok(manual > 0 && manualSend > manual);
    assert.match(buttons.slice(manualSend), /finally \{\s*if \(localAsrPreparation\) await localAsrPreparation.release\(\)/);
    const restart = source.indexOf('restartLocalAsrPreparation = await prepareLocalStreamingAsr');
    assert.ok(restart > 0 && source.indexOf("action: 'start_session'", restart) > restart);
    assert.match(source.slice(restart), /finally \{\s*if \(restartLocalAsrPreparation\) await restartLocalAsrPreparation.release\(\)/);
});

test('HTTP frontend without randomUUID uses a cryptographic canonical UUID', async () => {
    const f = fixture(async () => response({ success: true, ready: true, state: 'ready' }));
    f.window.crypto = { getRandomValues: bytes => require('node:crypto').randomFillSync(bytes) };
    const ticket = await f.prepare(() => true);
    assert.match(JSON.parse(f.calls[0].body).request_id, /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/);
    await ticket.release();
});

test('cancelling at the first visible control observes an already-rejected fetch', async () => {
    const f = fixture(async (_url, options) => {
        if (options.signal.aborted) throw new Error('fetch aborted before dispatch');
        return pendingUntilAbort(options);
    });
    f.window.showVoicePreparingToast = (_message, controls) => controls.onCancel();
    await assert.rejects(f.prepare(() => true), error => error.voiceStartCancelled);
    assert.equal(f.calls.at(-1).method, 'DELETE');
});
