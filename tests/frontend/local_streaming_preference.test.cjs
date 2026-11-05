const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

const root = path.join(__dirname, '../..');
const websocket = fs.readFileSync(path.join(root, 'static/app/app-websocket.js'), 'utf8');
const settings = fs.readFileSync(path.join(root, 'static/app/app-settings.js'), 'utf8');

function functionSource(source, name) {
    const start = source.indexOf(`    function ${name}(`);
    assert.ok(start >= 0, `Missing production function ${name}`);
    const end = source.indexOf('\n    }', start);
    assert.ok(end > start);
    return source.slice(start, end + '\n    }'.length);
}

function fixture() {
    const events = [];
    const S = {
        settingsHydrated: true,
        independentAsrProviderPreferenceAuthoritative: true,
        independentAsrProviderPreference: 'local_streaming',
    };
    const context = vm.createContext({
        S,
        window: { dispatchEvent: event => events.push(event) },
        CustomEvent: class { constructor(type, options) { this.type = type; this.detail = options.detail; } },
    });
    vm.runInContext([
        functionSource(websocket, 'attachStartSessionHandshake'),
        functionSource(websocket, 'publishCoreApiCapability'),
        functionSource(settings, '_normalizeIndependentAsrProviderPreference'),
    ].join('\n'), context);
    return { context, S, events };
}

test('local streaming preference survives normalization; unknown values use auto', () => {
    const { context } = fixture();
    for (const value of ['auto', 'faster_whisper', 'local_streaming']) {
        assert.equal(context._normalizeIndependentAsrProviderPreference(value), value);
    }
    for (const value of ['http://example.com', null, true, 'qwen']) {
        assert.equal(context._normalizeIndependentAsrProviderPreference(value), 'auto');
    }
});

test('both initial and replacement sockets carry authoritative local streaming preference', () => {
    const { context, S } = fixture();
    // A stale UI capability cannot overwrite the persisted user choice.
    S.coreApiSupportsIndependentAsr = false;
    for (let i = 0; i < 2; i++) {
        const sent = [];
        const ws = { send: value => sent.push(JSON.parse(value)) };
        context.attachStartSessionHandshake(ws);
        ws.send(JSON.stringify({ action: 'start_session', input_type: 'audio' }));
        assert.equal(sent[0].independent_asr_provider_preference, 'local_streaming');
    }
});

test('unhydrated or nonauthoritative provider choices cannot override backend persistence', () => {
    for (const field of ['settingsHydrated', 'independentAsrProviderPreferenceAuthoritative']) {
        const { context, S } = fixture();
        S[field] = false;
        const sent = [];
        const ws = { send: value => sent.push(JSON.parse(value)) };
        context.attachStartSessionHandshake(ws);
        ws.send(JSON.stringify({ action: 'start_session' }));
        assert.equal(Object.hasOwn(sent[0], 'independent_asr_provider_preference'), false);
    }
});

test('service-only capability change refreshes the menu without claiming service readiness', () => {
    const { context, S, events } = fixture();
    context.publishCoreApiCapability('qwen', true, false, false);
    const count = events.length;
    context.publishCoreApiCapability('qwen', true, false, true);
    assert.equal(events.length, count + 1);
    assert.equal(S.localAsrAvailable, false);
    assert.equal(S.localStreamingAsrConfigured, true);
    assert.equal(S.independentAsrActive, undefined);
    // Old backends or failed refreshes do not leave a stale available option.
    context.publishCoreApiCapability('qwen', true, true);
    assert.equal(S.localAsrAvailable, true);
    assert.equal(S.localStreamingAsrConfigured, null);
});
