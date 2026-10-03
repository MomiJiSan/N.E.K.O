'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const crypto = require('node:crypto').webcrypto;
const input = require('./js/microphone-input.js');
const readinessSource = fs.readFileSync(path.join(__dirname, 'js/voice-identity-readiness.js'), 'utf8');
const ownerSource = fs.readFileSync(path.join(__dirname, 'app/app-voice-readiness.js'), 'utf8');
function deferred() { let resolve; const promise = new Promise(r => { resolve = r; }); return { promise, resolve }; }
function trackStream(readyState = 'live') {
    const track = { readyState, label: 'Actual device', enabled: true, stop() { this.readyState = 'ended'; }, getSettings() { return { deviceId: 'actual-device' }; } };
    return { track, getTracks: () => [track], getAudioTracks: () => [track] };
}
function element() {
    const handlers = new Map();
    return { value: '', textContent: '', checked: false, children: [], disabled: false, hidden: false, classList: { toggle() {} }, setAttribute() {}, removeAttribute() {}, addEventListener(name, fn) { handlers.set(name, fn); }, appendChild(child) { this.children.push(child); }, append(...children) { this.children.push(...children); }, replaceChildren() { this.children = []; }, emit(name) { return handlers.get(name)?.({ target: this }); } };
}
function harness({ checkGate, captureGate, accepted = true, resourceReady = true, desktopGate, requestRouter } = {}) {
    const elements = new Map();
    const events = new Map();
    const mediaEvents = new Map();
    const calls = [];
    const storage = new Map();
    let stream;
    let stopped = 0;
    let controller;
    const root = { crypto, location: { origin: 'http://localhost:48911' }, opener: null, nekoMicrophoneInput: input, addEventListener(name, fn) { events.set(name, fn); }, removeEventListener(name, fn) { if(events.get(name)===fn)events.delete(name); }, setTimeout, clearTimeout };
    if (desktopGate) root.nekoVoiceEnrollment = { prepare: () => desktopGate.promise, release: async () => {} };
    const document = { getElementById(id) { if (!elements.has(id)) elements.set(id, element()); return elements.get(id); }, createElement: element };
    const hooks = {
        translate: (_, fallback) => fallback, render() {}, error: error => error.message,
        enrolling: () => false, stream: () => stream,
        async microphone() { stream = trackStream(); controller.receivedStream({ ...stream, label: stream.track.label, deviceId: 'actual-device', fallback: false }); },
        async capture() { if (captureGate) await captureGate.promise; return new ArrayBuffer(288000); },
        pause() {}, stop() { stopped++; if (stream) input.stop(stream); stream = null; }, cancel() {}, status: async () => {},
        async request(url, config) {
            calls.push({ url, config });
            if (requestRouter) return requestRouter(url, config);
            if (url === '/resources') return { can_enroll: resourceReady, resources: { campp: { state: resourceReady ? 'ready' : 'missing' }, wake_runtime: { state: 'ready' } }, wake_enabled: false };
            if (url === '/audio/check/isolation') return { token: 'opaque-ticket', ttl_seconds: 60 };
            if (url === '/audio/check') { if (checkGate) await checkGate.promise; return { accepted, reason: accepted ? null : 'no_speech_detected', audio_contract: { revision: 1, noise_reduction_enabled: false } }; }
            return {};
        }
    };
    vm.runInNewContext(readinessSource, { window: root, document, navigator: { mediaDevices: { enumerateDevices: async () => [], addEventListener(name,fn) { mediaEvents.set(name,fn); } } }, localStorage: { getItem: key => storage.get(key), setItem: (key, value) => storage.set(key, value) }, crypto, AbortController, Uint8Array, Date, Math, Promise }, { filename: path.join(__dirname, 'js/voice-identity-readiness.js') });
    controller = root.createVoiceIdentityReadiness(hooks);
    return { controller, calls, elements, events, mediaEvents, hooks, root, stopped: () => stopped };
}
test('opening a selected unavailable device reports the real fallback device', async () => {
    const stream = trackStream(); const constraints = [];
    const result = await input.open({ async getUserMedia(config) { constraints.push(config); if (constraints.length === 1) { const e = new Error(); e.name = 'OverconstrainedError'; throw e; } return stream; } }, 'foreign-id', () => true);
    assert.equal(result.fallback, true); assert.equal(result.deviceId, 'actual-device'); assert.equal(result.label, 'Actual device');
    assert.equal(constraints[0].audio.deviceId.exact, 'foreign-id'); assert.equal(constraints[1].audio.deviceId, undefined);
});
test('a late permission result is stopped and cannot become the current input', async () => {
    const gate = deferred(); const stream = trackStream(); let current = true;
    const pending = input.open({ getUserMedia: () => gate.promise }, '', () => current);
    current = false; gate.resolve(stream);
    await assert.rejects(pending, /capture_cancelled/); assert.equal(stream.track.readyState, 'ended');
});
test('an ended microphone stream is rejected before capture', async () => {
    const stream = trackStream('ended');
    await assert.rejects(input.open({ getUserMedia: async () => stream }, '', () => true), { name: 'NotReadableError' });
});
test('input checks require a server ticket and never create an enrollment', async () => {
    const h = harness(); await h.controller.refreshResources();
    assert.equal(h.controller.canStart(), false);
    await h.elements.get('voice-identity-test').emit('click');
    assert.equal(h.controller.canStart(), true);
    const check = h.calls.find(c => c.url === '/audio/check');
    assert.equal(check.config.headers['X-Voice-Input-Check'], 'opaque-ticket');
    assert.equal(check.config.body.byteLength, 288000);
    assert.equal(h.calls.some(c => /enrollment|profile/.test(c.url)), false);
    assert.equal(h.elements.get('voice-identity-actual-device').textContent, 'Actual device');
});
test('input quality alone cannot bypass missing required resources', async () => {
    const h = harness({ resourceReady: false }); await h.controller.refreshResources();
    await h.elements.get('voice-identity-test').emit('click');
    assert.equal(h.controller.canStart(), false);
});
test('postprocessing rejection keeps enrollment blocked', async () => {
    const h = harness({ accepted: false }); await h.controller.refreshResources();
    await h.elements.get('voice-identity-test').emit('click');
    assert.equal(h.controller.canStart(), false);
    assert.match(h.elements.get('voice-identity-test-result').textContent, /no_speech_detected/);
});
test('changing gain during a delayed input check fences its successful result', async () => {
    const gate = deferred(); const h = harness({ checkGate: gate }); await h.controller.refreshResources();
    const pending = h.elements.get('voice-identity-test').emit('click');
    while (!h.calls.some(c => c.url === '/audio/check')) await new Promise(resolve => setImmediate(resolve));
    const gain = h.elements.get('voice-identity-gain'); gain.value = '12'; gain.emit('change');
    gate.resolve(); await pending;
    assert.equal(h.controller.canStart(), false); assert.ok(h.stopped() > 0);
});
test('late desktop stop acknowledgement cannot enable a superseded test', async () => {
    const gate = deferred(); const h = harness({ desktopGate: gate }); await h.controller.refreshResources();
    const pending = h.elements.get('voice-identity-test').emit('click');
    const gain = h.elements.get('voice-identity-gain'); gain.value = '5'; gain.emit('change');
    gate.resolve({ operationId: 'old-operation', stopped: true, token: 'stale-ticket' }); await pending;
    assert.equal(h.controller.canStart(), false); assert.equal(h.calls.some(c => c.url === '/audio/check'), false);
});

test('a cancelled browser stop acknowledgement releases its own late token', async () => {
    const h=harness(); await h.controller.refreshResources(); const messages=[];
    h.root.opener={closed:false,postMessage(message){messages.push(message);}};
    const pending=h.elements.get('voice-identity-test').emit('click');
    h.elements.get('voice-identity-test-cancel').emit('click');
    const receive=h.events.get('message');
    receive({origin:h.root.location.origin,source:h.root.opener,data:{type:'neko-voice-enrollment-stopped',operationId:messages[0].operationId,stopped:true,token:'late-token'}});
    await pending;
    assert.equal(messages[1].type,'neko-voice-enrollment-release');assert.equal(messages[1].token,'late-token');
    assert.equal(h.calls.some(call=>call.url==='/audio/check'),false);assert.equal(h.controller.canStart(),false);
});

test('a timed-out opener acknowledgement cannot replace the server-proven inactive ticket', async () => {
    const capture=deferred();const h=harness({captureGate:capture});await h.controller.refreshResources();const messages=[];const timers=new Map();let next=0;
    h.root.setTimeout=(fn,delay)=>{const id=++next;timers.set(id,{fn,delay});return id;};h.root.clearTimeout=id=>timers.delete(id);
    h.root.opener={closed:false,postMessage(message){messages.push(message);}};
    const pending=h.elements.get('voice-identity-test').emit('click');
    [...timers.values()].find(timer=>timer.delay===10000).fn();
    while(!h.hooks.stream())await new Promise(resolve=>setImmediate(resolve));
    h.events.get('message')({origin:h.root.location.origin,source:h.root.opener,data:{type:'neko-voice-enrollment-stopped',operationId:messages[0].operationId,stopped:true,token:'expired-opener-ticket'}});
    capture.resolve();await pending;
    assert.equal(h.calls.find(call=>call.url==='/audio/check').config.headers['X-Voice-Input-Check'],'opaque-ticket');
    assert.equal(messages[1].token,'expired-opener-ticket');assert.equal(h.controller.canStart(),true);
});

test('device changes invalidate an accepted test after its tracks have been released', async () => {
    const h=harness();await h.controller.refreshResources();await h.elements.get('voice-identity-test').emit('click');
    assert.equal(h.hooks.stream(),null);assert.equal(h.controller.canStart(),true);
    await h.mediaEvents.get('devicechange')();assert.equal(h.controller.canStart(),false);
});
test('activation notifications are scoped to the live socket and microphone lifecycle', () => {
    const events = new Map(); const S = { socket: { readyState: 1 }, isRecording: true };
    const root = { crypto, t: key => key, addEventListener(name, fn) { events.set(name, fn); }, setTimeout, clearTimeout };
    const document = { readyState: 'complete', body: element(), documentElement: element(), createElement: element };
    vm.runInNewContext(ownerSource, { window: root, document, crypto, Set, Map }, { filename: path.join(__dirname, 'app/app-voice-readiness.js') });
    const controller = root.createVoiceCaptureReadiness(S, async () => {});
    const base = { session_id: 'one', microphone_generation: 1, route_generation: 1, profile_revision: 1, permission_revision: 1, revision: 1, state: 'waiting' };
    assert.equal(controller.activationStatus(base, {}), false);
    assert.equal(controller.activationStatus(base, S.socket), true);
    assert.equal(controller.activationStatus({ ...base, revision: 2, microphone_generation: 0, state: 'active' }, S.socket), false);
    assert.equal(controller.activationStatus({ ...base, revision: 1, route_generation: 2, state: 'preparing' }, S.socket), true);
    assert.equal(controller.activationStatus({ ...base, revision: 99, route_generation: 1, state: 'active' }, S.socket), false);
    controller.reset();
    assert.equal(controller.activationStatus({ ...base, revision: 3, state: 'active' }, S.socket), false);
});

test('desktop preparation stops live audio before the server isolation handshake and release never restarts it', async () => {
    const stream = trackStream(); const order = []; const registrations = [];
    let receive; let controller;
    const socket = { readyState: 1, send(raw) {
        const message = JSON.parse(raw); order.push(message.event);
        queueMicrotask(() => controller.controlResult({ event: message.event, request_id: message.request_id, ok: true, token: 'server-token', ttl_seconds: 60 }, socket));
    } };
    const S = { socket, isRecording: true, stream };
    const root = { crypto, t: key => key, addEventListener() {}, setTimeout, clearTimeout,
        nekoVoiceEnrollment: { onStopCapture(fn) { receive = fn; }, onOpen() {}, async registerCapture(details) { registrations.push(details); return { accepted: true }; } } };
    const document = { readyState: 'complete', body: element(), documentElement: element(), createElement: element };
    vm.runInNewContext(ownerSource, { window: root, document, crypto, Set, Map }, { filename: path.join(__dirname, 'app/app-voice-readiness.js') });
    controller = root.createVoiceCaptureReadiness(S, async () => { order.push('stop'); input.stop(stream); S.isRecording = false; });
    await Promise.resolve();
    const identity = registrations[0];
    const result = await receive({ event: 'prepare', operationId: 'op', sessionId: identity.sessionId, revision: identity.revision });
    assert.deepEqual(order, ['stop', 'preview_begin']); assert.equal(result.token, 'server-token');
    assert.equal(stream.track.readyState, 'ended'); assert.equal(controller.blocked(), true);
    await receive({ event: 'release', operationId: 'op', token: 'server-token' });
    assert.equal(controller.blocked(), false); assert.equal(S.isRecording, false);
    assert.deepEqual(order, ['stop', 'preview_begin', 'preview_end']);
});

test('wrong socket acknowledgement cannot finish the current isolation request', async () => {
    let receive; let sent; let controller;
    const socket = { readyState: 1, send(raw) { sent = JSON.parse(raw); } };
    const S = { socket, isRecording: false };
    const root = { crypto, t: key => key, addEventListener() {}, setTimeout, clearTimeout,
        nekoVoiceEnrollment: { onStopCapture(fn) { receive = fn; }, onOpen() {}, async registerCapture() { return { accepted: true }; } } };
    const document = { readyState: 'complete', body: element(), documentElement: element(), createElement: element };
    vm.runInNewContext(ownerSource, { window: root, document, crypto, Set, Map }, { filename: path.join(__dirname, 'app/app-voice-readiness.js') });
    controller = root.createVoiceCaptureReadiness(S, async () => {});
    const pending = receive({ operationId: 'op' });
    while (!sent) await new Promise(resolve => setImmediate(resolve));
    const ack = { event: 'preview_begin', request_id: sent.request_id, ok: true, token: 'ticket', ttl_seconds: 60 };
    controller.controlResult(ack, {});
    let settled = false; pending.then(() => { settled = true; }); await Promise.resolve(); assert.equal(settled, false);
    controller.controlResult(ack, socket); await pending; assert.equal(settled, true);
    const release = receive({ event: 'release', operationId: 'op', token: 'ticket' });
    controller.controlResult({ event: sent.event, request_id: sent.request_id, ok: true }, socket);
    await release;
});

function isolationHarness(stopFailure = false) {
    const timers = new Map(); let sequence = 0; let receive; let controller; const sent = [];
    const socket = { readyState: 1, send(raw) { sent.push(JSON.parse(raw)); } };
    const S = { socket, isRecording: false };
    const root = { crypto, t: key => key, addEventListener() {}, setTimeout(fn, delay) { const id=++sequence; timers.set(id,{ fn,delay }); return id; }, clearTimeout(id) { timers.delete(id); },
        nekoVoiceEnrollment: { onStopCapture(fn) { receive=fn; }, onOpen() {}, async registerCapture() { return { accepted:true }; } } };
    const document = { readyState:'complete', body:element(), documentElement:element(), createElement:element };
    vm.runInNewContext(ownerSource,{window:root,document,crypto,Set,Map}, { filename: path.join(__dirname, 'app/app-voice-readiness.js') });
    controller=root.createVoiceCaptureReadiness(S,async()=>{ if(stopFailure) throw new Error('stop_failed'); });
    function ack(message, extra = {}) { controller.controlResult({ ...message, ok:true, token:'ticket-'+message.request_id, ttl_seconds:60, ...extra },socket); }
    return {controller,receive:request=>receive(request),sent,timers,ack,S};
}

test('a failed actual stop releases only its local isolation fence', async () => {
    const h=isolationHarness(true);
    await assert.rejects(h.receive({operationId:'stop-failure'}),/stop_failed/);
    assert.equal(h.controller.blocked(),false); assert.equal(h.sent.length,0); assert.equal(h.S.isRecording,false);
});

test('a confirmed server failure permits an explicit new microphone attempt', async () => {
    const h=isolationHarness(); const pending=h.receive({operationId:'rejected'});
    await new Promise(resolve => setImmediate(resolve)); h.ack(h.sent[0],{ok:false,reason:'preview_unavailable'});
    await assert.rejects(pending,/preview_unavailable/);
    assert.equal(h.controller.blocked(),false); assert.equal(h.S.isRecording,false);
});

test('a lost begin acknowledgement is bounded and its late token is compensated', async () => {
    const h=isolationHarness(); const pending=h.receive({operationId:'late'});
    await new Promise(resolve => setImmediate(resolve)); const timeout=[...h.timers.values()].find(timer=>timer.delay===8000);
    timeout.fn(); await assert.rejects(pending,/voice_control_timeout/);
    assert.equal(h.controller.blocked(),true);
    h.ack(h.sent[0]); assert.equal(h.sent[1].event,'preview_end');
    h.ack(h.sent[1]); await Promise.resolve();
    assert.equal(h.controller.blocked(),false); assert.equal(h.S.isRecording,false);
});

test('ticket expiry retires its own fence and an old timer cannot free a new operation', async () => {
    const h=isolationHarness(); const first=h.receive({operationId:'first'});
    await new Promise(resolve => setImmediate(resolve)); h.ack(h.sent[0]); await first;
    const oldExpiry=[...h.timers.values()].find(timer=>timer.delay===60000).fn;
    const released=h.receive({event:'release',operationId:'first',token:'ticket-first'});
    h.ack(h.sent[1]); await released;
    const second=h.receive({operationId:'second'}); await new Promise(resolve => setImmediate(resolve)); h.ack(h.sent[2]); await second;
    oldExpiry(); assert.equal(h.controller.blocked(),true);
    const newExpiry=[...h.timers.values()].find(timer=>timer.delay===60000).fn;
    newExpiry(); assert.equal(h.controller.blocked(),false); assert.equal(h.S.isRecording,false);
});

test('resource queries only read snapshots and a cancelled operation targets its own id', async () => {
    const gate = deferred();
    const h = harness({ requestRouter: async url => {
        if (url === '/resources') return { can_enroll: false, resources: { campp: { state: 'unchecked' }, wake_runtime: { state: 'missing' } } };
        if (url === '/resources/prepare') return { operation_id: 'owned-resource', state: 'pending' };
        if (url === '/resources/operations/owned-resource') { await gate.promise; return { operation_id: 'owned-resource', state: 'running' }; }
        return {};
    } });
    await h.controller.refreshResources();
    assert.deepEqual(h.calls.map(call => call.url), ['/resources']);
    const preparing = h.elements.get('voice-identity-prepare').emit('click');
    while (!h.calls.some(call => call.url === '/resources/operations/owned-resource')) await new Promise(resolve => setImmediate(resolve));
    await h.elements.get('voice-identity-resource-cancel').emit('click');
    gate.resolve(); await preparing;
    assert.ok(h.calls.some(call => call.url === '/resources/operations/owned-resource/cancel' && call.config.method === 'POST'));
    assert.equal(h.controller.canStart(), false);
});

test('an older resource snapshot cannot overwrite the newer audio contract', async () => {
    const first = deferred(); let query = 0;
    const h = harness({ requestRouter: async () => { query++; if (query === 1) return first.promise; return { can_enroll: false, audio_contract: { noise_reduction_enabled: true }, resources: { campp: { state: 'missing' } } }; } });
    const older = h.controller.refreshResources(); await h.controller.refreshResources();
    first.resolve({ can_enroll: true, audio_contract: { noise_reduction_enabled: false }, resources: { campp: { state: 'ready' } } }); await older;
    const rows = h.elements.get('voice-identity-resources').children;
    assert.equal(rows.length, 1); assert.match(rows[0].textContent, /missing/);
});

test('resource repair in a remote browser cannot invoke a local desktop repair action', async () => {
    const h = harness({ requestRouter: async () => ({ can_enroll: false, repair_action: 'app_repair', resources: { wake_runtime: { state: 'missing' } } }) });
    let localRepairs = 0; let url;
    h.root.location.hostname = 'remote.example';
    h.root.nekoVoiceEnrollment = { async openRepair() { localRepairs++; } };
    h.root.open = value => { url = value; };
    await h.controller.refreshResources(); await h.elements.get('voice-identity-repair').emit('click');
    assert.equal(localRepairs, 0); assert.equal(url, '/api/voice-identity/resources/repair-guide');
});
