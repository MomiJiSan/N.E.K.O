// Coupled by test_voice_readiness_control.py to actual Core + ASGI handlers.
// Only the microphone, DOM, and IPC transport are controlled here; both product
// frontend controllers receive the real backend results over this JSON pipe.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const readline = require('node:readline');
const assert = require('node:assert/strict');
const { randomUUID } = require('node:crypto');

const requests = new Map();
let sequence = 0;
readline.createInterface({ input: process.stdin }).on('line', line => {
    const reply = JSON.parse(line);
    const pending = requests.get(reply.id);
    requests.delete(reply.id);
    if (reply.error) pending.reject(new Error(reply.error)); else pending.resolve(reply.payload);
});
function rpc(channel, payload) {
    return new Promise((resolve, reject) => {
        const id = ++sequence;
        requests.set(id, { resolve, reject });
        process.stdout.write(JSON.stringify({ id, channel, payload }) + '\n');
    });
}
function element() {
    return {
        value: '', hidden: false, disabled: false, textContent: '', handlers: {},
        classList: { toggle() {} }, append() {}, appendChild() {}, replaceChildren() {},
        setAttribute() {}, removeAttribute() {},
        addEventListener(name, handler) { this.handlers[name] = handler; },
    };
}
function environment() {
    const elements = new Map();
    const document = {
        readyState: 'complete', documentElement: element(), body: element(),
        createElement: element,
        getElementById(id) { if (!elements.has(id)) elements.set(id, element()); return elements.get(id); },
    };
    const root = {
        document, crypto: { randomUUID }, setTimeout, clearTimeout, AbortController,
        Uint8Array, Promise, Date, console,
        location: { origin: 'http://localhost', hostname: 'localhost' },
        addEventListener() {}, removeEventListener() {},
        localStorage: { getItem() { return null; }, setItem() {} },
        navigator: { mediaDevices: {
            enumerateDevices: async () => [{ kind: 'audioinput', deviceId: 'actual-device', label: 'Controlled microphone' }],
            addEventListener() {},
        } },
    };
    root.window = root;
    const context = vm.createContext(root);
    function load(relative) {
        vm.runInContext(fs.readFileSync(path.join(__dirname, '../..', relative), 'utf8'), context, { filename: relative });
    }
    load('static/js/microphone-input.js');
    return { root, elements, load };
}
async function run() {
    const main = environment();
    let identity;
    let prepare;
    main.root.nekoVoiceEnrollment = {
        registerCapture: async value => { identity = value; return { accepted: true }; },
        onStopCapture: handler => { prepare = handler; }, onOpen() {},
    };
    const S = { isRecording: true, stream: null };
    let owner;
    const socket = { readyState: 1, send(json) {
        rpc('control', JSON.parse(json)).then(details => owner.controlResult(details, socket), error => { throw error; });
    } };
    S.socket = socket;
    main.load('static/app/app-voice-readiness.js');
    owner = main.root.createVoiceCaptureReadiness(S, async () => { S.isRecording = false; S.stream = null; });
    await owner.register(true);

    const child = environment();
    let prepared;
    child.root.nekoVoiceEnrollment = {
        async prepare({ operationId }) {
            prepared = { operationId, sessionId: identity.sessionId, revision: identity.revision };
            const ack = await prepare(prepared);
            prepared.token = ack.token;
            return { ...ack, operationId };
        },
        async release({ operationId }) {
            assert.equal(operationId, prepared.operationId);
            return prepare({ ...prepared, event: 'release' });
        },
    };
    child.load('static/js/voice-identity-readiness.js');
    let stream = null;
    let readiness;
    let track;
    let microphoneStops = 0;
    const pcm = new ArrayBuffer(48000 * 3 * 2);
    readiness = child.root.createVoiceIdentityReadiness({
        translate: (_key, fallback) => fallback, enrolling: () => false,
        render() {}, cancel() {}, pause() {}, status() {}, error: error => error.message,
        stream: () => stream,
        async microphone() {
            track = { readyState: 'live', label: 'Controlled microphone', getSettings: () => ({ deviceId: 'actual-device' }), stop() { this.readyState = 'ended'; } };
            stream = { getAudioTracks: () => [track], getTracks: () => [track] };
            readiness.receivedStream({ stream, deviceId: 'actual-device', label: track.label, fallback: false });
        },
        stop() { microphoneStops++; if (stream) child.root.nekoMicrophoneInput.stop(stream); stream = null; },
        capture: async () => pcm,
        async request(url, options) {
            if (url === '/resources') return { can_enroll: true, resources: {}, audio_contract: 'owner-campplus-desktop-v1' };
            assert.equal(url, '/audio/check');
            assert.equal(options.body.byteLength, 288000);
            return rpc('audio-check', { headers: options.headers, bytes: options.body.byteLength });
        },
    });
    await readiness.refreshResources();
    await child.elements.get('voice-identity-test').handlers.click();
    process.stdout.write(JSON.stringify({ channel: 'result', payload: {
        canStart: readiness.canStart(), audioContract: readiness.audioContract(),
        ownerBlocked: owner.blocked(), recording: S.isRecording,
        trackEnded: track.readyState === 'ended', microphoneStops,
        message: child.elements.get('voice-identity-test-result').textContent,
    } }) + '\n');
}
run().then(() => process.exit(0), error => { console.error(error); process.exit(1); });
