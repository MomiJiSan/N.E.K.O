const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function fixture() {
    const frames = new Map();
    let nextFrame = 0;
    let owner = null;
    let opening = 0;
    const bridge = {
        beginLipSync(token) { owner = token; opening = 0; return true; },
        setMouth(value, token) {
            if (token !== owner) return false;
            opening = value;
            return true;
        },
        endLipSync(token) {
            if (token !== owner) return false;
            owner = null;
            opening = 0;
            return true;
        },
    };
    const namespace = {
        shouldGuideAudioDriveMouth: () => true,
        clamp: (value, low, high) => Math.max(low, Math.min(high, value)),
    };
    const window = {
        __YuiGuideDirector: namespace,
        LanLan1: bridge,
        requestAnimationFrame(fn) { frames.set(++nextFrame, fn); return nextFrame; },
        cancelAnimationFrame(id) { frames.delete(id); },
    };
    const context = vm.createContext({ window, performance, Uint8Array, console });
    vm.runInContext(fs.readFileSync(path.join(__dirname,
        '../../static/tutorial/yui-guide/director/voice-queue.js'), 'utf8'), context);
    const queue = new namespace.YuiGuideVoiceQueue();
    const analyser = {
        fftSize: 64, quiet: false,
        getByteTimeDomainData(data) { data.fill(this.quiet ? 128 : 150); },
        disconnect() {},
    };
    return { queue, analyser, bridge, window, frames,
        state: () => ({ owner, opening }),
        sample(session) {
            const fn = frames.get(session.animationFrameId);
            frames.delete(session.animationFrameId);
            assert.ok(fn);
            fn(100);
        },
    };
}

test('guide pauses keep mouth ownership and stop releases only its own speech', () => {
    const f = fixture();
    const session = f.queue.startGuideMouthMotion('guide', { analyser: f.analyser });
    f.sample(session);
    assert.ok(f.state().opening > 0);
    const owner = f.state().owner;
    f.analyser.quiet = true;
    for (let i = 0; i < 10; i++) f.sample(session);
    assert.equal(f.state().opening, 0);
    assert.equal(f.state().owner, owner);
    f.queue.stopGuideMouthMotion(session);
    assert.equal(f.state().owner, null);
    assert.equal(f.frames.size, 0);
});

test('late guide sample and cleanup cannot close a newer assistant speech', () => {
    const f = fixture();
    const session = f.queue.startGuideMouthMotion('guide', { analyser: f.analyser });
    const lateSample = f.frames.get(session.animationFrameId);
    f.frames.delete(session.animationFrameId);
    const assistant = {};
    f.bridge.beginLipSync(assistant);
    f.bridge.setMouth(0.7, assistant);
    lateSample(200);
    assert.equal(f.state().opening, 0.7);
    f.queue.stopGuideMouthMotion(session);
    assert.equal(f.state().owner, assistant);
    assert.equal(f.state().opening, 0.7);
    lateSample(300);
    assert.equal(f.frames.size, 0);
});

test('late cleanup of old guide preserves the replacement guide', () => {
    const f = fixture();
    const old = f.queue.startGuideMouthMotion('old', { analyser: f.analyser });
    const current = f.queue.startGuideMouthMotion('new', { analyser: f.analyser });
    f.sample(current);
    const before = f.state();
    f.queue.stopGuideMouthMotion(old);
    assert.equal(f.queue.currentMouthMotionSession, current);
    assert.deepEqual(f.state(), before);
    f.queue.stopGuideMouthMotion(current);
});

test('frame startup failure releases the acquired guide owner', () => {
    const f = fixture();
    f.window.requestAnimationFrame = () => { throw new Error('frame unavailable'); };
    assert.equal(f.queue.startGuideMouthMotion('guide', { analyser: f.analyser }), null);
    assert.equal(f.state().owner, null);
    assert.equal(f.queue.currentMouthMotionSession, null);
});
