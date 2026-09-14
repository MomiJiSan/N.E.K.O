const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

function harness() {
    const listeners = new Map(), timers = new Map();
    let nextTimer = 0, observerCallback, observing = false;
    const listen = (type, callback) => {
        const callbacks = listeners.get(type) || [];
        callbacks.push(callback); listeners.set(type, callbacks);
    };
    const classes = new Set(), artClasses = new Set(), attributes = new Map();
    const classList = set => ({
        add: name => set.add(name), remove: name => set.delete(name),
        contains: name => set.has(name),
        toggle: (name, enabled) => enabled ? set.add(name) : set.delete(name),
    });
    const art = { classList: classList(artClasses), getBoundingClientRect: () => ({ left: 0, right: 20, top: 0, bottom: 20 }) };
    const button = { isConnected: true, classList: classList(new Set()) };
    const container = {
        isConnected: true, style: { cursor: 'grab' }, classList: classList(classes),
        querySelector: () => art, contains: node => node === button,
        toggleAttribute: (key, value) => value ? attributes.set(key, '') : attributes.delete(key),
        getAttribute: key => attributes.get(key) ?? null,
    };
    const window = {
        edgePeekLockEnabled: true, addEventListener: listen,
        document: { documentElement: {}, addEventListener: listen },
        MutationObserver: class {
            constructor(callback) { observerCallback = callback; }
            observe() { observing = true; }
            disconnect() { observing = false; }
        },
    };
    vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../../static/avatar/avatar-ui-buttons/edge-peek-controller.js'), 'utf8'), {
        window, setTimeout: callback => { timers.set(++nextTimer, callback); return nextTimer; },
        clearTimeout: id => timers.delete(id),
    });
    return {
        window, button, container, artClasses, timers,
        controller: window.NekoEdgePeekController,
        emit(type, event = {}) { for (const callback of listeners.get(type) || []) callback(event); },
        fireTimers() { const callbacks = [...timers.values()]; timers.clear(); callbacks.forEach(callback => callback()); },
        remove() { button.isConnected = container.isConnected = false; observerCallback(); },
        get observing() { return observing; },
    };
}

test('ordinary and inactive model locks do not block an unanchored return ball', () => {
    const h = harness();
    const model = {};
    h.window.live2dManager = { isLocked: true, currentModel: model };
    h.window.vrmManager = { isLocked: true, interaction: { isLocked: true } };
    assert.equal(h.controller.shouldBlockReturnBallDrag(h.button, h.container), false);
    const state = h.window.live2dManager._live2DPeekState = { active: true, phase: 'peeking', model };
    assert.equal(h.controller.shouldBlockReturnBallDrag(h.button, h.container), true);
    for (const phase of ['revealing', 'hidden', 'idle']) {
        state.phase = phase;
        assert.equal(h.controller.shouldBlockReturnBallDrag(h.button, h.container), false);
    }
    state.phase = 'peeking';
    h.window.live2dManager.currentModel = {};
    assert.equal(h.controller.shouldBlockReturnBallDrag(h.button, h.container), false);
    h.window.live2dManager.currentModel = model;
    h.window.edgePeekLockEnabled = false;
    assert.equal(h.controller.shouldBlockReturnBallDrag(h.button, h.container), false);
});

for (const event of ['blur', 'mouseleave', 'visibilitychange']) {
    for (const faded of [false, true]) {
        test(`${event} clears ${faded ? 'active' : 'pending'} hover fade`, () => {
            const h = harness();
            h.controller.begin({ button: h.button, container: h.container, phase: 'peeking' });
            h.emit('mousemove', { clientX: 10, clientY: 10 });
            assert.equal(h.timers.size, 1);
            if (faded) { h.fireTimers(); assert.equal(h.artClasses.size, 1); }
            h.emit(event); h.fireTimers();
            assert.equal(h.artClasses.size, 0);
            assert.equal(h.timers.size, 0);
            assert.equal(h.controller.isLocked(h.button), true, 'leaving clears fade, not the lock');
        });
    }
}

test('removing a locked return ball releases timers, lock state and observer', () => {
    const h = harness();
    h.controller.begin({ button: h.button, container: h.container, phase: 'peeking' });
    h.emit('mousemove', { clientX: 10, clientY: 10 });
    assert.equal(h.controller.isAnyLocked(), true);
    assert.equal(h.observing, true);
    h.remove();
    assert.equal(h.controller.isActive(), false);
    assert.equal(h.controller.isAnyLocked(), false);
    assert.equal(h.timers.size, 0);
    assert.equal(h.observing, false);
    assert.equal(h.container.style.cursor, 'grab');
});

test('a same-turn detached ball cannot block input before the observer runs', () => {
    const h = harness();
    h.controller.begin({ button: h.button, container: h.container, phase: 'peeking' });
    h.container.isConnected = false;
    assert.equal(h.controller.isAnyLocked(), false);
    assert.equal(h.controller.shouldBlockReturnBallDrag(h.button, h.container), false);
    assert.equal(h.observing, false);
});
