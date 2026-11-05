const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const root = path.resolve(__dirname, '../..');
const read = name => fs.readFileSync(path.join(root, 'static/app', name), 'utf8');
const socket = read('app-websocket.js');
function between(source, start, end) {
    const first = source.indexOf(start);
    const last = source.indexOf(end, first + start.length);
    assert.ok(first >= 0 && last > first, `missing production function ${start}`);
    return source.slice(first, last);
}

// The complete production adapter, its real text splitter/queue/pending-host
// paths, and the actual WebSocket text-turn finalizer. Only DOM/host and time
// are controlled: no copied completion algorithm and no TTS completion signal.
function harness({ mounted = true } = {}) {
    let now = 10000, nextTimer = 0;
    const timers = new Map(), messages = [], updates = [], events = [];
    const host = {
        appendMessage(message) { messages.push({ ...message }); },
        updateMessage(id, patch) {
            updates.push({ id, ...patch });
            const item = messages.find(message => message.id === id);
            if (item) Object.assign(item, patch);
        },
        removeMessage(id) { const i = messages.findIndex(message => message.id === id); if (i >= 0) messages.splice(i, 1); },
        getState() { return { messages }; },
    };
    const window = {
        appState: { mergeMessagesEnabled: false }, appChat: {},
        reactChatWindowHost: mounted ? host : null,
        location: { pathname: '/chat_full' },
        addEventListener() {}, dispatchEvent() {}, getCurrentTimeString() { return '12:00:00'; },
        nekoReplyTail: { changed() {} },
    };
    const context = vm.createContext({
        window, console: { log() {}, warn() {} },
        document: { body: { classList: { add() {} } }, readyState: 'loading',
            getElementById() { return null; }, addEventListener() {} },
        localStorage: { getItem() { return null; } },
        CustomEvent: class { constructor(type, options) { this.type = type; this.detail = options.detail; } },
        Date: class extends Date { static now() { return now; } },
        setTimeout(fn, delay = 0) { const id = ++nextTimer; timers.set(id, { fn, at: now + delay }); return id; },
        clearTimeout(id) { timers.delete(id); },
        setInterval(fn, delay) { const id = ++nextTimer; timers.set(id, { fn, at: now + delay, delay }); return id; },
        clearInterval(id) { timers.delete(id); },
        resolveAssistantLifecycleTurnId() { return window._nekoAssistantTurnId; },
        emitAssistantLifecycleEvent(name, detail) {
            events.push({ name, detail });
            if (window.onTextEnding) window.onTextEnding(name, detail);
        },
        logAssistantLifecycle() {},
    });
    vm.runInContext(read('app-chat-text-utils.js'), context);
    vm.runInContext(read('app-chat-adapter.js'), context);
    vm.runInContext(between(socket, '    function flushRealisticBufferOnTurnEnd()', '    function clearPendingUserActivityCancel()'), context);
    vm.runInContext('var S = window.appState;\n' + between(socket,
        '    function stopAssistantTextOutputOnSessionEnd(',
        "    window.addEventListener('neko-assistant-turn-start'"), context);
    // The adapter's unrelated initial avatar refresh timers do not form part of
    // a response. Pending-host retry timers created below remain controlled.
    timers.clear();
    const api = {
        window, messages, updates, events, context,
        append(text, fresh = false, turn = 'turn-A') {
            window._nekoAssistantTurnId = turn;
            return window.appendMessage(text, 'gemini', fresh);
        },
        end() { return vm.runInContext('flushRealisticBufferOnTurnEnd()', context); },
        cancel() { return vm.runInContext("stopAssistantTextOutputOnSessionEnd('test-session-end')", context); },
        mount() { window.reactChatWindowHost = host; window._tryFlushPendingHostMessages(); },
        async advance(ms) {
            now += ms;
            const due = [...timers.entries()].filter(([, value]) => value.at <= now);
            for (const [id, value] of due) {
                if (!timers.has(id)) continue;
                if (value.delay) value.at = now + value.delay; else timers.delete(id);
                value.fn();
            }
            for (let i = 0; i < 6; i++) await Promise.resolve();
        },
        async drain() { for (let i = 0; i < 8; i++) await api.advance(2000); },
    };
    return api;
}
const statuses = h => h.messages.map(message => message.status);
const texts = h => h.messages.map(message => message.blocks.map(block => block.text || '').join(''));

test('short reply created by text turn-end is sent without any audio event', async () => {
    const h = harness();
    h.append('短回复尾段', true);
    assert.equal(h.messages.length, 0);
    h.end();
    await h.drain();
    assert.deepEqual(texts(h), ['短回复尾段']);
    assert.deepEqual(statuses(h), ['sent']);
});

test('text turn-end seals every existing and asynchronously queued sentence', async () => {
    const h = harness();
    h.append('第一句。第二句。第三句。尾段', true);
    assert.equal(h.messages.length, 1);
    h.end();
    await h.drain();
    assert.equal(texts(h).join(''), '第一句。第二句。第三句。尾段');
    assert.ok(h.messages.length >= 3, 'realistic pacing must exercise delayed bubbles');
    assert.ok(statuses(h).every(status => status === 'sent'));
});

test('all earlier already-rendered bubbles also become sent', async () => {
    const h = harness();
    h.append('第一句。\n', true);
    await h.drain();
    h.append('第二句。\n');
    await h.drain();
    h.append('第三句。\n');
    await h.drain();
    assert.equal(h.messages.length, 3);
    h.end();
    assert.deepEqual(statuses(h), ['sent', 'sent', 'sent']);
});

test('same-dialog next response drains completed old queue but stays streaming itself', async () => {
    const h = harness();
    h.append('旧一。旧二。旧尾', true, 'same-dialog');
    h.end();
    h.append('新段。\n', true, 'same-dialog');
    await h.drain();
    assert.deepEqual(texts(h), ['旧一。', '旧二。', '旧尾', '新段。']);
    assert.deepEqual(statuses(h), ['sent', 'sent', 'sent', 'streaming']);
    h.end();
    assert.deepEqual(statuses(h), ['sent', 'sent', 'sent', 'sent']);
});

test('late continuation creates a distinct segment and is not pre-completed', async () => {
    const h = harness();
    h.append('旧尾', true);
    h.end();
    h.append('续句。\n', false);
    await h.drain();
    assert.deepEqual(texts(h), ['旧尾', '续句。']);
    assert.deepEqual(statuses(h), ['sent', 'streaming']);
});

test('pending React host receives completed trailing text with sent status', async () => {
    const h = harness({ mounted: false });
    h.append('挂载前一。挂载前尾', true);
    h.end();
    await h.drain();
    assert.equal(h.messages.length, 0);
    h.mount();
    await h.advance(200);
    assert.deepEqual(texts(h), ['挂载前一。', '挂载前尾']);
    assert.deepEqual(statuses(h), ['sent', 'sent']);
});

test('version/owner invalidation cannot publish cancelled queue into the next response', async () => {
    const h = harness();
    h.append('取消一。取消二。取消尾', true);
    h.end();
    // Run the actual cancellation cleanup while its paced queue awaits time.
    h.cancel();
    h.append('后继。\n', true, 'turn-B');
    await h.drain();
    assert.deepEqual(texts(h), ['取消一。', '后继。']);
    assert.equal(h.messages.at(-1).status, 'streaming');
});

test('synchronous ending listener cannot seal or drain its successor segment', async () => {
    const h = harness();
    h.append('旧尾段', true);
    h.window.onTextEnding = () => h.append('重入后继。\n', true, 'turn-B');
    h.end();
    await h.drain();
    assert.deepEqual(texts(h), ['旧尾段', '重入后继。']);
    assert.deepEqual(statuses(h), ['sent', 'streaming']);
});

test('merge and structured replies settle only their own captured assistant refs', () => {
    const h = harness();
    h.window.mergeMessagesEnabled = true;
    h.append('合并回复', true);
    h.messages.push({id:'unrelated',role:'assistant',blocks:[{type:'text',text:'旁路'}],status:'streaming'});
    h.end();
    assert.deepEqual(statuses(h), ['sent', 'streaming']);
    h.window.mergeMessagesEnabled = false;
    h.append('# 富文本标题', true, 'turn-B');
    h.end();
    assert.equal(h.messages.at(-1).status, 'sent');
    assert.equal(h.messages.find(message => message.id === 'unrelated').status, 'streaming');
});
