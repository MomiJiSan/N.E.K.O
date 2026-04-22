const PLUGIN_ID = 'galgame_bridge';
const RUNS_URL = '/runs';

let latestAgentReply = '暂无交互';

async function callPlugin(entryId, args = {}) {
  const createResp = await fetch(RUNS_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      plugin_id: PLUGIN_ID,
      entry_id: entryId,
      args,
    }),
  });

  if (!createResp.ok) {
    throw new Error(`创建任务失败: HTTP ${createResp.status}`);
  }

  const createData = await createResp.json();
  const runId = createData.run_id || createData.id;
  if (!runId) {
    throw new Error('未获取到 run_id');
  }

  const deadline = Date.now() + 120000;
  while (Date.now() < deadline) {
    await new Promise((resolve) => setTimeout(resolve, 250));
    const pollResp = await fetch(`${RUNS_URL}/${runId}`);
    if (!pollResp.ok) {
      continue;
    }

    const runRecord = await pollResp.json();
    if (runRecord.status === 'succeeded') {
      const exportResp = await fetch(`${RUNS_URL}/${runId}/export`);
      if (!exportResp.ok) {
        return {};
      }
      const exportData = await exportResp.json();
      const items = exportData.items || [];
      const resultItem = items.find((item) => item.type === 'json' && item.json) || items[0];
      const pluginResponse = resultItem ? (resultItem.json || {}) : {};
      if (pluginResponse.success === false || pluginResponse.error) {
        throw new Error(pluginResponse.error?.message || pluginResponse.message || '插件调用失败');
      }
      return pluginResponse.data || {};
    }

    if (['failed', 'canceled', 'timeout'].includes(runRecord.status)) {
      throw new Error(runRecord.error?.message || runRecord.message || runRecord.status);
    }
  }

  throw new Error('插件调用超时');
}

async function safeCall(entryId, args = {}, fallback = {}) {
  try {
    return await callPlugin(entryId, args);
  } catch (error) {
    return { ...fallback, degraded: true, diagnostic: error instanceof Error ? error.message : String(error) };
  }
}

function escapeHtml(text) {
  if (text == null) return '';
  const div = document.createElement('div');
  div.textContent = String(text);
  return div.innerHTML;
}

function setFlash(message, type = 'info') {
  const node = document.getElementById('flashMessage');
  node.hidden = !message;
  node.textContent = message || '';
  node.className = `flash-message ${type}`;
}

function renderGrid(nodeId, rows) {
  const container = document.getElementById(nodeId);
  container.innerHTML = rows.map((row) => `
    <div class="data-row">
      <dt>${escapeHtml(row.label)}</dt>
      <dd>${escapeHtml(row.value)}</dd>
    </div>
  `).join('');
}

function renderStackList(nodeId, items, formatter) {
  const node = document.getElementById(nodeId);
  if (!items.length) {
    node.className = 'stack-list empty-state';
    node.textContent = '暂无数据';
    return;
  }
  node.className = 'stack-list';
  node.innerHTML = items.map(formatter).join('');
}

function renderStatus(status) {
  document.getElementById('summaryText').textContent = status.summary || '无摘要';
  document.getElementById('modeSelect').value = status.mode || 'companion';
  document.getElementById('pushToggle').checked = Boolean(status.push_notifications);
  document.getElementById('bindInput').value = status.bound_game_id || '';
  const memoryReaderRuntime = status.memory_reader_runtime || {};
  const memoryReaderProcess = memoryReaderRuntime.process_name
    ? `${memoryReaderRuntime.process_name} (${memoryReaderRuntime.pid || 0})`
    : '';

  renderGrid('statusGrid', [
    { label: 'connection_state', value: status.connection_state || '' },
    { label: 'active_data_source', value: status.active_data_source || '' },
    { label: 'mode', value: status.mode || '' },
    { label: 'push_notifications', value: String(Boolean(status.push_notifications)) },
    { label: 'bound_game_id', value: status.bound_game_id || '(auto)' },
    { label: 'active_session_id', value: status.active_session_id || '' },
    { label: 'last_seq', value: String(status.last_seq || 0) },
    { label: 'stream_reset_pending', value: String(Boolean(status.stream_reset_pending)) },
    { label: 'available_game_ids', value: (status.available_game_ids || []).join(', ') || '(none)' },
    { label: 'memory_reader_enabled', value: String(Boolean(status.memory_reader_enabled)) },
    { label: 'memory_reader_status', value: memoryReaderRuntime.status || '' },
    { label: 'memory_reader_detail', value: memoryReaderRuntime.detail || '' },
    { label: 'memory_reader_process', value: memoryReaderProcess },
    { label: 'last_error', value: status.last_error?.message || '' },
  ]);
}

function renderSnapshot(snapshot) {
  const state = snapshot.snapshot || {};
  renderGrid('snapshotGrid', [
    { label: 'game_id', value: snapshot.game_id || '' },
    { label: 'session_id', value: snapshot.session_id || '' },
    { label: 'speaker', value: state.speaker || '' },
    { label: 'text', value: state.text || '' },
    { label: 'scene_id', value: state.scene_id || '' },
    { label: 'line_id', value: state.line_id || '' },
    { label: 'route_id', value: state.route_id || '' },
    { label: 'is_menu_open', value: String(Boolean(state.is_menu_open)) },
    { label: 'snapshot_ts', value: snapshot.snapshot_ts || '' },
    { label: 'stale', value: String(Boolean(snapshot.stale)) },
  ]);
}

function renderHistory(history) {
  renderStackList('linesList', history.stable_lines || [], (item) => `
    <article class="list-card">
      <p class="list-kicker">${escapeHtml(item.speaker || '旁白')} · ${escapeHtml(item.scene_id || '')}</p>
      <h3>${escapeHtml(item.line_id || '')}</h3>
      <p>${escapeHtml(item.text || '')}</p>
    </article>
  `);

  renderStackList('choicesList', history.choices || [], (item) => `
    <article class="list-card">
      <p class="list-kicker">${escapeHtml(item.action || '')} · #${escapeHtml(item.index || 0)}</p>
      <h3>${escapeHtml(item.choice_id || '')}</h3>
      <p>${escapeHtml(item.text || '')}</p>
    </article>
  `);

  renderStackList('eventsList', history.events || [], (item) => `
    <article class="list-card compact">
      <p class="list-kicker">seq ${escapeHtml(item.seq || 0)} · ${escapeHtml(item.type || '')}</p>
      <h3>${escapeHtml(item.line_id || item.scene_id || '')}</h3>
      <p>${escapeHtml(JSON.stringify(item.payload || {}))}</p>
    </article>
  `);
}

function renderAgentStatus(payload) {
  document.getElementById('agentReplyText').textContent = latestAgentReply;
  renderGrid('agentStatusGrid', [
    { label: 'status', value: payload.status || 'standby' },
    { label: 'result', value: payload.result || '' },
    { label: 'recent_pushes', value: String((payload.recent_pushes || []).length) },
  ]);

  renderStackList('pushesList', payload.recent_pushes || [], (item) => `
    <article class="list-card compact">
      <p class="list-kicker">${escapeHtml(item.kind || '')} · ${escapeHtml(item.ts || '')}</p>
      <h3>${escapeHtml(item.scene_id || '')}</h3>
      <p>${escapeHtml(item.content || '')}</p>
    </article>
  `);
}

function renderExplain(payload) {
  const node = document.getElementById('explainPanel');
  node.className = 'result-panel';
  node.innerHTML = `
    <p class="list-kicker">${escapeHtml(payload.line_id || '')} · degraded=${escapeHtml(Boolean(payload.degraded))}</p>
    <h3>${escapeHtml(payload.speaker || '旁白')}</h3>
    <p>${escapeHtml(payload.text || '')}</p>
    <p class="result-main">${escapeHtml(payload.explanation || payload.diagnostic || '暂无解释')}</p>
    <p class="result-note">${escapeHtml(payload.diagnostic || '')}</p>
  `;
}

function renderSummary(payload) {
  const node = document.getElementById('summaryPanel');
  node.className = 'result-panel';
  const points = payload.key_points || [];
  node.innerHTML = `
    <p class="list-kicker">${escapeHtml(payload.scene_id || '')} · degraded=${escapeHtml(Boolean(payload.degraded))}</p>
    <p class="result-main">${escapeHtml(payload.summary || payload.diagnostic || '暂无总结')}</p>
    <p class="result-note">${escapeHtml(payload.diagnostic || '')}</p>
    <div class="chip-row">
      ${points.map((item) => `<span class="chip">${escapeHtml(item.type || '')}: ${escapeHtml(item.text || '')}</span>`).join('')}
    </div>
  `;
}

function renderSuggest(payload) {
  const node = document.getElementById('suggestPanel');
  node.className = 'result-panel';
  const choices = payload.choices || [];
  node.innerHTML = `
    <p class="list-kicker">${escapeHtml(payload.scene_id || '')} · degraded=${escapeHtml(Boolean(payload.degraded))}</p>
    <div class="stack-list">
      ${choices.length ? choices.map((item) => `
        <article class="list-card compact">
          <p class="list-kicker">rank ${escapeHtml(item.rank || 0)} · ${escapeHtml(item.choice_id || '')}</p>
          <h3>${escapeHtml(item.text || '')}</h3>
          <p>${escapeHtml(item.reason || '')}</p>
        </article>
      `).join('') : `<div class="empty-inline">${escapeHtml(payload.diagnostic || '暂无建议')}</div>`}
    </div>
  `;
}

async function refreshAll() {
  setFlash('', 'info');
  try {
    const [status, snapshot, history, agentStatus] = await Promise.all([
      callPlugin('galgame_get_status'),
      callPlugin('galgame_get_snapshot'),
      callPlugin('galgame_get_history', { limit: 20, include_events: true }),
      safeCall('galgame_agent_command', { action: 'query_status' }, { action: 'query_status', result: '', status: 'standby', recent_pushes: [] }),
    ]);
    renderStatus(status);
    renderSnapshot(snapshot);
    renderHistory(history);
    renderAgentStatus(agentStatus);

    const currentLineId = snapshot.snapshot?.line_id || '';
    const currentSceneId = snapshot.snapshot?.scene_id || '';
    const hasChoices = Boolean(snapshot.snapshot?.is_menu_open) && Array.isArray(snapshot.snapshot?.choices) && snapshot.snapshot.choices.length > 0;
    const [explain, summary, suggest] = await Promise.all([
      currentLineId
        ? safeCall('galgame_explain_line', { line_id: currentLineId }, { line_id: currentLineId, speaker: '', text: '', explanation: '', evidence: [] })
        : Promise.resolve({ degraded: true, line_id: '', speaker: '', text: '', explanation: '', evidence: [], diagnostic: 'missing line_id' }),
      safeCall('galgame_summarize_scene', currentSceneId ? { scene_id: currentSceneId } : {}, { scene_id: currentSceneId, summary: '', key_points: [] }),
      hasChoices
        ? safeCall('galgame_suggest_choice', {}, { scene_id: currentSceneId, choices: [] })
        : Promise.resolve({ degraded: true, scene_id: currentSceneId, choices: [], diagnostic: 'no visible choices' }),
    ]);

    renderExplain(explain);
    renderSummary(summary);
    renderSuggest(suggest);
  } catch (error) {
    setFlash(error instanceof Error ? error.message : String(error), 'error');
  }
}

async function saveMode() {
  const mode = document.getElementById('modeSelect').value;
  const pushNotifications = document.getElementById('pushToggle').checked;
  try {
    await callPlugin('galgame_set_mode', {
      mode,
      push_notifications: pushNotifications,
    });
    setFlash('模式已保存', 'success');
    await refreshAll();
  } catch (error) {
    setFlash(error instanceof Error ? error.message : String(error), 'error');
  }
}

async function bindGame() {
  const gameId = document.getElementById('bindInput').value.trim();
  try {
    await callPlugin('galgame_bind_game', { game_id: gameId });
    setFlash(gameId ? `已绑定 ${gameId}` : '已恢复自动选择', 'success');
    await refreshAll();
  } catch (error) {
    setFlash(error instanceof Error ? error.message : String(error), 'error');
  }
}

async function setStandby(standby) {
  try {
    const payload = await callPlugin('galgame_agent_command', {
      action: 'set_standby',
      standby,
    });
    latestAgentReply = payload.result || latestAgentReply;
    setFlash(standby ? '已切换到待机' : '已恢复活跃', 'success');
    await refreshAll();
  } catch (error) {
    setFlash(error instanceof Error ? error.message : String(error), 'error');
  }
}

async function askAgent(action) {
  const prompt = document.getElementById('agentPromptInput').value.trim();
  if (!prompt) {
    setFlash('请输入要发送给 Agent 的文本', 'error');
    return;
  }

  try {
    const payload = await callPlugin('galgame_agent_command', action === 'query_context'
      ? { action, context_query: prompt }
      : { action, message: prompt });
    latestAgentReply = payload.result || 'Agent 未返回文本';
    setFlash('Agent 已响应', 'success');
    await refreshAll();
  } catch (error) {
    setFlash(error instanceof Error ? error.message : String(error), 'error');
  }
}

document.getElementById('refreshBtn').addEventListener('click', refreshAll);
document.getElementById('saveModeBtn').addEventListener('click', saveMode);
document.getElementById('bindBtn').addEventListener('click', bindGame);
document.getElementById('clearBindBtn').addEventListener('click', async () => {
  document.getElementById('bindInput').value = '';
  await bindGame();
});
document.getElementById('standbyOnBtn').addEventListener('click', () => setStandby(true));
document.getElementById('standbyOffBtn').addEventListener('click', () => setStandby(false));
document.getElementById('queryContextBtn').addEventListener('click', () => askAgent('query_context'));
document.getElementById('sendMessageBtn').addEventListener('click', () => askAgent('send_message'));

refreshAll();
