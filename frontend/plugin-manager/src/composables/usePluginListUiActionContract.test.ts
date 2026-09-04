// @vitest-environment happy-dom

import { beforeEach, describe, expect, it, vi } from 'vitest'
import { ElMessage } from 'element-plus'
import { deletePlugin, launchLocalApp } from '@/api/plugins'
import { usePluginListContextActions } from './usePluginListContextActions'
import type { PluginListAction, PluginMeta } from '@/types/api'

const mocks = vi.hoisted(() => ({
  openExternalUrl: vi.fn(),
  routerPush: vi.fn(),
}))

vi.mock('vue-router', () => ({
  useRouter: () => ({
    push: mocks.routerPush,
  }),
}))

vi.mock('vue-i18n', () => ({
  useI18n: () => ({
    locale: { value: 'zh-CN' },
    t: (key: string) => key,
  }),
}))

vi.mock('element-plus', () => ({
  ElMessage: {
    error: vi.fn(),
    success: vi.fn(),
    warning: vi.fn(),
  },
  ElMessageBox: { confirm: vi.fn() },
}))

vi.mock('@/stores/plugin', () => ({
  usePluginStore: () => ({
    start: vi.fn(),
    stop: vi.fn(),
    reload: vi.fn(),
  }),
}))

vi.mock('@/api/plugins', () => ({ deletePlugin: vi.fn(), launchLocalApp: vi.fn() }))
vi.mock('@/api/pluginCli', () => ({ buildPluginCli: vi.fn() }))
vi.mock('@/utils/openExternal', () => ({
  openExternalUrl: mocks.openExternalUrl,
}))

function makePlugin(action: PluginListAction): PluginMeta {
  return {
    id: 'generic-plugin',
    name: 'Generic plugin',
    description: 'Generic plugin description',
    version: '1.0.0',
    status: 'running',
    list_actions: [action],
  }
}

beforeEach(() => {
  vi.clearAllMocks()
})

describe('plugin list UI action navigation contract', () => {
  it('launches only the host-declared app id and disables missing installations', async () => {
    const plugin: PluginMeta = {
      id: 'study_companion',
      name: 'Study Companion',
      description: 'Study Companion',
      version: '1.0.0',
      status: 'running',
      local_app: {
        app_id: 'knowledge_dungeon',
        title: 'Knowledge Dungeon',
        available: true,
      },
    }
    const { buildActions, executeAction } = usePluginListContextActions()
    const action = buildActions(plugin).find(
      (candidate) => candidate.id === 'launch_local_app',
    )

    expect(action).toMatchObject({
      kind: 'builtin',
      disabled: false,
      requires_running: true,
    })
    await executeAction(action!, plugin)
    expect(launchLocalApp).toHaveBeenCalledWith('knowledge_dungeon')
    expect(ElMessage.success).toHaveBeenCalled()

    plugin.local_app!.available = false
    const disabled = buildActions(plugin).find(
      (candidate) => candidate.id === 'launch_local_app',
    )
    expect(disabled?.disabled).toBe(true)
  })

  it('uses openExternalUrl for the default new-tab path', async () => {
    const plugin = makePlugin({
      id: 'open_ui',
      kind: 'ui',
      target: '/plugin/generic/ui/',
    })
    const { buildActions, executeAction } = usePluginListContextActions()
    const action = buildActions(plugin).find((candidate) => candidate.id === 'open_ui')
    expect(action).toBeDefined()

    await executeAction(action!, plugin)

    expect(mocks.openExternalUrl).toHaveBeenCalledWith('/plugin/generic/ui/')
    expect(mocks.routerPush).not.toHaveBeenCalled()
  })

  it('uses current-window navigation for same_tab', async () => {
    const open = vi.spyOn(window, 'open').mockImplementation(() => null)
    const plugin = makePlugin({
      id: 'open_ui',
      kind: 'ui',
      target: '/plugin/generic/ui/',
      open_in: 'same_tab',
    })
    const { buildActions, executeAction } = usePluginListContextActions()
    const action = buildActions(plugin).find((candidate) => candidate.id === 'open_ui')

    await executeAction(action!, plugin)

    expect(open).toHaveBeenCalledWith('/plugin/generic/ui/', '_self')
    expect(mocks.openExternalUrl).not.toHaveBeenCalled()
  })

  it('warns when deletion restores a builtin that cannot restart', async () => {
    vi.mocked(deletePlugin).mockResolvedValue({
      success: true,
      plugin_id: 'generic-plugin',
      plugin_dir: '/plugins/generic-plugin',
      deleted_from_disk: true,
      restored_builtin: true,
      restored_builtin_started: false,
      restored_builtin_restart_error: {
        code: 'PLUGIN_BUILTIN_RESTORE_START_FAILED',
        message: 'startup failed',
        error_type: 'RuntimeError',
      },
      message: 'Plugin deleted successfully',
    })
    const plugin = makePlugin({ id: 'open_detail', kind: 'builtin' })
    const { buildActions, executeAction } = usePluginListContextActions()
    const action = buildActions(plugin).find((candidate) => candidate.id === 'delete')

    await executeAction(action!, plugin)

    expect(ElMessage.warning).toHaveBeenCalledWith(
      'messages.pluginDeletedBuiltinRestartFailed',
    )
    expect(ElMessage.success).not.toHaveBeenCalledWith('messages.pluginDeleted')
  })
})
