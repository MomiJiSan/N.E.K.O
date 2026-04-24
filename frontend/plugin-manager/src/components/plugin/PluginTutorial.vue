<template>
  <div class="plugin-tutorial">
    <div v-if="pluginId === 'galgame_plugin'" class="tutorial-content">
      <el-alert
        title="Galgame 游玩助手使用教程"
        type="info"
        :closable="false"
        description="本教程帮助你快速上手 Galgame 游玩助手的核心功能。"
        class="tutorial-alert"
      />

      <el-collapse v-model="activeNames">
        <el-collapse-item title="1. 插件简介与运行顺序" name="1">
          <p>Galgame 游玩助手的正式运行顺序固定为：</p>
          <el-tag type="primary" size="small">Bridge SDK</el-tag>
          <el-text class="mx-1">&gt;</el-text>
          <el-tag type="success" size="small">OCR Reader</el-tag>
          <el-text class="mx-1">&gt;</el-text>
          <el-tag type="warning" size="small">Memory Reader</el-tag>
          <p class="mt-2">
            OCR Reader 内部现已切换为 <strong>RapidOCR 主后端</strong> + <strong>Tesseract 兼容兜底</strong>。当 RapidOCR 不可用时，会自动回退到 Tesseract。
          </p>
        </el-collapse-item>

        <el-collapse-item title="2. 快速开始" name="2">
          <ol>
            <li>确保已安装 RapidOCR 或 Tesseract（在"打开界面"标签页可一键安装）。</li>
            <li>启动你的 Galgame 游戏。</li>
            <li>在"打开界面"标签页中，点击<strong>刷新窗口</strong>，然后点击<strong>选择识别窗口</strong>锁定游戏窗口。</li>
            <li>选择运行模式：
              <ul>
                <li><strong>静默</strong>：只记录，不推送通知</li>
                <li><strong>伴读</strong>：自动识别台词并给出解释</li>
                <li><strong>主动陪伴</strong>：在选项时给出建议</li>
              </ul>
            </li>
            <li>点击<strong>保存模式</strong>即可开始游玩。</li>
          </ol>
        </el-collapse-item>

        <el-collapse-item title="3. OCR 目标窗口选择" name="3">
          <p>OCR 目标窗口区域会显示当前锁定的游戏窗口。你可以：</p>
          <ul>
            <li>点击<strong>选择识别窗口</strong>按钮，从弹窗中选择要识别的游戏窗口。</li>
            <li>点击<strong>恢复自动</strong>让插件自动选择前台窗口。</li>
            <li>点击<strong>查看排除窗口与原因</strong>查看为什么某些窗口不可用。</li>
          </ul>
          <el-alert
            title="提示"
            type="warning"
            :closable="false"
            class="mt-2"
            show-icon
          >
            N.E.K.O 自身界面会被自动排除，只允许锁定可用游戏窗口。
          </el-alert>
        </el-collapse-item>

        <el-collapse-item title="4. OCR 截图校准" name="4">
          <p>按进程名保存窗口裁剪比例，避免识别到无关区域：</p>
          <ul>
            <li><strong>通用区域</strong>：适用于大部分游戏，保存通用的截图范围。</li>
            <li><strong>对白区</strong>：专门裁剪台词对话框区域（如哀鸿等游戏）。</li>
            <li><strong>菜单区</strong>：专门裁剪选项菜单区域。</li>
          </ul>
          <p class="mt-2">填写左侧、右侧、顶部、底部裁剪比例后，点击<strong>保存校准</strong>即可。</p>
        </el-collapse-item>

        <el-collapse-item title="5. 游戏 Agent 交互" name="5">
          <p>在"打开界面"标签页下方，你可以与游戏 Agent 进行交互：</p>
          <ul>
            <li><strong>查询上下文</strong>：询问 Agent 当前游戏进行到什么剧情。</li>
            <li><strong>发送消息</strong>：给 Agent 发送自定义消息。</li>
          </ul>
          <p class="mt-2">Agent 会自动分析当前台词、场景，并在需要的时候给出选项建议。</p>
        </el-collapse-item>

        <el-collapse-item title="6. 安装 OCR 后端" name="6">
          <p>如果 RapidOCR 或 Tesseract 未安装，可以在"打开界面"中看到安装提示：</p>
          <ul>
            <li><strong>一键安装 RapidOCR</strong>：安装插件隔离的 RapidOCR 运行时（推荐，Windows 平台）。</li>
            <li><strong>一键安装 Tesseract</strong>：安装兼容兜底方案，包含 chi_sim + jpn + eng 语言包。</li>
          </ul>
        </el-collapse-item>

        <el-collapse-item title="7. 常见问题" name="7">
          <el-descriptions :column="1" border>
            <el-descriptions-item label="OCR 不识别文字">
              检查是否正确锁定了游戏窗口；尝试调整截图校准的裁剪范围。
            </el-descriptions-item>
            <el-descriptions-item label="Agent 没有反应">
              确认插件已启动且模式不是"静默"；检查是否有最近错误提示。
            </el-descriptions-item>
            <el-descriptions-item label="安装失败">
              确保网络连接正常；如果 RapidOCR 安装失败，可以先用 Tesseract 作为兜底。
            </el-descriptions-item>
            <el-descriptions-item label="窗口列表为空">
              确保游戏已经启动；点击"刷新窗口"重新扫描。
            </el-descriptions-item>
          </el-descriptions>
        </el-collapse-item>
      </el-collapse>
    </div>

    <div v-else class="tutorial-empty">
      <el-empty :description="$t('plugins.noTutorial')" />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'

const props = defineProps<{
  pluginId: string
}>()

const activeNames = ref(['1', '2'])
</script>

<style scoped>
.plugin-tutorial {
  padding: 20px 0;
}

.tutorial-alert {
  margin-bottom: 20px;
}

.tutorial-content ol,
.tutorial-content ul {
  padding-left: 20px;
  line-height: 1.8;
  color: var(--el-text-color-regular);
}

.tutorial-content p {
  line-height: 1.8;
  color: var(--el-text-color-regular);
}

.mt-2 {
  margin-top: 12px;
}

.mx-1 {
  margin: 0 4px;
}

.tutorial-empty {
  padding: 40px 0;
}
</style>
