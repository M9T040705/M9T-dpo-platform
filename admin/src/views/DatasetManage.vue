<template>
  <div class="dataset-manage">
    <el-card>
      <div class="toolbar">
        <el-button type="primary" @click="showBuild = true"><el-icon><Plus /></el-icon>构造偏好数据集</el-button>
        <el-button @click="loadDatasets"><el-icon><Refresh /></el-icon>刷新</el-button>
      </div>

      <el-table :data="datasets" v-loading="loading" stripe>
        <el-table-column prop="id" label="ID" width="60" />
        <el-table-column prop="name" label="数据集名称" min-width="160" />
        <el-table-column prop="build_strategy" label="构造策略" width="120">
          <template #default="{ row }"><el-tag size="small">{{ strategyLabel(row.build_strategy) }}</el-tag></template>
        </el-table-column>
        <el-table-column prop="num_samples" label="样本数" width="100" align="center" />
        <el-table-column prop="format" label="格式" width="80" align="center" />
        <el-table-column prop="avg_prompt_length" label="平均Prompt长度" width="130" align="center" />
        <el-table-column prop="avg_chosen_length" label="平均Chosen长度" width="130" align="center" />
        <el-table-column prop="created_at" label="创建时间" width="170" />
        <el-table-column label="操作" width="180" fixed="right">
          <template #default="{ row }">
            <el-dropdown @command="(cmd) => exportDs(row, cmd)">
              <el-button type="primary" link size="small">导出<el-icon><ArrowDown /></el-icon></el-button>
              <template #dropdown>
                <el-dropdown-menu>
                  <el-dropdown-item command="json">JSON</el-dropdown-item>
                  <el-dropdown-item command="jsonl">JSONL</el-dropdown-item>
                  <el-dropdown-item command="trl">TRL格式</el-dropdown-item>
                </el-dropdown-menu>
              </template>
            </el-dropdown>
            <el-button type="danger" link size="small" @click="deleteDs(row)">删除</el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <!-- 构造数据集对话框 -->
    <el-dialog v-model="showBuild" title="构造偏好数据集" width="700px">
      <el-form label-width="100px">
        <el-form-item label="数据集名称" required><el-input v-model="buildForm.name" placeholder="如：医疗偏好数据-v1" /></el-form-item>
        <el-form-item label="构造策略" required>
          <el-select v-model="buildForm.strategy" style="width:100%;">
            <el-option label="打分排序法（多回复按分数排序）" value="score_ranking" />
            <el-option label="规则构造法（自动生成正负样本）" value="rule_based" />
            <el-option label="模型对比法（多模型输出对比）" value="model_compare" />
          </el-select>
        </el-form-item>
        <el-form-item label="输出格式">
          <el-radio-group v-model="buildForm.output_format">
            <el-radio value="json">JSON</el-radio>
            <el-radio value="jsonl">JSONL</el-radio>
            <el-radio value="trl">TRL格式</el-radio>
          </el-radio-group>
        </el-form-item>
        <el-form-item label="描述"><el-input v-model="buildForm.description" type="textarea" :rows="2" /></el-form-item>
        <el-form-item label="输入数据 (JSON)" required>
          <el-input v-model="inputJson" type="textarea" :rows="8" placeholder='[{"prompt":"问题","responses":[{"text":"回复1","score":0.9},{"text":"回复2","score":0.3}]}]' />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="showBuild = false">取消</el-button>
        <el-button type="primary" @click="buildDataset">构造</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { getDatasets, buildDataset, exportDataset, deleteDataset } from '../api'

const datasets = ref([])
const loading = ref(false)
const showBuild = ref(false)
const inputJson = ref('')
const buildForm = ref({ name: '', strategy: 'score_ranking', description: '', output_format: 'json' })

const strategyLabel = (s) => ({ score_ranking: '打分排序', rule_based: '规则构造', model_compare: '模型对比' }[s] || s)

const loadDatasets = async () => {
  loading.value = true
  try { const { data } = await getDatasets({ limit: 50 }); datasets.value = data }
  catch (e) { ElMessage.error('加载失败') } finally { loading.value = false }
}

const buildDataset = async () => {
  if (!buildForm.value.name || !inputJson.value) { ElMessage.warning('请填写名称和输入数据'); return }
  try {
    let inputData
    try { inputData = JSON.parse(inputJson.value) } catch { ElMessage.error('输入数据不是有效JSON'); return }
    await buildDataset({ ...buildForm.value, input_data: inputData })
    ElMessage.success('构造成功')
    showBuild.value = false
    inputJson.value = ''
    loadDatasets()
  } catch (e) { ElMessage.error('构造失败：' + (e.response?.data?.detail || e.message)) }
}

const exportDs = async (row, fmt) => {
  try { await exportDataset(row.id, fmt); ElMessage.success(`已导出为 ${fmt}`) }
  catch (e) { ElMessage.error('导出失败') }
}

const deleteDs = async (row) => {
  try { await ElMessageBox.confirm(`确定删除数据集「${row.name}」吗？`, '确认删除', { type: 'warning' }); await deleteDataset(row.id); ElMessage.success('已删除'); loadDatasets() }
  catch (e) { if (e !== 'cancel') ElMessage.error('删除失败') }
}

onMounted(loadDatasets)
</script>

<style scoped>
.toolbar { display: flex; gap: 12px; margin-bottom: 16px; align-items: center; }
</style>
