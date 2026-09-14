<template>
  <div class="training-jobs">
    <el-card>
      <div class="toolbar">
        <el-select v-model="statusFilter" placeholder="全部状态" clearable style="width:140px;" @change="loadJobs">
          <el-option label="待启动" value="pending" />
          <el-option label="运行中" value="running" />
          <el-option label="已完成" value="completed" />
          <el-option label="失败" value="failed" />
        </el-select>
        <el-button type="primary" @click="showCreate = true"><el-icon><Plus /></el-icon>新建训练任务</el-button>
        <el-button @click="loadJobs"><el-icon><Refresh /></el-icon>刷新</el-button>
      </div>

      <el-table :data="jobs" v-loading="loading" stripe>
        <el-table-column prop="id" label="ID" width="60" />
        <el-table-column prop="job_name" label="任务名称" min-width="160" />
        <el-table-column prop="status" label="状态" width="100">
          <template #default="{ row }"><el-tag :type="statusType(row.status)" size="small">{{ row.status }}</el-tag></template>
        </el-table-column>
        <el-table-column prop="model_name" label="基座模型" width="140" show-overflow-tooltip />
        <el-table-column prop="final_loss" label="最终损失" width="100">
          <template #default="{ row }">{{ row.final_loss !== null ? row.final_loss.toFixed(4) : '-' }}</template>
        </el-table-column>
        <el-table-column prop="final_reward_margin" label="奖励边际" width="100">
          <template #default="{ row }">{{ row.final_reward_margin !== null ? row.final_reward_margin.toFixed(4) : '-' }}</template>
        </el-table-column>
        <el-table-column prop="epochs_trained" label="轮数" width="70" align="center" />
        <el-table-column prop="created_at" label="创建时间" width="170" />
        <el-table-column label="操作" width="200" fixed="right">
          <template #default="{ row }">
            <el-button type="success" link size="small" @click="viewMetrics(row)" :disabled="row.steps_trained === 0">指标</el-button>
            <el-button type="primary" link size="small" @click="startJob(row)" :disabled="row.status === 'running'">启动</el-button>
            <el-button type="warning" link size="small" @click="stopJob(row)" :disabled="row.status !== 'running'">停止</el-button>
            <el-button type="danger" link size="small" @click="deleteJob(row)">删除</el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <!-- 新建任务对话框 -->
    <el-dialog v-model="showCreate" title="新建训练任务" width="600px">
      <el-form label-width="100px">
        <el-form-item label="任务名称" required><el-input v-model="form.job_name" placeholder="如：医疗DPO微调-v1" /></el-form-item>
        <el-form-item label="基座模型"><el-input v-model="form.model_name" placeholder="如：Qwen2.5-7B / meta-llama/Llama-3-8B" /></el-form-item>
        <el-form-item label="数据集路径" required><el-input v-model="form.dataset_path" placeholder="偏好数据集JSON文件路径" /></el-form-item>
        <el-form-item label="实现方式">
          <el-radio-group v-model="form.implementation">
            <el-radio value="custom">从零实现</el-radio>
            <el-radio value="trl">TRL 库</el-radio>
          </el-radio-group>
        </el-form-item>
        <el-divider content-position="left">训练参数</el-divider>
        <el-row :gutter="20">
          <el-col :span="12"><el-form-item label="β 温度"><el-input-number v-model="form.config.beta" :step="0.05" :min="0.01" :max="1" /></el-form-item></el-col>
          <el-col :span="12"><el-form-item label="学习率"><el-input-number v-model="form.config.learning_rate" :step="1e-5" :min="1e-6" :max="1e-3" /></el-form-item></el-col>
          <el-col :span="12"><el-form-item label="训练轮数"><el-input-number v-model="form.config.num_epochs" :min="1" :max="20" /></el-form-item></el-col>
          <el-col :span="12"><el-form-item label="批次大小"><el-input-number v-model="form.config.batch_size" :min="1" :max="64" /></el-form-item></el-col>
        </el-row>
      </el-form>
      <template #footer>
        <el-button @click="showCreate = false">取消</el-button>
        <el-button type="primary" @click="createJob">创建</el-button>
      </template>
    </el-dialog>

    <!-- 指标曲线对话框 -->
    <el-dialog v-model="showMetrics" title="训练指标曲线" width="800px">
      <div ref="chartRef" style="height:400px;"></div>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, onMounted, nextTick } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import * as echarts from 'echarts'
import { getTrainingJobs, createTrainingJob, startTrainingJob, stopTrainingJob, getJobMetrics, deleteTrainingJob } from '../api'

const jobs = ref([])
const loading = ref(false)
const statusFilter = ref('')
const showCreate = ref(false)
const showMetrics = ref(false)
const chartRef = ref(null)
let chart = null

const form = ref({
  job_name: '', model_name: '', dataset_path: '', implementation: 'custom',
  config: { beta: 0.1, learning_rate: 5e-5, num_epochs: 3, batch_size: 4, max_length: 1024 }
})

const statusType = (s) => ({ pending: 'info', running: 'warning', completed: 'success', failed: 'danger' }[s] || '')

const loadJobs = async () => {
  loading.value = true
  try {
    const { data } = await getTrainingJobs({ status: statusFilter.value || undefined, limit: 50 })
    jobs.value = data
  } catch (e) { ElMessage.error('加载失败') } finally { loading.value = false }
}

const createJob = async () => {
  if (!form.value.job_name || !form.value.dataset_path) { ElMessage.warning('请填写任务名称和数据集路径'); return }
  try {
    await createTrainingJob(form.value)
    ElMessage.success('创建成功')
    showCreate.value = false
    loadJobs()
  } catch (e) { ElMessage.error('创建失败') }
}

const startJob = async (row) => {
  try { await startTrainingJob(row.id); ElMessage.success('任务已启动'); loadJobs() }
  catch (e) { ElMessage.error('启动失败') }
}

const stopJob = async (row) => {
  try { await ElMessageBox.confirm(`确定停止任务「${row.job_name}」吗？`, '确认停止', { type: 'warning' }); await stopTrainingJob(row.id); ElMessage.success('已停止'); loadJobs() }
  catch (e) { if (e !== 'cancel') ElMessage.error('停止失败') }
}

const deleteJob = async (row) => {
  try { await ElMessageBox.confirm(`确定删除任务「${row.job_name}」吗？`, '确认删除', { type: 'warning' }); await deleteTrainingJob(row.id); ElMessage.success('已删除'); loadJobs() }
  catch (e) { if (e !== 'cancel') ElMessage.error('删除失败') }
}

const viewMetrics = async (row) => {
  showMetrics.value = true
  await nextTick()
  const { data } = await getJobMetrics(row.id, { limit: 500 })
  if (!chart) { chart = echarts.init(chartRef.value) }
  chart.setOption({
    tooltip: { trigger: 'axis' },
    legend: { data: ['Loss', '奖励边际'] },
    xAxis: { type: 'category', data: data.map(m => m.step) },
    yAxis: [{ type: 'value', name: 'Loss' }, { type: 'value', name: '奖励边际' }],
    series: [
      { name: 'Loss', type: 'line', data: data.map(m => m.loss), smooth: true },
      { name: '奖励边际', type: 'line', yAxisIndex: 1, data: data.map(m => m.reward_margin), smooth: true }
    ]
  })
}

onMounted(loadJobs)
</script>

<style scoped>
.toolbar { display: flex; gap: 12px; margin-bottom: 16px; align-items: center; }
</style>
