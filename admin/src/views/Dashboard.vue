<template>
  <div class="dashboard">
    <el-row :gutter="20">
      <el-col :span="6"><el-card class="stat-card"><div class="stat-icon" style="background:#409eff;"><el-icon :size="28"><VideoPlay /></el-icon></div><div class="stat-info"><div class="stat-value">{{ stats.total_jobs }}</div><div class="stat-label">训练任务总数</div></div></el-card></el-col>
      <el-col :span="6"><el-card class="stat-card"><div class="stat-icon" style="background:#67c23a;"><el-icon :size="28"><CircleCheck /></el-icon></div><div class="stat-info"><div class="stat-value">{{ stats.completed_jobs }}</div><div class="stat-label">已完成任务</div></div></el-card></el-col>
      <el-col :span="6"><el-card class="stat-card"><div class="stat-icon" style="background:#e6a23c;"><el-icon :size="28"><Folder /></el-icon></div><div class="stat-info"><div class="stat-value">{{ stats.total_datasets }}</div><div class="stat-label">偏好数据集</div></div></el-card></el-col>
      <el-col :span="6"><el-card class="stat-card"><div class="stat-icon" style="background:#f56c6c;"><el-icon :size="28"><TrendCharts /></el-icon></div><div class="stat-info"><div class="stat-value">{{ stats.avg_loss }}</div><div class="stat-label">平均最终损失</div></div></el-card></el-col>
    </el-row>

    <el-row :gutter="20" style="margin-top:20px;">
      <el-col :span="12">
        <el-card>
          <template #header><span style="font-weight:600;">DPO 核心原理</span></template>
          <div class="formula-box">
            <p><strong>L_DPO</strong> = -E[log σ(β (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))]</p>
            <el-divider />
            <p>• <strong>y_w</strong>：chosen（优选回复）</p>
            <p>• <strong>y_l</strong>：rejected（劣选回复）</p>
            <p>• <strong>β</strong>：温度系数（控制与参考模型偏离程度）</p>
            <p>• <strong>π</strong>：策略模型（待训练）</p>
            <p>• <strong>π_ref</strong>：参考模型（冻结）</p>
          </div>
        </el-card>
      </el-col>
      <el-col :span="12">
        <el-card>
          <template #header><span style="font-weight:600;">训练流程</span></template>
          <el-steps direction="vertical" :active="4" finish-status="success">
            <el-step title="偏好数据构造" description="打分排序 / 规则构造 / 模型对比" />
            <el-step title="数据预处理" description="清洗 / 去重 / 质量评分 / 格式化" />
            <el-step title="DPO 训练" description="从零实现 / TRL 实现，损失曲线实时监控" />
            <el-step title="效果评测" description="损失对比 / 奖励边际 / 准确率 / 人工评测" />
          </el-steps>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { getTrainingJobs, getDatasets } from '../api'

const stats = ref({ total_jobs: 0, completed_jobs: 0, total_datasets: 0, avg_loss: '0.00' })

onMounted(async () => {
  try {
    const [jobsRes, dsRes] = await Promise.all([getTrainingJobs({ limit: 100 }), getDatasets({ limit: 100 })])
    const jobs = jobsRes.data
    const completed = jobs.filter(j => j.status === 'completed')
    const losses = completed.map(j => j.final_loss).filter(l => l !== null)
    stats.value = {
      total_jobs: jobs.length,
      completed_jobs: completed.length,
      total_datasets: dsRes.data.length,
      avg_loss: losses.length ? (losses.reduce((a,b) => a+b, 0) / losses.length).toFixed(4) : '0.00'
    }
  } catch (e) { console.error(e) }
})
</script>

<style scoped>
.stat-card { display: flex; align-items: center; gap: 16px; }
.stat-icon { width: 56px; height: 56px; border-radius: 12px; display: flex; align-items: center; justify-content: center; color: #fff; }
.stat-value { font-size: 28px; font-weight: 700; color: #303133; }
.stat-label { font-size: 13px; color: #909399; margin-top: 2px; }
.formula-box { font-size: 13px; line-height: 1.8; color: #606266; }
.formula-box p { margin: 4px 0; }
</style>
