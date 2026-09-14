<template>
  <div class="loss-calculator">
    <el-row :gutter="20">
      <el-col :span="12">
        <el-card>
          <template #header><span style="font-weight:600;">DPO 损失计算器</span></template>
          <el-form label-width="140px">
            <el-form-item label="β 温度系数">
              <el-input-number v-model="beta" :step="0.05" :min="0.01" :max="1" />
            </el-form-item>
            <el-divider content-position="left">策略模型 log 概率</el-divider>
            <el-form-item label="Chosen (policy)">
              <el-input v-model="policyChosen" type="textarea" :rows="2" placeholder="如：-2.5, -3.1, -1.8" />
            </el-form-item>
            <el-form-item label="Rejected (policy)">
              <el-input v-model="policyRejected" type="textarea" :rows="2" placeholder="如：-4.2, -3.8, -5.1" />
            </el-form-item>
            <el-divider content-position="left">参考模型 log 概率</el-divider>
            <el-form-item label="Chosen (ref)">
              <el-input v-model="refChosen" type="textarea" :rows="2" placeholder="如：-2.8, -3.0, -2.0" />
            </el-form-item>
            <el-form-item label="Rejected (ref)">
              <el-input v-model="refRejected" type="textarea" :rows="2" placeholder="如：-4.0, -3.5, -4.8" />
            </el-form-item>
            <el-form-item>
              <el-button type="primary" @click="compute">计算损失</el-button>
              <el-button @click="fillExample">填充示例</el-button>
            </el-form-item>
          </el-form>
        </el-card>
      </el-col>
      <el-col :span="12">
        <el-card v-if="result">
          <template #header><span style="font-weight:600;">计算结果</span></template>
          <el-descriptions :column="1" border>
            <el-descriptions-item label="DPO 损失"><span style="color:#f56c6c;font-weight:600;font-size:18px;">{{ result.loss.toFixed(6) }}</span></el-descriptions-item>
            <el-descriptions-item label="Chosen 隐式奖励">{{ result.chosen_rewards.toFixed(4) }}</el-descriptions-item>
            <el-descriptions-item label="Rejected 隐式奖励">{{ result.rejected_rewards.toFixed(4) }}</el-descriptions-item>
            <el-descriptions-item label="奖励边际 (Chosen - Rejected)"><span style="color:#67c23a;font-weight:600;">{{ result.reward_margin.toFixed(4) }}</span></el-descriptions-item>
            <el-descriptions-item label="准确率 (Chosen > Rejected)">{{ (result.accuracy * 100).toFixed(1) }}%</el-descriptions-item>
            <el-descriptions-item label="Chosen log ratio">{{ result.chosen_logratios.toFixed(4) }}</el-descriptions-item>
            <el-descriptions-item label="Rejected log ratio">{{ result.rejected_logratios.toFixed(4) }}</el-descriptions-item>
          </el-descriptions>
          <el-alert style="margin-top:16px;" type="info" :closable="false">
            <template #title>损失越低，说明模型越能区分 chosen 和 rejected；奖励边际为正且越大，说明偏好对齐效果越好。</template>
          </el-alert>
        </el-card>
        <el-card v-else>
          <el-empty description="输入数据后点击计算" />
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { ElMessage } from 'element-plus'
import { computeLoss } from '../api'

const beta = ref(0.1)
const policyChosen = ref('')
const policyRejected = ref('')
const refChosen = ref('')
const refRejected = ref('')
const result = ref(null)

const fillExample = () => {
  policyChosen.value = '-2.5, -3.1, -1.8'
  policyRejected.value = '-4.2, -3.8, -5.1'
  refChosen.value = '-2.8, -3.0, -2.0'
  refRejected.value = '-4.0, -3.5, -4.8'
}

const parseArray = (s) => s.split(',').map(x => parseFloat(x.trim())).filter(x => !isNaN(x))

const compute = async () => {
  const pc = parseArray(policyChosen.value)
  const pr = parseArray(policyRejected.value)
  const rc = parseArray(refChosen.value)
  const rr = parseArray(refRejected.value)
  if (!pc.length || !pr.length || !rc.length || !rr.length) { ElMessage.warning('请填写所有 log 概率'); return }
  if (pc.length !== pr.length || pr.length !== rc.length || rc.length !== rr.length) { ElMessage.warning('四个数组长度必须一致'); return }
  try {
    const { data } = await computeLoss({ policy_chosen_logps: pc, policy_rejected_logps: pr, ref_chosen_logps: rc, ref_rejected_logps: rr, beta: beta.value })
    result.value = data
  } catch (e) { ElMessage.error('计算失败') }
}
</script>
