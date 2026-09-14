import { createRouter, createWebHashHistory } from 'vue-router'
import Dashboard from './views/Dashboard.vue'
import TrainingJobs from './views/TrainingJobs.vue'
import DatasetManage from './views/DatasetManage.vue'
import LossCalculator from './views/LossCalculator.vue'

const routes = [
  { path: '/', redirect: '/dashboard' },
  { path: '/dashboard', name: 'Dashboard', component: Dashboard, meta: { title: '仪表盘' } },
  { path: '/training', name: 'TrainingJobs', component: TrainingJobs, meta: { title: '训练任务' } },
  { path: '/dataset', name: 'DatasetManage', component: DatasetManage, meta: { title: '数据集管理' } },
  { path: '/loss', name: 'LossCalculator', component: LossCalculator, meta: { title: '损失计算器' } },
]

const router = createRouter({
  history: createWebHashHistory(),
  routes
})

export default router
