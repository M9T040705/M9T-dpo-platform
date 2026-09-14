import axios from 'axios'

const api = axios.create({ baseURL: '/api', timeout: 30000 })

// 训练任务
export const getTrainingJobs = (params) => api.get('/training/jobs', { params })
export const getTrainingJob = (id) => api.get(`/training/jobs/${id}`)
export const createTrainingJob = (data) => api.post('/training/jobs', data)
export const startTrainingJob = (id) => api.post(`/training/jobs/${id}/start`)
export const stopTrainingJob = (id) => api.post(`/training/jobs/${id}/stop`)
export const getJobMetrics = (id, params) => api.get(`/training/jobs/${id}/metrics`, { params })
export const deleteTrainingJob = (id) => api.delete(`/training/jobs/${id}`)

// 数据集
export const getDatasets = (params) => api.get('/dataset/datasets', { params })
export const getDataset = (id) => api.get(`/dataset/datasets/${id}`)
export const buildDataset = (data) => api.post('/dataset/datasets/build', data)
export const exportDataset = (id, fmt) => api.post(`/dataset/datasets/${id}/export/${fmt}`)
export const deleteDataset = (id) => api.delete(`/dataset/datasets/${id}`)

// 损失计算
export const computeLoss = (data) => api.post('/dataset/loss/compute', data)

export default api
