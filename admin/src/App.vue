<template>
  <el-container class="app-container">
    <el-aside width="220px" class="sidebar">
      <div class="logo">
        <el-icon :size="24"><Cpu /></el-icon>
        <span>DPO 微调平台</span>
      </div>
      <el-menu :default-active="activeMenu" router background-color="#1f2d3d" text-color="#bfcbd9" active-text-color="#409eff">
        <el-menu-item index="/dashboard"><el-icon><Odometer /></el-icon><span>仪表盘</span></el-menu-item>
        <el-menu-item index="/training"><el-icon><VideoPlay /></el-icon><span>训练任务</span></el-menu-item>
        <el-menu-item index="/dataset"><el-icon><Folder /></el-icon><span>数据集管理</span></el-menu-item>
        <el-menu-item index="/loss"><el-icon><Calculator /></el-icon><span>损失计算器</span></el-menu-item>
      </el-menu>
    </el-aside>
    <el-container>
      <el-header class="header">
        <span class="page-title">{{ pageTitle }}</span>
        <el-tag type="success" size="small">DPO + FastAPI + Vue 3</el-tag>
      </el-header>
      <el-main class="main-content"><router-view /></el-main>
    </el-container>
  </el-container>
</template>

<script setup>
import { computed } from 'vue'
import { useRoute } from 'vue-router'
const route = useRoute()
const activeMenu = computed(() => route.path)
const pageTitle = computed(() => route.meta.title || 'DPO 微调平台')
</script>

<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
html, body, #app { height: 100%; }
.app-container { height: 100vh; }
.sidebar { background-color: #1f2d3d; color: #fff; }
.logo { height: 60px; display: flex; align-items: center; justify-content: center; gap: 8px; font-size: 16px; font-weight: 600; color: #fff; border-bottom: 1px solid #2d3e50; }
.el-menu { border-right: none; }
.header { background: #fff; border-bottom: 1px solid #e4e7ed; display: flex; align-items: center; justify-content: space-between; }
.page-title { font-size: 18px; font-weight: 600; color: #303133; }
.main-content { background: #f5f7fa; padding: 20px; }
</style>
