<template>
  <n-config-provider :theme-overrides="themeOverrides">
    <n-message-provider :max="2" placement="bottom">
      <AppHeader />
      <main>
        <RouterView />
      </main>
      <AppFooter />
    </n-message-provider>
  </n-config-provider>
</template>

<script setup>
import { RouterView } from 'vue-router'
import { NConfigProvider, NMessageProvider } from 'naive-ui'
import { themeOverrides } from './theme'
import AppHeader from './components/AppHeader.vue'
import AppFooter from './components/AppFooter.vue'
</script>
