<template>
  <n-message-provider>
    <n-layout>
      <!-- Header -->
      <n-layout-header bordered class="header">
        <n-space justify="space-between" align="center" class="header-content">
          <!-- Clickable Title -->
          <router-link to="/" class="logo-link">
            <div class="logo">Melanoma Detector</div>
          </router-link>

          <n-space>
            <n-button ghost tag="a" href="/about">About</n-button>
            <n-button
              ghost
              tag="a"
              href="https://github.com/hlupek/Lumen-i-/"
              target="_blank"
            >
              GitHub
            </n-button>
          </n-space>
        </n-space>
      </n-layout-header>

      <!-- Main Page Content -->
      <n-layout-content content-style="padding: 24px;">
        <router-view />
      </n-layout-content>
    </n-layout>
  </n-message-provider>
</template>

<script setup>
import { RouterView } from 'vue-router'
</script>

<style scoped>
.header {
  padding: 12px 32px;
  background-color: #f9fafb;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}

.header-content {
  max-width: 1000px;
  margin: 0 auto;
  width: 100%;
  padding: 0 24px;
  box-sizing: border-box;
}

@media (max-width: 600px) {
  .header {
    padding: 12px 24px;
  }

  .header-content {
    padding: 0 16px;
  }
}


.logo {
  font-weight: bold;
  font-size: 20px;
  color: #333;
}

.logo-link {
  text-decoration: none;
  color: inherit;
}

body {
  font-family: Arial, sans-serif;
  margin: 0;
  padding: 0;
}
</style>
