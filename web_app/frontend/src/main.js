// main.js
import { createApp } from 'vue'
import { createPinia } from 'pinia'
import { create, NConfigProvider, NMessageProvider, NUpload, NButton, NSpin } from 'naive-ui'

// self-hosted variable fonts (no CDN)
import '@fontsource-variable/bricolage-grotesque'
import '@fontsource-variable/hanken-grotesk'

// design tokens + base styles
import './styles/tokens.css'

import App from './App.vue'
import router from './router'

const naive = create({
  components: [NConfigProvider, NMessageProvider, NUpload, NButton, NSpin],
})

const app = createApp(App)
app.use(createPinia())
app.use(router)
app.use(naive)
app.mount('#app')
