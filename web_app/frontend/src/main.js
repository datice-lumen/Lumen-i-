// main.js
import { createApp } from 'vue'
import App from './App.vue'
import { createPinia } from 'pinia'
import router from './router'
import { create } from 'naive-ui'
import {
  NMessageProvider,
  NCard,
  NSpace,
  NUpload,
  NButton,
  NImage
} from 'naive-ui'

const naive = create({
  components: [
    NMessageProvider,
    NCard,
    NSpace,
    NUpload,
    NButton,
    NImage
  ]
})

const app = createApp(App)
app.use(createPinia())
app.use(router)
app.use(naive)
app.mount('#app')
