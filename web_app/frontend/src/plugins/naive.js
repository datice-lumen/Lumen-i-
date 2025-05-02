// src/plugins/naive.js
import {
  create,
  NButton,
  NUpload,
  NImage,
  NMessageProvider,
  NCard,
  NSpace,
  NSpin,
  NProgress,
} from 'naive-ui'

export function createNaiveUi() {
  return create({
    components: [
      NButton,
      NUpload,
      NImage,
      NMessageProvider,
      NCard,
      NSpace,
      NSpin,
      NProgress,
    ],
  })
}
