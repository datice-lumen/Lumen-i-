<template>
  <div class="page-container">
    <n-card>
      <n-spin :spinning="processing" block size="large" show-text>
        <n-space vertical size="large">
          <n-space>
            <n-upload
              accept="image/*"
              capture="environment"
              :custom-request="handleUpload"
              :show-file-list="false"
              :multiple="false"
            >
              <n-button type="primary">Upload from device</n-button>
            </n-upload>
            <n-upload
              accept="image/*"
              capture="environment"
              :custom-request="handleUpload"
              :show-file-list="false"
              :multiple="false"
            >
              <n-button>Take Photo</n-button>
            </n-upload>
          </n-space>

          <!-- Prediction (hidden while processing) -->
          <n-card
            v-if="!processing && prediction !== null"
            title="Prediction Result"
            size="small"
          >
            <p>
              Melanoma Detected:
              <strong>{{ predictionClass === 1 ? 'Yes' : 'No' }}</strong>
            </p>
            <p>
              Probability:
              <strong>{{ (prediction * 100).toFixed(2) }}%</strong>
            </p>
          </n-card>

          <!-- Timeline -->
          <n-card title="Processing Steps" size="small">
            <n-timeline size="medium">
              <n-timeline-item
                v-for="step in stepOrder"
                :key="step.key"
                :title="step.label"
                type="success"
              >
                <div class="step-content">
                  <n-image
                    v-if="outputs[step.key]?.image"
                    :src="outputs[step.key].image"
                    width="180"
                  />
                  <p v-else-if="outputs[step.key]?.text">
                    {{ outputs[step.key].text }}
                  </p>
                </div>
              </n-timeline-item>
            </n-timeline>
          </n-card>

        </n-space>
      </n-spin>
    </n-card>
  </div>
</template>

<script setup>
import {ref, reactive} from 'vue'
import {
  useMessage,
  NCard,
  NSpace,
  NUpload,
  NButton,
  NSpin,
  NTimeline,
  NTimelineItem,
  NSkeleton,
  NImage,
  NIcon
} from 'naive-ui'

const message = useMessage()
const processing = ref(false)
const prediction = ref(null)
const predictionClass = ref(null)

const outputs = reactive({})

const stepOrder = [
  {key: 'load_image', label: 'Original Image',},
  {key: 'hair_mask', label: 'Hair Mask',},
  {key: 'remove_hair', label: 'Hair Removal',},
  {key: 'preprocess', label: 'Approximated Skin Group',},
  {key: 'gradcam', label: 'Grad-CAM Visualization',}
]


async function handleUpload({file, onFinish, onError}) {
  // RESET state
  processing.value = true
  prediction.value = null
  predictionClass.value = null
  Object.keys(outputs).forEach(k => delete outputs[k])

  try {
    const form = new FormData()
    form.append('file', file.file)

    const res = await fetch('/image/process', {
      method: 'POST',
      body: form
    })
    if (!res.ok || !res.body) {
      throw new Error('Upload failed')
    }

    const reader = res.body.getReader()
    const dec = new TextDecoder()
    let buf = ''

    while (true) {
      const {value, done} = await reader.read()
      if (done) break
      buf += dec.decode(value, {stream: true})

      const chunks = buf.split('\n\n')
      buf = chunks.pop()

      for (const c of chunks) {
        const line = c.split('\n').find(l => l.startsWith('data:'))
        if (!line) continue
        const p = JSON.parse(line.replace(/^data:\s*/, ''))
        // populate outputs & prediction
        switch (p.step) {
          case 'load_image':
            outputs.load_image = {
              image: `data:image/png;base64,${p.image_base64}`
            }
            break
          case 'remove_hair':
            outputs.remove_hair = {
              image: `data:image/png;base64,${p.inpainted_image}`
            }
            outputs.hair_mask = {
              image: `data:image/png;base64,${p.hair_mask}`
            }
            break
          case 'preprocess':
            outputs.preprocess = {text: `${p.skin_group}`}
            break
          case 'model_prediction':
            prediction.value = p.probability
            predictionClass.value = p.predicted_class
            outputs.model_prediction = {}
            break
          case 'gradcam':
            outputs.gradcam = {
              image: `data:image/png;base64,${p.gradcam}`
            }
            break
          case 'error':
            message.error(p.message || 'Unknown error')
            processing.value = false
            break
          case 'done':
            message.success('Processing completed!')
            processing.value = false
            break
        }
      }
    }

    onFinish()  // clear the upload so old files don’t stick around
  } catch (err) {
    message.error(err.message || 'Processing error')
    processing.value = false
    onError(err)
  }
}
</script>

<style scoped>
.page-container {
  max-width: 1005px;
  margin: 50px auto;
  padding: 20px;
}

.step-content {
  margin-left: 8px;
}
</style>
