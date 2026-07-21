import { computed, reactive } from 'vue'
import { buildDemoRun } from './demoData'

// Ordered pipeline stages. `key` matches the reactive step slots; the labels
// drive the on-screen stepper. Keeping this here (not in the view) means the
// composable owns the full contract with the backend SSE stream.
export const STAGES = [
  { key: 'original', label: 'Original photo', hint: 'Your uploaded image, squared for analysis.' },
  { key: 'hairMask', label: 'Hair mask', hint: 'Hairs are detected so they can be removed.' },
  { key: 'hairRemoved', label: 'Hair removed', hint: 'Inpainting clears artefacts that confuse the model.' },
  { key: 'processed', label: 'Skin-tone estimate', hint: 'Fitzpatrick group read from lesion-free skin.' },
  { key: 'gradcam', label: 'Where it looked', hint: 'Grad-CAM highlights the pixels that drove the call.' },
]

// Backend sends raw base64 PNG; the sample run sends ready-made data URIs.
const png = (v) => (typeof v === 'string' && v.startsWith('data:') ? v : `data:image/png;base64,${v}`)

function freshState() {
  return {
    phase: 'idle', // idle | streaming | done | error
    isDemo: false,
    // true when the backend's skin gate declined the photo (not a skin image).
    unclassified: false,
    errorMessage: '',
    meta: { width: 0, height: 0 },
    skinGroup: '',
    probability: null,
    predictedClass: null,
    // metadata the model actually reasoned with (echoed by the backend), or null
    metadataUsed: null,
    steps: {
      original: null,
      hairMask: null,
      hairRemoved: null,
      processed: null,
      gradcam: null,
    },
  }
}

export function useAnalyzer() {
  const state = reactive(freshState())

  function reset() {
    Object.assign(state, freshState())
  }

  const hasResult = computed(
    () => state.probability !== null && state.predictedClass !== null,
  )

  const activeStepIndex = computed(() => {
    // index of the last stage that has arrived
    let idx = -1
    STAGES.forEach((s, i) => {
      if (state.steps[s.key]) idx = i
    })
    return idx
  })

  // Apply one decoded SSE payload to the reactive state.
  function applyEvent(p) {
    switch (p.step) {
      case 'load_image':
        state.meta = { width: p.width, height: p.height }
        state.steps.original = png(p.image_base64)
        break
      case 'remove_hair':
        state.steps.hairMask = png(p.hair_mask)
        state.steps.hairRemoved = png(p.inpainted_image)
        break
      case 'preprocess':
        state.skinGroup = p.skin_group
        // backend also streams the processed image — surface it here
        if (p.processed_image) state.steps.processed = png(p.processed_image)
        break
      case 'model_prediction':
        state.probability = p.probability
        state.predictedClass = p.predicted_class
        if (p.metadata_used) state.metadataUsed = p.metadata_used
        break
      case 'gradcam':
        state.steps.gradcam = png(p.gradcam)
        break
      case 'unclassified':
        // The skin gate declined the photo: it isn't a close-up of skin, so the
        // melanoma model was NOT run. Surface the backend's guidance via the same
        // terminal display as errors; `unclassified` lets the view style it as a
        // gentle "please upload a skin photo" notice rather than a failure.
        state.unclassified = true
        state.errorMessage = p.message || 'This image was not recognised as skin.'
        state.phase = 'error'
        break
      case 'error':
        state.errorMessage = p.message || 'Something went wrong during analysis.'
        state.phase = 'error'
        break
      case 'done':
        if (state.phase !== 'error') state.phase = 'done'
        break
    }
  }

  // Stream a real analysis from the FastAPI backend.
  // `metadata` is optional: { age, sex, anatom_site }. Missing/unknown values
  // are left to the backend's unknown/missing encoding.
  // `mode` selects the model: 'phone' (mobile fine-tune) or 'derm' (dermatoscopic).
  async function analyze(file, metadata = {}, mode = 'phone') {
    reset()
    state.phase = 'streaming'

    try {
      const form = new FormData()
      form.append('file', file)
      if (metadata.age !== null && metadata.age !== undefined && metadata.age !== '') {
        form.append('age', String(metadata.age))
      }
      form.append('sex', metadata.sex || 'unknown')
      form.append('anatom_site', metadata.anatom_site || 'unknown')
      form.append('mode', mode)

      const res = await fetch('/image/process', { method: 'POST', body: form })
      if (!res.ok || !res.body) throw new Error(`Server responded ${res.status}`)

      const reader = res.body.getReader()
      const dec = new TextDecoder()
      let buf = ''

      // Parse the text/event-stream: events are separated by a blank line.
       
      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        buf += dec.decode(value, { stream: true })

        const chunks = buf.split('\n\n')
        buf = chunks.pop()

        for (const chunk of chunks) {
          const line = chunk.split('\n').find((l) => l.startsWith('data:'))
          if (!line) continue
          applyEvent(JSON.parse(line.replace(/^data:\s*/, '')))
        }
      }

      if (state.phase === 'streaming') state.phase = 'done'
    } catch (err) {
      state.errorMessage =
        err?.message === 'Failed to fetch'
          ? "Couldn't reach the analysis server. Is the backend running?"
          : err?.message || 'Analysis failed.'
      state.phase = 'error'
    }
  }

  // Play a labelled sample run so the pipeline can be demonstrated without a
  // live model. Uses clearly-synthetic placeholder imagery.
  let demoTimers = []
  function stopDemo() {
    demoTimers.forEach(clearTimeout)
    demoTimers = []
  }

  function playDemo(metadata = {}) {
    reset()
    stopDemo()
    state.phase = 'streaming'
    state.isDemo = true

    const events = buildDemoRun(metadata)
    let t = 260
    events.forEach((ev) => {
      demoTimers.push(
        setTimeout(() => {
          applyEvent(ev)
        }, t),
      )
      t += ev.step === 'gradcam' ? 900 : 620
    })
  }

  return {
    state,
    STAGES,
    hasResult,
    activeStepIndex,
    reset,
    analyze,
    playDemo,
    stopDemo,
  }
}
