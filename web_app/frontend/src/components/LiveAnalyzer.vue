<template>
  <section id="analyzer" class="analyzer">
    <div class="shell">
      <header class="sec-head">
        <p class="eyebrow">Live analyzer</p>
        <h2>See it work on a photo</h2>
        <p class="lead">{{ leadText }}</p>
      </header>

      <div class="stage-card">
        <!-- IDLE -->
        <transition name="fade" mode="out-in">
          <div v-if="state.phase === 'idle'" key="idle" class="idle">
            <div class="mode-toggle" role="radiogroup" aria-label="Image type">
              <button
                type="button"
                role="radio"
                :aria-checked="mode === 'phone'"
                :class="{ active: mode === 'phone' }"
                @click="mode = 'phone'"
              >
                <svg viewBox="0 0 24 24" width="17" height="17" aria-hidden="true">
                  <rect x="7" y="2.5" width="10" height="19" rx="2.2" fill="none"
                    stroke="currentColor" stroke-width="1.8" />
                  <path d="M10.5 5.2h3" fill="none" stroke="currentColor"
                    stroke-width="1.8" stroke-linecap="round" />
                </svg>
                Phone photo
              </button>
              <button
                type="button"
                role="radio"
                :aria-checked="mode === 'derm'"
                :class="{ active: mode === 'derm' }"
                @click="mode = 'derm'"
              >
                <svg viewBox="0 0 24 24" width="17" height="17" aria-hidden="true">
                  <circle cx="11" cy="11" r="6.2" fill="none" stroke="currentColor"
                    stroke-width="1.8" />
                  <circle cx="11" cy="11" r="2.4" fill="none" stroke="currentColor"
                    stroke-width="1.8" />
                  <path d="m16 16 4 4" fill="none" stroke="currentColor"
                    stroke-width="1.8" stroke-linecap="round" />
                </svg>
                Dermatoscope
              </button>
            </div>

            <div class="meta-card">
              <p class="meta-head">
                About this spot
                <span class="meta-opt">optional · sharpens the read</span>
              </p>
              <div class="meta-fields">
                <label class="mfield">
                  <span>Age</span>
                  <input
                    v-model="age"
                    type="number"
                    min="0"
                    max="120"
                    inputmode="numeric"
                    placeholder="—"
                  />
                </label>
                <label class="mfield">
                  <span>Sex</span>
                  <select v-model="sex">
                    <option v-for="o in SEX_OPTIONS" :key="o.value" :value="o.value">
                      {{ o.label }}
                    </option>
                  </select>
                </label>
                <label class="mfield">
                  <span>Body site</span>
                  <select v-model="site">
                    <option v-for="o in SITE_OPTIONS" :key="o.value" :value="o.value">
                      {{ o.label }}
                    </option>
                  </select>
                </label>
              </div>
            </div>

            <n-upload
              :custom-request="onUpload"
              :show-file-list="false"
              :multiple="false"
              accept="image/*"
            >
              <n-upload-dragger class="drop">
                <span class="drop-icon" aria-hidden="true">
                  <svg viewBox="0 0 24 24" width="30" height="30">
                    <path d="M12 16V4m0 0 4 4m-4-4L8 8M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2"
                      fill="none" stroke="currentColor" stroke-width="2"
                      stroke-linecap="round" stroke-linejoin="round" />
                  </svg>
                </span>
                <p class="drop-title">Drop a photo here, or <span>browse</span></p>
                <p class="drop-sub">{{ dropSub }}</p>
              </n-upload-dragger>
            </n-upload>

            <p v-if="mode === 'derm'" class="derm-note">
              <svg viewBox="0 0 24 24" width="17" height="17" aria-hidden="true">
                <circle cx="10" cy="10" r="5.6" fill="none" stroke="currentColor"
                  stroke-width="1.8" />
                <path d="m14.4 14.4 4.4 4.4" fill="none" stroke="currentColor"
                  stroke-width="1.8" stroke-linecap="round" />
              </svg>
              Captured with a dermatoscope. The model was trained on contact
              dermatoscopy images, so it reads these most confidently.
            </p>

            <details v-if="mode === 'phone'" class="shots">
              <summary>
                <svg viewBox="0 0 24 24" width="17" height="17" aria-hidden="true">
                  <path d="M4 8h3l2-2.5h6L17 8h3a1 1 0 0 1 1 1v9a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V9a1 1 0 0 1 1-1Z"
                    fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round" />
                  <circle cx="12" cy="13" r="3.2" fill="none" stroke="currentColor" stroke-width="1.8" />
                </svg>
                How to take a good photo
                <span class="sum-hint">examples inside</span>
                <svg class="chev" viewBox="0 0 24 24" width="16" height="16" aria-hidden="true">
                  <path d="m6 9 6 6 6-6" fill="none" stroke="currentColor"
                    stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />
                </svg>
              </summary>

              <ul class="shot-list">
                <li>Use bright, even light — no harsh flash glare.</li>
                <li>Fill the frame with the spot, kept in the centre.</li>
                <li>Hold steady, about 10&nbsp;cm away, until it's sharp.</li>
                <li>Clear hair off the spot and avoid casting shadows.</li>
              </ul>

              <div class="examples">
                <figure class="ex-good">
                  <img
                    src="/photo-guide/good.jpg"
                    width="220"
                    height="220"
                    loading="lazy"
                    alt="A good photo: the spot centred, sharp and evenly lit, filling the frame"
                  />
                  <span class="tag do">
                    <svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true">
                      <path d="m5 13 4 4L19 7" fill="none" stroke="currentColor"
                        stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" />
                    </svg>
                    Do this
                  </span>
                  <figcaption>Centred, sharp, fills the frame, even light</figcaption>
                </figure>

                <p class="ex-head">Common mistakes to avoid</p>
                <div class="dont-grid">
                  <figure v-for="d in dontShots" :key="d.src" class="ex-dont">
                    <img :src="d.src" width="220" height="220" loading="lazy" :alt="d.alt" />
                    <span class="tag x" aria-hidden="true">
                      <svg viewBox="0 0 24 24" width="12" height="12">
                        <path d="M6 6l12 12M18 6 6 18" fill="none" stroke="currentColor"
                          stroke-width="2.6" stroke-linecap="round" />
                      </svg>
                    </span>
                    <figcaption>{{ d.cap }}</figcaption>
                  </figure>
                </div>
              </div>
            </details>

            <div class="idle-actions">
              <n-upload
                v-if="mode === 'phone'"
                :custom-request="onUpload"
                :show-file-list="false"
                :multiple="false"
                accept="image/*"
                capture="environment"
              >
                <button type="button" class="btn ghost">
                  <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
                    <path d="M4 8h3l2-2.5h6L17 8h3a1 1 0 0 1 1 1v9a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V9a1 1 0 0 1 1-1Z"
                      fill="none" stroke="currentColor" stroke-width="1.8"
                      stroke-linejoin="round" />
                    <circle cx="12" cy="13" r="3.2" fill="none" stroke="currentColor" stroke-width="1.8" />
                  </svg>
                  Take a photo
                </button>
              </n-upload>

              <span v-if="mode === 'phone'" class="or">or</span>

              <button type="button" class="btn text" @click="playDemo(buildMeta())">
                Watch a sample run
                <svg viewBox="0 0 24 24" width="15" height="15" aria-hidden="true">
                  <path d="M8 5v14l11-7z" fill="currentColor" />
                </svg>
              </button>
            </div>
            <p class="tip">{{ tipText }}</p>
          </div>

          <!-- ERROR -->
          <div v-else-if="state.phase === 'error'" key="error" class="error">
            <span class="err-icon" aria-hidden="true">
              <svg viewBox="0 0 24 24" width="26" height="26">
                <path d="M12 8v5m0 3.5h.01M10.3 3.9 2.4 18a2 2 0 0 0 1.7 3h15.8a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z"
                  fill="none" stroke="currentColor" stroke-width="2"
                  stroke-linecap="round" stroke-linejoin="round" />
              </svg>
            </span>
            <h3>Analysis didn't finish</h3>
            <p>{{ state.errorMessage }}</p>
            <button type="button" class="btn primary" @click="reset">Try again</button>
          </div>

          <!-- STREAMING / DONE -->
          <div v-else key="result" class="result">
            <div v-if="state.isDemo" class="demo-banner">
              <strong>Sample walkthrough</strong> — synthetic placeholder data, not a real
              analysis. Upload a photo for a genuine read.
            </div>

            <div class="result-grid">
              <div class="col-main">
                <VerdictCard
                  v-if="hasResult"
                  :probability="state.probability"
                  :predicted-class="state.predictedClass"
                  :is-demo="state.isDemo"
                />
                <div v-else class="skel verdict-skel">
                  <n-spin size="small" />
                  <span>Running the model…</span>
                </div>
              </div>
              <div class="col-side">
                <SkinToneBadge v-if="state.skinGroup" :skin-group="state.skinGroup" />
                <div v-else class="skel side-skel"><span>Reading skin tone…</span></div>
              </div>
            </div>

            <p v-if="hasResult" class="reasoned">
              <svg viewBox="0 0 24 24" width="15" height="15" aria-hidden="true">
                <circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" stroke-width="1.7" />
                <path d="M12 11v5m0-8.5h.01" fill="none" stroke="currentColor"
                  stroke-width="1.7" stroke-linecap="round" />
              </svg>
              <span v-if="reasonedSummary">Reasoned with <strong>{{ reasonedSummary }}</strong></span>
              <span v-else>Based on the image alone — add age, sex, or body site above for a sharper read.</span>
            </p>

            <div v-if="camBase && state.steps.gradcam" class="cam-block">
              <h3 class="cam-title">Where the model looked</h3>
              <GradCamViewer :original="camBase" :heatmap="state.steps.gradcam" />
            </div>

            <PipelineSteps
              :stages="STAGES"
              :steps="state.steps"
              :skin-group="state.skinGroup"
              :active-index="activeStepIndex"
              :streaming="state.phase === 'streaming'"
            />

            <div class="result-foot">
              <button
                v-if="hasResult"
                type="button"
                class="btn primary"
                @click="showSave = true"
              >
                <svg viewBox="0 0 24 24" width="17" height="17" aria-hidden="true">
                  <path d="M5 4h11l3 3v13a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V5a1 1 0 0 1 1-1Z"
                    fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round" />
                  <path d="M8 4v5h7M8 21v-6h8v6" fill="none" stroke="currentColor"
                    stroke-width="1.8" stroke-linejoin="round" />
                </svg>
                Save to a mole
              </button>
              <button type="button" class="btn ghost" @click="reset">
                Analyze another photo
              </button>
            </div>
          </div>
        </transition>
      </div>
    </div>

    <SaveToMoleDialog
      :open="showSave"
      :image-src="state.steps.original || ''"
      :probability="state.probability"
      :predicted-class="state.predictedClass"
      :skin-group="state.skinGroup"
      :metadata="state.metadataUsed"
      :is-demo="state.isDemo"
      @close="showSave = false"
      @saved="onSaved"
    />
  </section>
</template>

<script setup>
import { computed, onMounted, ref, watch } from 'vue'
import { NUpload, NUploadDragger, NSpin, useMessage } from 'naive-ui'
import { useAnalyzer } from '../composables/useAnalyzer'
import { SEX_OPTIONS, SITE_OPTIONS, metadataSummary } from '../composables/metadata'
import VerdictCard from './VerdictCard.vue'
import SkinToneBadge from './SkinToneBadge.vue'
import GradCamViewer from './GradCamViewer.vue'
import PipelineSteps from './PipelineSteps.vue'
import SaveToMoleDialog from './SaveToMoleDialog.vue'

const message = useMessage()
const { state, STAGES, hasResult, activeStepIndex, reset, analyze, playDemo } = useAnalyzer()

// Which model to run: 'phone' (mobile fine-tune) or 'derm' (dermatoscopic model_10_6).
// Persists across "Analyze another photo" so the chosen mode sticks.
const mode = ref('phone')

// Copy that adapts to the selected mode.
const leadText = computed(() =>
  mode.value === 'derm'
    ? 'Upload a dermatoscopic image of a lesion. Datice runs the dermatoscope-trained model and streams every step of its reasoning — the cleanup, the skin-tone read, the call, and where it looked. Nothing is hidden behind a single score.'
    : 'Upload a clear phone photo of a skin spot. Datice streams every step of its reasoning as it runs — the cleanup, the skin-tone read, the call, and where it looked. Nothing is hidden behind a single score.',
)
const dropSub = computed(() =>
  mode.value === 'derm'
    ? 'JPG or PNG · a dermatoscope image of a single lesion'
    : 'JPG or PNG · a single skin lesion, well lit and in focus',
)
const tipText = computed(() =>
  mode.value === 'derm'
    ? 'Your image is analysed on the fly and never stored. Dermatoscope images give the most confident read.'
    : 'Your image is analysed on the fly and never stored. Clear, well-lit phone photos work best — switch to Dermatoscope above if you have a dermatoscope image.',
)

// Optional metadata the user can add before analyzing.
const age = ref('')
const sex = ref('unknown')
const site = ref('unknown')
function buildMeta() {
  return { age: age.value === '' ? null : Number(age.value), sex: sex.value, anatom_site: site.value }
}

// Illustrative (synthetic) example photos for the "how to take a good photo" guide.
const dontShots = [
  { src: '/photo-guide/dont-far.jpg', cap: 'Too far away', alt: 'Photo taken too far away, so the spot is tiny' },
  { src: '/photo-guide/dont-blurry.jpg', cap: 'Out of focus', alt: 'Blurry, out-of-focus photo' },
  { src: '/photo-guide/dont-glare.jpg', cap: 'Flash glare', alt: 'Flash glare washing out the detail' },
  { src: '/photo-guide/dont-dark.jpg', cap: 'Too dark', alt: 'Photo that is too dark or in shadow' },
  { src: '/photo-guide/dont-hair.jpg', cap: 'Hair on the spot', alt: 'Hair covering the spot' },
]

const reasonedSummary = computed(() => metadataSummary(state.metadataUsed))

const showSave = ref(false)

function onSaved({ label }) {
  showSave.value = false
  message.success(`Saved to “${label}” · see My moles`)
  const el = document.querySelector('#moles')
  if (el) el.scrollIntoView({ behavior: 'smooth' })
}

// compare the heatmap against the raw uploaded photo (before any preprocessing).
// The model saw a center square crop of this image, and the square viewer uses
// object-fit: cover (also a center crop), so the warm zone still lands on the lesion.
const camBase = computed(() => state.steps.original)

function onUpload({ file, onFinish, onError }) {
  const raw = file.file
  if (!raw) {
    onError()
    return
  }
  analyze(raw, buildMeta(), mode.value).finally(onFinish)
}

// Shareable "?demo" deep link auto-starts the sample run.
onMounted(() => {
  if (new URLSearchParams(window.location.search).has('demo')) playDemo(buildMeta())
})

watch(
  () => state.phase,
  (phase, prev) => {
    if (phase === 'error') message.error(state.errorMessage)
    else if (phase === 'done' && prev === 'streaming' && !state.isDemo) {
      message.success('Analysis complete')
    }
  },
)
</script>

<style scoped>
.analyzer {
  padding-block: var(--section-y);
  scroll-margin-top: 84px;
}
.sec-head {
  max-width: 62ch;
  margin-bottom: 2rem;
}
.sec-head h2 {
  font-size: var(--fs-h1);
  margin: 0.7rem 0 0.9rem;
}
.lead {
  font-size: var(--fs-lead);
  color: var(--ink-soft);
  line-height: 1.6;
}
.stage-card {
  background: var(--surface-warm);
  border: 1px solid var(--line);
  border-radius: var(--r-xl);
  padding: clamp(1.1rem, 0.6rem + 2vw, 2.4rem);
  box-shadow: var(--shadow-md);
}

/* --- mode toggle (phone / dermatoscope) --- */
.mode-toggle {
  display: flex;
  gap: 0.3rem;
  margin-bottom: 1.5rem;
  padding: 0.3rem;
  border: 1px solid var(--line);
  border-radius: var(--r-pill);
  background: var(--surface);
}
.mode-toggle button {
  flex: 1;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 0.45rem;
  font-family: var(--font-body);
  font-weight: 700;
  font-size: 0.92rem;
  color: var(--ink-soft);
  padding: 0.55rem 0.9rem;
  border: none;
  border-radius: var(--r-pill);
  background: transparent;
  cursor: pointer;
  transition: color 0.2s var(--ease), background 0.2s var(--ease),
    box-shadow 0.2s var(--ease);
}
.mode-toggle button svg {
  flex: none;
}
.mode-toggle button:hover:not(.active) {
  color: var(--ink);
  background: var(--coral-wash);
}
.mode-toggle button.active {
  color: #fff;
  background: var(--glow);
  box-shadow: var(--shadow-glow);
}
.mode-toggle button:focus-visible {
  outline: 2px solid var(--coral);
  outline-offset: 2px;
}

/* --- dermatoscope note (shown in derm mode, in place of the phone guide) --- */
.derm-note {
  display: flex;
  align-items: flex-start;
  gap: 0.55rem;
  margin-top: 1.2rem;
  padding: 0.8rem 1rem;
  border: 1px solid var(--line);
  border-radius: var(--r-lg);
  background: var(--surface);
  color: var(--ink-soft);
  font-size: 0.9rem;
  line-height: 1.5;
}
.derm-note svg {
  flex: none;
  margin-top: 0.1rem;
  color: var(--coral-deep);
}

/* --- idle --- */
.drop {
  border-radius: var(--r-lg) !important;
  padding: clamp(1.8rem, 1rem + 4vw, 3.4rem) 1.5rem !important;
  text-align: center;
  transition: border-color 0.2s var(--ease), background 0.2s var(--ease);
}
.drop-icon {
  display: inline-grid;
  place-items: center;
  width: 62px;
  height: 62px;
  border-radius: 50%;
  color: #fff;
  background: var(--glow);
  box-shadow: var(--shadow-glow);
  margin-bottom: 1rem;
}
.drop-title {
  font-family: var(--font-display);
  font-weight: 700;
  font-size: 1.2rem;
  color: var(--ink);
}
.drop-title span {
  color: var(--coral-deep);
  text-decoration: underline;
  text-underline-offset: 3px;
}
.drop-sub {
  margin-top: 0.4rem;
  color: var(--ink-faint);
  font-size: 0.9rem;
}
.idle-actions {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 1rem;
  flex-wrap: wrap;
  margin-top: 1.4rem;
}
.or {
  color: var(--ink-faint);
  font-size: 0.9rem;
}
.tip {
  margin-top: 1.2rem;
  text-align: center;
  color: var(--ink-faint);
  font-size: 0.86rem;
}

/* --- how to take a good photo --- */
.shots {
  margin-top: 1.2rem;
  border: 1px solid var(--line);
  border-radius: var(--r-lg);
  background: var(--surface);
  overflow: hidden;
}
.shots > summary {
  display: flex;
  align-items: center;
  gap: 0.55rem;
  padding: 0.85rem 1.1rem;
  cursor: pointer;
  list-style: none;
  font-family: var(--font-display);
  font-weight: 700;
  color: var(--ink);
  user-select: none;
}
.shots > summary::-webkit-details-marker {
  display: none;
}
.shots > summary svg:first-of-type {
  flex: none;
  color: var(--coral-deep);
}
.shots > summary .chev {
  margin-left: auto;
  color: var(--ink-faint);
  transition: transform 0.2s var(--ease);
}
.shots[open] > summary .chev {
  transform: rotate(180deg);
}
.shots > summary:focus-visible {
  outline: 2px solid var(--coral);
  outline-offset: -2px;
}
.shot-list {
  margin: 0;
  padding: 0 1.1rem 1rem 1.1rem;
  list-style: none;
  display: grid;
  gap: 0.5rem;
}
.shot-list li {
  position: relative;
  padding-left: 1.4rem;
  color: var(--ink-soft);
  font-size: 0.9rem;
  line-height: 1.45;
}
.shot-list li::before {
  content: '';
  position: absolute;
  left: 0.2rem;
  top: 0.5rem;
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--coral);
}
.sum-hint {
  font-family: var(--font-body);
  font-weight: 700;
  font-size: 0.66rem;
  letter-spacing: 0.03em;
  text-transform: uppercase;
  color: var(--coral-deep);
  background: var(--coral-wash);
  padding: 0.14rem 0.45rem;
  border-radius: var(--r-pill);
}

/* --- do / don't example gallery --- */
.examples {
  padding: 0 1.1rem 1.1rem;
}
.tag {
  display: inline-flex;
  align-items: center;
  gap: 0.28rem;
  font-family: var(--font-body);
  font-weight: 700;
  font-size: 0.72rem;
  line-height: 1;
  color: #fff;
}
.ex-good {
  position: relative;
  display: flex;
  align-items: center;
  gap: 0.9rem;
  margin: 0.1rem 0 1.1rem;
  padding: 0.7rem;
  border: 1px solid var(--benign);
  border-radius: var(--r-lg);
  background: var(--benign-wash);
}
.ex-good img {
  flex: none;
  width: 116px;
  height: 116px;
  object-fit: cover;
  border-radius: var(--r-md);
}
.ex-good figcaption {
  font-size: 0.92rem;
  font-weight: 600;
  color: var(--benign-ink);
  line-height: 1.4;
}
.tag.do {
  position: absolute;
  top: 1.2rem;
  left: 1.2rem;
  padding: 0.22rem 0.5rem;
  border-radius: var(--r-pill);
  background: var(--benign);
  box-shadow: 0 1px 5px rgba(0, 0, 0, 0.2);
}
.ex-head {
  font-weight: 700;
  font-size: 0.82rem;
  color: var(--ink-2);
  margin: 0 0 0.6rem;
}
.dont-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
  gap: 0.7rem;
}
.ex-dont {
  position: relative;
  margin: 0;
}
.ex-dont img {
  width: 100%;
  height: auto; /* override the width/height attrs so aspect-ratio governs */
  aspect-ratio: 1;
  object-fit: cover;
  border-radius: var(--r-md);
  border: 1px solid var(--malignant);
}
.ex-dont figcaption {
  margin-top: 0.35rem;
  font-size: 0.76rem;
  color: var(--ink-faint);
  text-align: center;
}
.tag.x {
  position: absolute;
  top: 5px;
  right: 5px;
  width: 20px;
  height: 20px;
  padding: 0;
  justify-content: center;
  border-radius: 50%;
  background: var(--malignant);
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.25);
}

/* --- optional metadata card --- */
.meta-card {
  margin-bottom: 1.5rem;
  padding: 1rem 1.1rem 1.1rem;
  border: 1px solid var(--line);
  border-radius: var(--r-lg);
  background: var(--surface);
}
.meta-head {
  display: flex;
  align-items: baseline;
  gap: 0.6rem;
  flex-wrap: wrap;
  font-family: var(--font-display);
  font-weight: 700;
  color: var(--ink);
  margin-bottom: 0.85rem;
}
.meta-opt {
  font-family: var(--font-body);
  font-weight: 600;
  font-size: 0.78rem;
  color: var(--ink-faint);
}
.meta-fields {
  display: grid;
  grid-template-columns: 0.7fr 1fr 1.3fr;
  gap: 0.8rem;
}
.mfield {
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
}
.mfield > span {
  font-weight: 700;
  font-size: 0.78rem;
  color: var(--ink-2);
}
.mfield input,
.mfield select {
  width: 100%;
  font-family: var(--font-body);
  font-size: 0.95rem;
  color: var(--ink);
  padding: 0.55rem 0.7rem;
  border: 1px solid var(--line-strong);
  border-radius: var(--r-md);
  background: var(--surface);
}
.mfield input:focus,
.mfield select:focus {
  outline: none;
  border-color: var(--coral);
  box-shadow: 0 0 0 3px var(--coral-wash);
}
@media (max-width: 560px) {
  .meta-fields {
    grid-template-columns: 1fr 1fr;
  }
  .mfield:last-child {
    grid-column: 1 / -1;
  }
}

/* --- reasoned-with line --- */
.reasoned {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-top: 1rem;
  color: var(--ink-soft);
  font-size: 0.9rem;
}
.reasoned svg {
  flex: none;
  color: var(--coral-deep);
}
.reasoned strong {
  color: var(--ink);
  font-weight: 700;
}

/* --- buttons --- */
.btn {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  font-family: var(--font-body);
  font-weight: 700;
  font-size: 0.95rem;
  border-radius: var(--r-pill);
  padding: 0.7rem 1.3rem;
  cursor: pointer;
  border: 1px solid transparent;
  transition: transform 0.2s var(--ease), box-shadow 0.2s var(--ease),
    background 0.2s var(--ease);
}
.btn.primary {
  color: #fff;
  background: var(--glow);
  box-shadow: var(--shadow-glow);
}
.btn.primary:hover {
  transform: translateY(-1px);
}
.btn.ghost {
  color: var(--ink);
  background: var(--surface);
  border-color: var(--line-strong);
}
.btn.ghost:hover {
  border-color: var(--coral);
  color: var(--coral-deep);
}
.btn.text {
  color: var(--coral-deep);
  background: transparent;
  padding: 0.7rem 0.6rem;
}
.btn.text:hover {
  background: var(--coral-wash);
}

/* --- error --- */
.error {
  text-align: center;
  padding: clamp(1.5rem, 1rem + 3vw, 3rem) 1rem;
}
.err-icon {
  display: inline-grid;
  place-items: center;
  width: 56px;
  height: 56px;
  border-radius: 50%;
  color: var(--malignant);
  background: var(--malignant-wash);
  margin-bottom: 1rem;
}
.error h3 {
  font-size: var(--fs-h3);
  margin-bottom: 0.5rem;
}
.error p {
  color: var(--ink-soft);
  margin-bottom: 1.4rem;
  max-width: 46ch;
  margin-inline: auto;
}

/* --- result --- */
.demo-banner {
  background: var(--amber-wash);
  color: #7a4d09;
  border: 1px solid rgba(245, 166, 35, 0.4);
  border-radius: var(--r-md);
  padding: 0.75rem 1rem;
  font-size: 0.9rem;
  margin-bottom: 1.4rem;
}
.result-grid {
  display: grid;
  grid-template-columns: 1.5fr 1fr;
  gap: 1.2rem;
  align-items: stretch;
}
.cam-block {
  margin-top: 1.6rem;
}
.cam-title {
  font-size: var(--fs-h3);
  margin-bottom: 1rem;
}
.skel {
  display: flex;
  align-items: center;
  gap: 0.7rem;
  justify-content: center;
  min-height: 120px;
  border-radius: var(--r-lg);
  border: 1px dashed var(--line-strong);
  background: var(--surface);
  color: var(--ink-faint);
  font-size: 0.92rem;
}
.result-foot {
  margin-top: 1.8rem;
  display: flex;
  justify-content: center;
  gap: 1rem;
  flex-wrap: wrap;
}

.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.25s var(--ease);
}
.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}

@media (max-width: 760px) {
  .result-grid {
    grid-template-columns: 1fr;
  }
}
</style>
