<template>
  <figure class="gradcam">
    <div
      ref="track"
      class="stage"
      @pointerdown="startDrag"
    >
      <!-- base: the original photo -->
      <img class="layer" :src="original" alt="Original lesion photo" draggable="false" />

      <!-- revealed: where the model looked (clipped to the beam) -->
      <img
        class="layer heat"
        :src="heatmap"
        alt="Grad-CAM heatmap of the model's attention"
        draggable="false"
        :style="{ clipPath: `inset(0 ${100 - pos}% 0 0)` }"
      />

      <span class="tag left" :style="{ opacity: pos > 16 ? 1 : 0 }">Where it looked</span>
      <span class="tag right" :style="{ opacity: pos < 84 ? 1 : 0 }">Original</span>

      <!-- the beam of light -->
      <div
        class="beam"
        role="slider"
        tabindex="0"
        aria-label="Reveal the model's attention heatmap"
        aria-valuemin="0"
        aria-valuemax="100"
        :aria-valuenow="Math.round(pos)"
        :style="{ left: pos + '%' }"
        @keydown="onKey"
      >
        <span class="glow"></span>
        <span class="knob" aria-hidden="true">
          <svg viewBox="0 0 24 24" width="16" height="16">
            <path d="M9 6l-4 6 4 6M15 6l4 6-4 6" fill="none" stroke="currentColor"
              stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" />
          </svg>
        </span>
      </div>
    </div>
    <figcaption>
      Drag the beam to sweep light across the lesion — the warm zone is where the
      model concentrated to reach its call.
    </figcaption>
  </figure>
</template>

<script setup>
import { onBeforeUnmount, onMounted, ref } from 'vue'

defineProps({
  original: { type: String, required: true },
  heatmap: { type: String, required: true },
})

const track = ref(null)
const pos = ref(12)
let dragging = false
let raf = 0

function setFromClientX(clientX) {
  const el = track.value
  if (!el) return
  const rect = el.getBoundingClientRect()
  const p = ((clientX - rect.left) / rect.width) * 100
  pos.value = Math.min(100, Math.max(0, p))
}

function startDrag(e) {
  dragging = true
  cancelAnimationFrame(raf)
  setFromClientX(e.clientX)
  window.addEventListener('pointermove', onMove)
  window.addEventListener('pointerup', endDrag)
}
function onMove(e) {
  if (dragging) setFromClientX(e.clientX)
}
function endDrag() {
  dragging = false
  window.removeEventListener('pointermove', onMove)
  window.removeEventListener('pointerup', endDrag)
}
function onKey(e) {
  const step = e.shiftKey ? 12 : 4
  if (e.key === 'ArrowLeft') pos.value = Math.max(0, pos.value - step)
  else if (e.key === 'ArrowRight') pos.value = Math.min(100, pos.value + step)
  else if (e.key === 'Home') pos.value = 0
  else if (e.key === 'End') pos.value = 100
  else return
  e.preventDefault()
}

// one-time illumination sweep on reveal
onMounted(() => {
  const reduce = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
  if (reduce) {
    pos.value = 55
    return
  }
  const start = performance.now()
  const from = 8
  const to = 58
  const dur = 1150
  const tick = (now) => {
    const t = Math.min(1, (now - start) / dur)
    const eased = 1 - Math.pow(1 - t, 3)
    pos.value = from + (to - from) * eased
    if (t < 1) raf = requestAnimationFrame(tick)
  }
  raf = requestAnimationFrame(tick)
})

onBeforeUnmount(() => {
  cancelAnimationFrame(raf)
  endDrag()
})
</script>

<style scoped>
.gradcam {
  margin: 0;
}
.stage {
  position: relative;
  width: 100%;
  aspect-ratio: 1 / 1;
  max-height: 460px;
  border-radius: var(--r-lg);
  overflow: hidden;
  background: #101a3a;
  cursor: ew-resize;
  touch-action: none;
  box-shadow: var(--shadow-md);
  user-select: none;
}
.layer {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: cover;
}
.heat {
  will-change: clip-path;
}
.tag {
  position: absolute;
  top: 14px;
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #fff;
  padding: 5px 11px;
  border-radius: var(--r-pill);
  background: rgba(20, 14, 30, 0.55);
  backdrop-filter: blur(6px);
  transition: opacity 0.25s var(--ease);
  pointer-events: none;
}
.tag.left {
  left: 14px;
}
.tag.right {
  right: 14px;
}
.beam {
  position: absolute;
  top: 0;
  bottom: 0;
  width: 0;
  transform: translateX(-50%);
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: ew-resize;
}
.beam:focus-visible {
  outline: none;
}
.glow {
  position: absolute;
  top: 0;
  bottom: 0;
  width: 3px;
  background: #fff;
  box-shadow: 0 0 14px 3px rgba(255, 236, 190, 0.9),
    0 0 40px 10px rgba(245, 166, 35, 0.55);
}
.beam:focus-visible .glow {
  width: 4px;
  box-shadow: 0 0 18px 5px rgba(255, 236, 190, 1), 0 0 52px 14px rgba(244, 114, 107, 0.7);
}
.knob {
  position: relative;
  z-index: 1;
  display: grid;
  place-items: center;
  width: 40px;
  height: 40px;
  border-radius: 50%;
  color: var(--coral-deep);
  background: #fff;
  box-shadow: 0 4px 14px rgba(40, 20, 12, 0.35), 0 0 0 4px rgba(255, 255, 255, 0.35);
}
figcaption {
  margin-top: 0.9rem;
  color: var(--ink-soft);
  font-size: 0.92rem;
  line-height: 1.5;
}
</style>
