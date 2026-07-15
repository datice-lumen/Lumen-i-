<template>
  <div class="verdict" :class="tone">
    <div class="head">
      <span class="pill">
        <svg v-if="malignant" viewBox="0 0 24 24" width="17" height="17" aria-hidden="true">
          <path d="M12 8v5m0 3.5h.01M10.3 3.9 2.4 18a2 2 0 0 0 1.7 3h15.8a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z"
            fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />
        </svg>
        <svg v-else viewBox="0 0 24 24" width="17" height="17" aria-hidden="true">
          <path d="m20 7-9 9-5-5" fill="none" stroke="currentColor" stroke-width="2.4"
            stroke-linecap="round" stroke-linejoin="round" />
        </svg>
        {{ malignant ? 'Worth checking' : 'Likely benign' }}
      </span>
      <span v-if="isDemo" class="demo">Sample data</span>
    </div>

    <h3 class="title">{{ malignant ? 'Signs worth a closer look' : 'This lesion looks benign' }}</h3>

    <div class="meter" role="img" :aria-label="`Malignancy likelihood ${percent} percent`">
      <div class="scale">
        <span class="marker" :style="{ left: clamped + '%' }">
          <span class="dot"></span>
          <span class="val">{{ percent }}%</span>
        </span>
      </div>
      <div class="ends">
        <span>Benign</span>
        <span>Malignant</span>
      </div>
    </div>

    <p class="read">
      The model puts the chance this is malignant at <strong>{{ percent }}%</strong>.
      {{ malignant
        ? 'That crosses its threshold — have a dermatologist take a look.'
        : 'That sits below its threshold, but keep an eye on any change.' }}
    </p>

    <p class="fineprint">
      Datice is a research tool, not a diagnosis. When in doubt, see a clinician.
    </p>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  probability: { type: Number, required: true },
  predictedClass: { type: Number, required: true },
  isDemo: { type: Boolean, default: false },
})

const malignant = computed(() => props.predictedClass === 1)
const tone = computed(() => (malignant.value ? 'is-malignant' : 'is-benign'))
const percent = computed(() => Math.round(props.probability * 100))
const clamped = computed(() => Math.min(97, Math.max(3, props.probability * 100)))
</script>

<style scoped>
.verdict {
  --accent: var(--benign);
  --accent-ink: var(--benign-ink);
  --accent-wash: var(--benign-wash);
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: var(--r-lg);
  padding: clamp(1.3rem, 1rem + 1.4vw, 2rem);
  box-shadow: var(--shadow-md);
  position: relative;
  overflow: hidden;
}
.verdict.is-malignant {
  --accent: var(--malignant);
  --accent-ink: var(--malignant-ink);
  --accent-wash: var(--malignant-wash);
}
.verdict::before {
  content: '';
  position: absolute;
  inset: 0 0 auto 0;
  height: 5px;
  background: var(--accent);
}
.head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.6rem;
  margin-bottom: 0.9rem;
}
.pill {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  font-weight: 700;
  font-size: 0.86rem;
  padding: 0.34rem 0.75rem;
  border-radius: var(--r-pill);
  color: var(--accent-ink);
  background: var(--accent-wash);
}
.demo {
  font-size: 0.7rem;
  font-weight: 800;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--amber);
  border: 1px dashed var(--amber);
  border-radius: var(--r-pill);
  padding: 0.28rem 0.6rem;
}
.title {
  font-size: var(--fs-h3);
  margin-bottom: 1.3rem;
}
.scale {
  position: relative;
  height: 12px;
  border-radius: var(--r-pill);
  background: linear-gradient(90deg, var(--benign) 0%, var(--honey) 50%, var(--malignant) 100%);
}
.marker {
  position: absolute;
  top: 50%;
  transform: translate(-50%, -50%);
  display: grid;
  justify-items: center;
}
.dot {
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background: #fff;
  border: 4px solid var(--accent);
  box-shadow: 0 2px 8px rgba(40, 20, 12, 0.25);
}
.val {
  position: absolute;
  top: -26px;
  font-weight: 800;
  font-size: 0.82rem;
  color: var(--accent-ink);
  white-space: nowrap;
}
.ends {
  display: flex;
  justify-content: space-between;
  margin-top: 0.55rem;
  font-size: 0.8rem;
  font-weight: 600;
  color: var(--ink-faint);
}
.read {
  margin-top: 1.4rem;
  color: var(--ink-2);
  line-height: 1.6;
}
.read strong {
  color: var(--accent-ink);
}
.fineprint {
  margin-top: 0.9rem;
  padding-top: 0.9rem;
  border-top: 1px solid var(--line);
  color: var(--ink-faint);
  font-size: 0.85rem;
}
</style>
