<template>
  <div class="pipeline">
    <div class="head">
      <p class="eyebrow">The pipeline, step by step</p>
      <div class="progress">
        <span class="count">{{ doneCount }} / {{ stages.length }}</span>
        <span class="bar"><span class="fill" :style="{ width: pct + '%' }"></span></span>
      </div>
    </div>

    <ol class="grid">
      <li
        v-for="(s, i) in stages"
        :key="s.key"
        class="step"
        :class="{ done: !!steps[s.key], current: i === activeIndex }"
      >
        <div class="frame">
          <img v-if="steps[s.key]" :src="steps[s.key]" :alt="s.label" />
          <div v-else class="ph" :class="{ live: streaming && i === nextIndex }">
            <span class="shimmer"></span>
          </div>
          <span class="num">{{ String(i + 1).padStart(2, '0') }}</span>
        </div>
        <div class="meta">
          <p class="label">
            {{ s.label }}
            <span v-if="s.key === 'processed' && skinGroup" class="chip">{{ skinGroup }}</span>
          </p>
          <p class="hint">{{ s.hint }}</p>
        </div>
      </li>
    </ol>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  stages: { type: Array, required: true },
  steps: { type: Object, required: true },
  skinGroup: { type: String, default: '' },
  activeIndex: { type: Number, default: -1 },
  streaming: { type: Boolean, default: false },
})

const doneCount = computed(() => props.stages.filter((s) => props.steps[s.key]).length)
const pct = computed(() => (doneCount.value / props.stages.length) * 100)
const nextIndex = computed(() => props.stages.findIndex((s) => !props.steps[s.key]))
</script>

<style scoped>
.pipeline {
  margin-top: 1.8rem;
}
.head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  flex-wrap: wrap;
  margin-bottom: 1.1rem;
}
.progress {
  display: flex;
  align-items: center;
  gap: 0.7rem;
}
.count {
  font-weight: 800;
  font-size: 0.85rem;
  color: var(--ink-soft);
  font-variant-numeric: tabular-nums;
}
.bar {
  width: 120px;
  height: 6px;
  border-radius: var(--r-pill);
  background: var(--sand-deep);
  overflow: hidden;
}
.fill {
  display: block;
  height: 100%;
  border-radius: var(--r-pill);
  background: var(--glow);
  transition: width 0.5s var(--ease);
}
.grid {
  list-style: none;
  margin: 0;
  padding: 0;
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 0.9rem;
}
.step {
  opacity: 0.5;
  transition: opacity 0.4s var(--ease), transform 0.4s var(--ease);
}
.step.done {
  opacity: 1;
  animation: pop 0.45s var(--ease);
}
@keyframes pop {
  from {
    transform: translateY(8px) scale(0.98);
    opacity: 0;
  }
  to {
    transform: translateY(0) scale(1);
    opacity: 1;
  }
}
.frame {
  position: relative;
  aspect-ratio: 1 / 1;
  border-radius: var(--r-md);
  overflow: hidden;
  background: var(--sand-deep);
  border: 1px solid var(--line);
}
.step.current .frame {
  border-color: var(--coral);
  box-shadow: 0 0 0 3px var(--coral-wash);
}
.frame img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}
.ph {
  position: absolute;
  inset: 0;
  overflow: hidden;
}
.ph.live .shimmer {
  position: absolute;
  inset: 0;
  background: linear-gradient(
    100deg,
    transparent 20%,
    rgba(244, 114, 107, 0.18) 50%,
    transparent 80%
  );
  animation: sweep 1.2s infinite;
}
@keyframes sweep {
  from {
    transform: translateX(-100%);
  }
  to {
    transform: translateX(100%);
  }
}
.num {
  position: absolute;
  top: 8px;
  left: 8px;
  font-family: var(--font-display);
  font-weight: 800;
  font-size: 0.72rem;
  color: #fff;
  background: rgba(20, 14, 10, 0.5);
  border-radius: var(--r-pill);
  padding: 2px 8px;
  letter-spacing: 0.05em;
}
.meta {
  padding-top: 0.65rem;
}
.label {
  font-weight: 700;
  font-size: 0.9rem;
  color: var(--ink);
  display: flex;
  flex-direction: column;
  gap: 0.3rem;
}
.chip {
  align-self: flex-start;
  font-size: 0.72rem;
  font-weight: 700;
  color: var(--coral-deep);
  background: var(--coral-wash);
  border-radius: var(--r-pill);
  padding: 2px 8px;
}
.hint {
  font-size: 0.8rem;
  color: var(--ink-faint);
  line-height: 1.45;
  margin-top: 0.3rem;
}

@media (max-width: 900px) {
  .grid {
    grid-template-columns: repeat(2, 1fr);
    gap: 1.1rem;
  }
}
@media (max-width: 460px) {
  .grid {
    grid-template-columns: 1fr;
  }
}
</style>
