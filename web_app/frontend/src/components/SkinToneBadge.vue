<template>
  <div class="tone-card">
    <div class="top">
      <p class="eyebrow">Estimated skin tone</p>
      <h4 class="label">
        <span class="roman">{{ roman }}</span>
        <span class="desc">{{ desc }}</span>
      </h4>
    </div>

    <div class="scale" role="img" :aria-label="`Fitzpatrick scale, estimated group ${roman}`">
      <span
        v-for="(c, i) in swatches"
        :key="i"
        class="sw"
        :class="{ on: i + 1 === index }"
        :style="{ background: c }"
      >
        <span v-if="i + 1 === index" class="tick" aria-hidden="true"></span>
      </span>
    </div>

    <p class="note">
      Skin tone is read from lesion-free skin, not the mole itself. Datice was trained
      with an equalized-odds objective so its sensitivity stays even across tones.
      <a href="#fairness">How fairness is measured →</a>
    </p>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  skinGroup: { type: String, default: '' },
})

const swatches = [
  'var(--skin-1)',
  'var(--skin-2)',
  'var(--skin-3)',
  'var(--skin-4)',
  'var(--skin-5)',
  'var(--skin-6)',
]

const ROMAN = { I: 1, II: 2, III: 3, IV: 4, V: 5, VI: 6 }

// backend sends e.g. "III (Intermediate)"
const roman = computed(() => (props.skinGroup.split('(')[0] || '').trim() || '—')
const desc = computed(() => {
  const m = props.skinGroup.match(/\(([^)]+)\)/)
  return m ? m[1] : ''
})
const index = computed(() => ROMAN[roman.value] || 0)
</script>

<style scoped>
.tone-card {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: var(--r-lg);
  padding: clamp(1.2rem, 1rem + 1vw, 1.6rem);
  box-shadow: var(--shadow-sm);
  height: 100%;
}
.label {
  display: flex;
  align-items: baseline;
  gap: 0.55rem;
  margin-top: 0.5rem;
}
.roman {
  font-family: var(--font-display);
  font-size: 1.8rem;
  font-weight: 800;
  color: var(--ink);
}
.desc {
  font-size: 0.98rem;
  color: var(--ink-soft);
}
.scale {
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  gap: 6px;
  margin: 1.1rem 0;
}
.sw {
  position: relative;
  height: 30px;
  border-radius: 7px;
  transition: transform 0.2s var(--ease);
}
.sw.on {
  transform: scaleY(1.32);
  box-shadow: 0 0 0 3px var(--surface), 0 0 0 5px var(--ink);
  z-index: 1;
}
.tick {
  position: absolute;
  left: 50%;
  bottom: -12px;
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--ink);
  transform: translateX(-50%);
}
.note {
  font-size: 0.88rem;
  color: var(--ink-soft);
  line-height: 1.55;
}
.note a {
  color: var(--coral-deep);
  font-weight: 700;
  text-decoration: none;
  white-space: nowrap;
}
.note a:hover {
  text-decoration: underline;
}
</style>
