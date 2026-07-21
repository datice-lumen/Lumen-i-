<template>
  <div class="trend">
    <div class="head">
      <span class="lbl">Malignancy % over time</span>
      <span v-if="last !== null" class="now" :class="rising ? 'up' : 'down'">
        {{ Math.round(last * 100) }}%
        <svg v-if="pts.length > 1" viewBox="0 0 24 24" width="14" height="14" aria-hidden="true">
          <path :d="rising ? 'M7 14l5-5 5 5' : 'M7 10l5 5 5-5'" fill="none"
            stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" />
        </svg>
      </span>
    </div>

    <svg v-if="pts.length > 1" class="spark" viewBox="0 0 100 44" preserveAspectRatio="none">
      <defs>
        <linearGradient id="trendFill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stop-color="rgba(244,114,107,0.28)" />
          <stop offset="1" stop-color="rgba(244,114,107,0)" />
        </linearGradient>
      </defs>
      <polygon :points="areaPoints" fill="url(#trendFill)" />
      <polyline
        :points="linePoints"
        fill="none"
        stroke="#f4726b"
        stroke-width="2"
        stroke-linecap="round"
        stroke-linejoin="round"
        vector-effect="non-scaling-stroke"
      />
      <circle :cx="coords[coords.length - 1].x" :cy="coords[coords.length - 1].y" r="2.6" fill="#d9544d" />
    </svg>

    <p v-else class="single">
      Add another check to see the trend.
    </p>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  // chronological array of { date, value } where value is 0..1 or null
  points: { type: Array, default: () => [] },
})

const pts = computed(() => props.points.filter((p) => p.value != null))
const last = computed(() => (pts.value.length ? pts.value[pts.value.length - 1].value : null))
const rising = computed(
  () => pts.value.length > 1 && last.value >= pts.value[pts.value.length - 2].value,
)

// map to a 0..100 x / 0..44 y viewbox (y inverted; higher % sits higher)
const coords = computed(() => {
  const n = pts.value.length
  return pts.value.map((p, i) => ({
    x: n === 1 ? 50 : (i / (n - 1)) * 100,
    y: 42 - p.value * 40,
  }))
})
const linePoints = computed(() => coords.value.map((c) => `${c.x},${c.y}`).join(' '))
const areaPoints = computed(() => {
  const c = coords.value
  if (!c.length) return ''
  return `${c[0].x},44 ` + c.map((p) => `${p.x},${p.y}`).join(' ') + ` ${c[c.length - 1].x},44`
})
</script>

<style scoped>
.trend {
  background: var(--surface-warm);
  border: 1px solid var(--line);
  border-radius: var(--r-md);
  padding: 1rem 1.1rem;
}
.head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 0.6rem;
}
.lbl {
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--ink-faint);
}
.now {
  display: inline-flex;
  align-items: center;
  gap: 0.25rem;
  font-family: var(--font-display);
  font-weight: 800;
  font-size: 1.15rem;
  font-variant-numeric: tabular-nums;
}
.now.up {
  color: var(--malignant-ink);
}
.now.down {
  color: var(--benign-ink);
}
.spark {
  display: block;
  width: 100%;
  height: 52px;
}
.single {
  color: var(--ink-faint);
  font-size: 0.86rem;
  margin: 0.2rem 0;
}
</style>
