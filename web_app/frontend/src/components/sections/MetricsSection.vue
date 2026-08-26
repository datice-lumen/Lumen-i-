<template>
  <section id="results" ref="root" class="metrics">
    <div class="shell">
      <header class="sec-head">
        <p class="eyebrow">Results</p>
        <h2>Measured on held-out test data</h2>
        <p class="lead">
          A frozen DINOv2 backbone paired with a small trainable vision and metadata
          head — 722k trained parameters. Here's how it scores on the 10,326 held-out
          lesions it never saw during training.
        </p>
      </header>

      <div class="grid">
        <article v-for="(m, i) in metrics" :key="m.key" class="tile" :class="{ hero: i < 2 }">
          <div class="val">{{ display[i] }}</div>
          <div class="name">{{ m.label }}</div>
          <div class="track">
            <span
              class="fill"
              :class="m.higherBetter ? 'good' : 'watch'"
              :style="{ width: shown ? m.value * 100 + '%' : '0%' }"
            ></span>
          </div>
          <p class="note">
            {{ m.note }}
            <span class="dir">{{ m.higherBetter ? 'higher is better' : 'lower is better' }}</span>
          </p>
        </article>
      </div>
    </div>
  </section>
</template>

<script setup>
import { onBeforeUnmount, onMounted, reactive, ref } from 'vue'

// Test-set figures for the served dermatoscopic checkpoint (model_10_6, n=10,326).
// Source of truth: docs/training/model_10_6_logs/results_20260610_230527.txt.
// Keep these in sync with that file — they are quoted in the paper as well.
const metrics = [
  { key: 'tpr', value: 0.912, label: 'Sensitivity', note: 'Share of true malignancies it catches.', higherBetter: true },
  { key: 'acc', value: 0.876, label: 'Accuracy', note: 'Overall correct calls on the test set.', higherBetter: true },
  { key: 'prec', value: 0.687, label: 'Precision', note: 'Share of its malignant calls that are right.', higherBetter: true },
  { key: 'fpr', value: 0.136, label: 'False-positive rate', note: 'Benign lesions it flags by mistake.', higherBetter: false },
]

const root = ref(null)
const shown = ref(false)
const display = reactive(metrics.map(() => '0.00'))
let observer
let safety

const setFinal = () => metrics.forEach((m, i) => (display[i] = m.value.toFixed(2)))

function animate() {
  shown.value = true
  const reduce = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
  if (reduce) {
    setFinal()
    return
  }
  const dur = 1100
  const start = performance.now()
  const tick = (now) => {
    const t = Math.min(1, (now - start) / dur)
    const eased = 1 - Math.pow(1 - t, 3)
    metrics.forEach((m, i) => {
      display[i] = (m.value * eased).toFixed(2)
    })
    if (t < 1) requestAnimationFrame(tick)
  }
  requestAnimationFrame(tick)
  // guarantee final values even if rAF is throttled or cancelled
  safety = setTimeout(setFinal, dur + 350)
}

onMounted(() => {
  observer = new IntersectionObserver(
    (entries) => {
      if (entries[0].isIntersecting) {
        animate()
        observer.disconnect()
      }
    },
    { threshold: 0.35 },
  )
  if (root.value) observer.observe(root.value)
})
onBeforeUnmount(() => {
  observer?.disconnect()
  clearTimeout(safety)
})
</script>

<style scoped>
.metrics {
  padding-block: var(--section-y);
  scroll-margin-top: 84px;
}
.sec-head {
  max-width: 60ch;
  margin-bottom: 2.4rem;
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
.grid {
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  gap: 1.1rem;
}
.tile {
  grid-column: span 3;
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: var(--r-lg);
  padding: 1.5rem;
  box-shadow: var(--shadow-sm);
}
.tile.hero {
  grid-column: span 3;
  background: linear-gradient(180deg, #fffdf9, #fff8f0);
}
.val {
  font-family: var(--font-display);
  font-weight: 800;
  font-size: clamp(2.2rem, 1.6rem + 2vw, 3.2rem);
  line-height: 1;
  letter-spacing: -0.03em;
  color: var(--ink);
  font-variant-numeric: tabular-nums;
}
.tile.hero .val {
  background: var(--glow);
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
}
.name {
  margin-top: 0.5rem;
  font-weight: 700;
  color: var(--ink-2);
}
.track {
  margin: 0.9rem 0 0.7rem;
  height: 7px;
  border-radius: var(--r-pill);
  background: var(--sand-deep);
  overflow: hidden;
}
.fill {
  display: block;
  height: 100%;
  border-radius: var(--r-pill);
  transition: width 1s var(--ease);
}
.fill.good {
  background: linear-gradient(90deg, var(--benign), #5bbf98);
}
.fill.watch {
  background: linear-gradient(90deg, var(--amber), var(--coral));
}
.note {
  font-size: 0.86rem;
  color: var(--ink-soft);
  line-height: 1.45;
}
.dir {
  display: block;
  margin-top: 0.35rem;
  font-size: 0.74rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--ink-faint);
}

@media (max-width: 900px) {
  .grid {
    grid-template-columns: repeat(2, 1fr);
  }
  .tile,
  .tile.hero {
    grid-column: span 1;
  }
}
@media (max-width: 480px) {
  .grid {
    grid-template-columns: 1fr;
  }
}
</style>
