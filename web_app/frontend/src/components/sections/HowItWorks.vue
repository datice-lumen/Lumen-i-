<template>
  <section id="how" class="how">
    <div class="shell">
      <header class="sec-head">
        <p class="eyebrow">How it works</p>
        <h2>Five steps, all in the open</h2>
        <p class="lead">
          Most tools hand you a single number. Datice shows every stage between a photo
          and its verdict — because a screening call you can inspect is one you can trust.
        </p>
      </header>

      <ol class="flow">
        <li v-for="(s, i) in steps" :key="i" class="row">
          <div class="spine">
            <span class="num">{{ String(i + 1).padStart(2, '0') }}</span>
          </div>
          <div class="card">
            <span class="icon" v-html="s.icon" aria-hidden="true"></span>
            <div>
              <h3>{{ s.title }}</h3>
              <p>{{ s.body }}</p>
            </div>
          </div>
        </li>
      </ol>
    </div>
  </section>
</template>

<script setup>
const steps = [
  {
    title: 'Square & focus',
    body: 'The photo is cropped to a centred square so the lesion fills the frame the model expects — phone shot or dermatoscope alike.',
    icon: `<svg viewBox="0 0 24 24" width="22" height="22"><rect x="4" y="4" width="16" height="16" rx="3" fill="none" stroke="currentColor" stroke-width="1.8"/><circle cx="12" cy="12" r="3.2" fill="none" stroke="currentColor" stroke-width="1.8"/></svg>`,
  },
  {
    title: 'Remove hair',
    body: 'Black-hat filtering detects hair strands, then inpainting paints them out — so the model reads skin, not artefacts.',
    icon: `<svg viewBox="0 0 24 24" width="22" height="22"><path d="M4 18c4-10 12-10 16 0M7 6c3 4 7 4 10 0" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>`,
  },
  {
    title: 'Measure skin tone',
    body: 'The Individual Typology Angle is computed from eight lesion-free patches around the edge, then mapped to a Fitzpatrick group.',
    icon: `<svg viewBox="0 0 24 24" width="22" height="22"><circle cx="12" cy="12" r="8.5" fill="none" stroke="currentColor" stroke-width="1.8"/><path d="M12 3.5v17" stroke="currentColor" stroke-width="1.8"/><path d="M12 12a8.5 8.5 0 0 1 8.5-8.5" fill="currentColor" opacity="0.18"/></svg>`,
  },
  {
    title: 'Classify',
    body: 'A custom 6.7M-parameter CNN — trained from scratch with a fairness-aware loss — returns the probability the lesion is malignant.',
    icon: `<svg viewBox="0 0 24 24" width="22" height="22"><circle cx="6" cy="7" r="2" fill="none" stroke="currentColor" stroke-width="1.8"/><circle cx="6" cy="17" r="2" fill="none" stroke="currentColor" stroke-width="1.8"/><circle cx="18" cy="12" r="2" fill="none" stroke="currentColor" stroke-width="1.8"/><path d="M8 7c5 0 5 5 8 5M8 17c5 0 5-5 8-5" fill="none" stroke="currentColor" stroke-width="1.8"/></svg>`,
  },
  {
    title: 'Explain',
    body: 'Grad-CAM turns the model’s attention into a heatmap, so you can see the exact region that drove the call — not just the score.',
    icon: `<svg viewBox="0 0 24 24" width="22" height="22"><path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12Z" fill="none" stroke="currentColor" stroke-width="1.8"/><circle cx="12" cy="12" r="3" fill="currentColor"/></svg>`,
  },
]
</script>

<style scoped>
.how {
  padding-block: var(--section-y);
  scroll-margin-top: 84px;
}
.sec-head {
  max-width: 60ch;
  margin-bottom: 2.6rem;
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
.flow {
  list-style: none;
  margin: 0;
  padding: 0;
  max-width: 820px;
}
.row {
  display: grid;
  grid-template-columns: 56px 1fr;
  gap: 1.2rem;
}
.spine {
  display: flex;
  flex-direction: column;
  align-items: center;
}
.num {
  display: grid;
  place-items: center;
  width: 46px;
  height: 46px;
  flex: none;
  border-radius: 50%;
  font-family: var(--font-display);
  font-weight: 800;
  font-size: 0.9rem;
  color: var(--coral-deep);
  background: var(--surface);
  border: 2px solid var(--coral-wash);
  box-shadow: var(--shadow-sm);
}
.row:not(:last-child) .spine::after {
  content: '';
  flex: 1;
  width: 2px;
  margin: 6px 0;
  background: linear-gradient(var(--coral-wash), var(--amber-wash));
  border-radius: 2px;
}
.card {
  display: flex;
  gap: 1rem;
  align-items: flex-start;
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: var(--r-lg);
  padding: 1.2rem 1.4rem;
  margin-bottom: 1.1rem;
  box-shadow: var(--shadow-sm);
  transition: transform 0.2s var(--ease), box-shadow 0.2s var(--ease);
}
.card:hover {
  transform: translateY(-2px);
  box-shadow: var(--shadow-md);
}
.icon {
  display: inline-grid;
  place-items: center;
  width: 44px;
  height: 44px;
  flex: none;
  border-radius: var(--r-md);
  color: var(--coral-deep);
  background: var(--glow-soft);
}
.card h3 {
  font-size: 1.16rem;
  margin-bottom: 0.35rem;
}
.card p {
  color: var(--ink-soft);
  font-size: 0.96rem;
  line-height: 1.55;
}

@media (max-width: 560px) {
  .row {
    grid-template-columns: 40px 1fr;
    gap: 0.8rem;
  }
  .num {
    width: 38px;
    height: 38px;
    font-size: 0.8rem;
  }
  .card {
    flex-direction: column;
    gap: 0.7rem;
    padding: 1.1rem;
  }
}
</style>
