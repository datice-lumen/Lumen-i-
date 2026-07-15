<template>
  <section id="fairness" class="fairness">
    <div class="shell">
      <div class="split">
        <div class="intro">
          <p class="eyebrow">Fairness</p>
          <h2>Built to work across every skin tone</h2>
          <p class="lead">
            Skin-cancer AI has a well-documented blind spot: trained mostly on light skin,
            it tends to miss more on darker skin. Datice treats that gap as a first-class
            problem — measured during training, penalised in the loss, and reported openly
            rather than buried.
          </p>
        </div>

        <div class="scale-card">
          <p class="scale-label">The Fitzpatrick scale Datice balances across</p>
          <div class="scale">
            <div v-for="t in tones" :key="t.n" class="tone">
              <span class="chip" :style="{ background: t.c }"></span>
              <span class="rn">{{ t.n }}</span>
              <span class="nm">{{ t.name }}</span>
            </div>
          </div>
        </div>
      </div>

      <div class="cards">
        <article v-for="c in methods" :key="c.title" class="mcard">
          <span class="ic" v-html="c.icon" aria-hidden="true"></span>
          <h3>{{ c.title }}</h3>
          <p>{{ c.body }}</p>
        </article>
      </div>
    </div>
  </section>
</template>

<script setup>
const tones = [
  { n: 'I', name: 'Very light', c: 'var(--skin-1)' },
  { n: 'II', name: 'Light', c: 'var(--skin-2)' },
  { n: 'III', name: 'Intermediate', c: 'var(--skin-3)' },
  { n: 'IV', name: 'Tan', c: 'var(--skin-4)' },
  { n: 'V', name: 'Brown', c: 'var(--skin-5)' },
  { n: 'VI', name: 'Dark brown', c: 'var(--skin-6)' },
]

const methods = [
  {
    title: 'Measured, not assumed',
    body: 'Skin tone is computed from the Individual Typology Angle across eight lesion-free patches, so the estimate reflects the person, not the mole.',
    icon: `<svg viewBox="0 0 24 24" width="20" height="20"><circle cx="12" cy="12" r="8.5" fill="none" stroke="currentColor" stroke-width="1.8"/><path d="M12 3.5v17" stroke="currentColor" stroke-width="1.8"/></svg>`,
  },
  {
    title: 'Trained to equalise',
    body: 'The loss adds an Equalized-Odds penalty and per-class recall weighting, pushing the model toward similar sensitivity for light and dark skin.',
    icon: `<svg viewBox="0 0 24 24" width="20" height="20"><path d="M12 3v18M5 8h14M7 8l-3 6a3 3 0 0 0 6 0L7 8Zm10 0-3 6a3 3 0 0 0 6 0l-3-6Z" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/></svg>`,
  },
  {
    title: 'Reported honestly',
    body: 'The Equalized-Odds gap is published alongside accuracy. Closing it fully is hard and ongoing — so we show the number instead of hiding it.',
    icon: `<svg viewBox="0 0 24 24" width="20" height="20"><path d="M4 19V5m0 14h16M8 15l3-4 3 3 4-6" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>`,
  },
]
</script>

<style scoped>
.fairness {
  padding-block: var(--section-y);
  scroll-margin-top: 84px;
}
.split {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: clamp(1.6rem, 1rem + 3vw, 3.5rem);
  align-items: center;
}
.intro h2 {
  font-size: var(--fs-h1);
  margin: 0.7rem 0 0.9rem;
}
.lead {
  font-size: var(--fs-lead);
  color: var(--ink-soft);
  line-height: 1.62;
}
.scale-card {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: var(--r-xl);
  padding: clamp(1.3rem, 1rem + 1.5vw, 2rem);
  box-shadow: var(--shadow-md);
}
.scale-label {
  font-size: 0.8rem;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--ink-faint);
  margin-bottom: 1.1rem;
}
.scale {
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  gap: 0.55rem;
}
.tone {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.4rem;
  text-align: center;
}
.chip {
  width: 100%;
  height: 68px;
  border-radius: var(--r-md);
  box-shadow: inset 0 0 0 1px rgba(0, 0, 0, 0.06);
  transition: transform 0.2s var(--ease);
}
.tone:hover .chip {
  transform: translateY(-4px);
}
.rn {
  font-family: var(--font-display);
  font-weight: 800;
  font-size: 0.95rem;
  color: var(--ink);
}
.nm {
  font-size: 0.68rem;
  color: var(--ink-faint);
  line-height: 1.2;
}
.cards {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1.2rem;
  margin-top: clamp(2rem, 1.4rem + 2vw, 3.4rem);
}
.mcard {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: var(--r-lg);
  padding: 1.5rem;
  box-shadow: var(--shadow-sm);
}
.ic {
  display: inline-grid;
  place-items: center;
  width: 42px;
  height: 42px;
  border-radius: var(--r-md);
  color: var(--coral-deep);
  background: var(--coral-wash);
  margin-bottom: 1rem;
}
.mcard h3 {
  font-size: 1.12rem;
  margin-bottom: 0.5rem;
}
.mcard p {
  color: var(--ink-soft);
  font-size: 0.94rem;
  line-height: 1.55;
}

@media (max-width: 880px) {
  .split {
    grid-template-columns: 1fr;
  }
}
@media (max-width: 720px) {
  .cards {
    grid-template-columns: 1fr;
  }
}
@media (max-width: 420px) {
  .scale {
    gap: 0.35rem;
  }
  .chip {
    height: 52px;
  }
  .nm {
    display: none;
  }
}
</style>
