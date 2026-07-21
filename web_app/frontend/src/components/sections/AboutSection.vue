<template>
  <section id="about" class="about">
    <div class="shell">
      <div class="split">
        <div class="story">
          <p class="eyebrow">About</p>
          <h2>Open source, and it shows its work</h2>
          <p>
            <strong>Datice</strong> is a melanoma screening tool built around one idea: a model
            you can see into. Rather than fine-tuning a black box, we trained a custom CNN from
            scratch — so every stage, from photo to verdict, is ours to inspect and explain.
          </p>
          <p>
            Every stage of the pipeline, the model architecture, and the training code are
            open source. This app is the interactive window onto that work: upload a photo
            and watch the same pipeline run end to end.
          </p>

          <div class="facts">
            <div class="fact">
              <span class="fv">6.7M</span>
              <span class="fl">parameters, from scratch</span>
            </div>
            <div class="fact">
              <span class="fv">33k+</span>
              <span class="fl">training images</span>
            </div>
            <div class="fact">
              <span class="fv">5-fold</span>
              <span class="fl">patient-level splits</span>
            </div>
          </div>

          <div class="links">
            <a class="btn ghost" href="https://github.com/datice-lumen/Lumen-i-" target="_blank" rel="noopener">
              View the code
            </a>
            <a class="btn text" href="https://github.com/datice-lumen/Lumen-i-/wiki" target="_blank" rel="noopener">
              Read the methodology →
            </a>
          </div>
        </div>

        <aside class="disclaimer" role="note">
          <span class="ic" aria-hidden="true">
            <svg viewBox="0 0 24 24" width="24" height="24">
              <circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" stroke-width="1.8" />
              <path d="M12 8h.01M11 12h1v4h1" fill="none" stroke="currentColor" stroke-width="1.8"
                stroke-linecap="round" stroke-linejoin="round" />
            </svg>
          </span>
          <h3>Not a medical diagnosis</h3>
          <p>
            Datice is a research and education tool. It is not a certified medical device,
            has not been clinically validated, and can be wrong in both directions.
          </p>
          <p>
            If you're worried about a mole or a change in your skin, see a dermatologist.
            Nothing here replaces professional care.
          </p>
        </aside>
      </div>
    </div>
  </section>
</template>

<script setup></script>

<style scoped>
.about {
  padding-block: var(--section-y);
  scroll-margin-top: 84px;
}
.split {
  display: grid;
  grid-template-columns: 1.4fr 1fr;
  gap: clamp(1.6rem, 1rem + 3vw, 3.5rem);
  align-items: start;
}
.story h2 {
  font-size: var(--fs-h1);
  margin: 0.7rem 0 1.1rem;
}
.story p {
  color: var(--ink-soft);
  font-size: var(--fs-lead);
  line-height: 1.62;
  margin-bottom: 1rem;
}
.story strong {
  color: var(--ink);
}
.facts {
  display: flex;
  flex-wrap: wrap;
  gap: 1rem 2rem;
  margin: 1.6rem 0;
  padding: 1.3rem 0;
  border-top: 1px solid var(--line);
  border-bottom: 1px solid var(--line);
}
.fact {
  display: flex;
  flex-direction: column;
}
.fv {
  font-family: var(--font-display);
  font-weight: 800;
  font-size: 1.55rem;
  color: var(--coral-deep);
}
.fl {
  font-size: 0.85rem;
  color: var(--ink-faint);
}
.links {
  display: flex;
  gap: 0.8rem;
  flex-wrap: wrap;
  margin-top: 1.6rem;
}
.btn {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  text-decoration: none;
  font-weight: 700;
  font-size: 0.95rem;
  border-radius: var(--r-pill);
  padding: 0.7rem 1.3rem;
  transition: border-color 0.2s var(--ease), color 0.2s var(--ease), background 0.2s var(--ease);
}
.btn.ghost {
  color: var(--ink);
  background: var(--surface);
  border: 1px solid var(--line-strong);
}
.btn.ghost:hover {
  border-color: var(--coral);
  color: var(--coral-deep);
}
.btn.text {
  color: var(--coral-deep);
}
.btn.text:hover {
  background: var(--coral-wash);
}
.disclaimer {
  background: var(--coral-wash);
  border: 1px solid rgba(244, 114, 107, 0.3);
  border-radius: var(--r-xl);
  padding: clamp(1.4rem, 1rem + 1.5vw, 2rem);
  box-shadow: var(--shadow-sm);
}
.ic {
  display: inline-grid;
  place-items: center;
  width: 48px;
  height: 48px;
  border-radius: 50%;
  color: #fff;
  background: var(--coral);
  margin-bottom: 1rem;
}
.disclaimer h3 {
  font-size: 1.25rem;
  margin-bottom: 0.7rem;
}
.disclaimer p {
  color: var(--ink-2);
  font-size: 0.95rem;
  line-height: 1.55;
  margin-bottom: 0.8rem;
}
.disclaimer p:last-child {
  margin-bottom: 0;
}

@media (max-width: 880px) {
  .split {
    grid-template-columns: 1fr;
  }
}
</style>
