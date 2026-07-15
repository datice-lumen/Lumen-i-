<template>
  <section id="top" class="hero">
    <div class="shell grid">
      <div class="copy">
        <p class="eyebrow">Fairness-first melanoma screening</p>
        <h1 class="headline">
          See your skin<br />in a <span class="text-gradient">new light</span>.
        </h1>
        <p class="sub">
          Datice reads a skin-lesion photo, tells you whether it's worth a closer look,
          and shows exactly how it got there — hair cleared, skin tone measured, and the
          precise pixels it focused on. Built to stay accurate across every skin tone.
        </p>

        <div class="cta-row">
          <a class="btn primary" href="#analyzer">
            Analyze a photo
            <svg viewBox="0 0 24 24" width="17" height="17" aria-hidden="true">
              <path d="M5 12h14m-6-6 6 6-6 6" fill="none" stroke="currentColor"
                stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />
            </svg>
          </a>
          <a class="btn ghost" href="#how">How it works</a>
        </div>

        <ul class="chips">
          <li>Explainable by design</li>
          <li>Even across skin tones</li>
          <li>0.86 AUC on ISIC 2020</li>
        </ul>
      </div>

      <div class="visual" aria-hidden="true">
        <div class="lens">
          <div class="halo"></div>
          <svg class="reticle" viewBox="0 0 300 300">
            <defs>
              <radialGradient id="heroCore" cx="50%" cy="46%" r="55%">
                <stop offset="0%" stop-color="#ffe9c4" />
                <stop offset="45%" stop-color="#f5a623" />
                <stop offset="100%" stop-color="#f4726b" />
              </radialGradient>
              <radialGradient id="heroLesion" cx="48%" cy="46%" r="60%">
                <stop offset="0%" stop-color="#7a5340" />
                <stop offset="100%" stop-color="#3a241b" />
              </radialGradient>
            </defs>
            <circle cx="150" cy="150" r="140" fill="none" stroke="#e7d7c8" stroke-width="1.5" opacity="0.7" />
            <circle cx="150" cy="150" r="110" fill="none" stroke="#e7d7c8" stroke-width="1.5" opacity="0.5" />
            <circle cx="150" cy="150" r="96" fill="url(#heroCore)" opacity="0.22" />
            <ellipse cx="150" cy="150" rx="52" ry="46" fill="url(#heroLesion)" />
            <ellipse cx="134" cy="138" rx="18" ry="15" fill="#5c3a2b" opacity="0.7" />
            <ellipse cx="166" cy="162" rx="20" ry="16" fill="#2f1d15" opacity="0.8" />
            <line x1="150" y1="6" x2="150" y2="26" stroke="#f4726b" stroke-width="2.5" stroke-linecap="round" />
            <line x1="150" y1="274" x2="150" y2="294" stroke="#f4726b" stroke-width="2.5" stroke-linecap="round" />
            <line x1="6" y1="150" x2="26" y2="150" stroke="#f4726b" stroke-width="2.5" stroke-linecap="round" />
            <line x1="274" y1="150" x2="294" y2="150" stroke="#f4726b" stroke-width="2.5" stroke-linecap="round" />
          </svg>
          <div class="spectrum-ring"></div>
        </div>

        <div class="float f-cam">
          <span class="dot"></span> Grad-CAM focus
        </div>
        <div class="float f-tone">
          <span class="swatches"></span> 6 skin tones
        </div>
        <div class="float f-odds">Equalized odds</div>
      </div>
    </div>
  </section>
</template>

<script setup></script>

<style scoped>
.hero {
  padding-top: clamp(2.5rem, 1.5rem + 4vw, 5rem);
  padding-bottom: var(--section-y);
  overflow: clip;
}
.grid {
  display: grid;
  grid-template-columns: 1.05fr 0.95fr;
  gap: clamp(2rem, 1rem + 4vw, 4rem);
  align-items: center;
}
.headline {
  font-size: var(--fs-display);
  margin: 1rem 0 1.3rem;
  letter-spacing: -0.03em;
}
.sub {
  font-size: var(--fs-lead);
  color: var(--ink-soft);
  max-width: 52ch;
  line-height: 1.62;
}
.cta-row {
  display: flex;
  gap: 0.8rem;
  flex-wrap: wrap;
  margin: 2rem 0 1.6rem;
}
.btn {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  text-decoration: none;
  font-weight: 700;
  font-size: 1rem;
  border-radius: var(--r-pill);
  padding: 0.85rem 1.5rem;
  transition: transform 0.2s var(--ease), box-shadow 0.2s var(--ease),
    border-color 0.2s var(--ease), color 0.2s var(--ease);
}
.btn.primary {
  color: #fff;
  background: var(--glow);
  box-shadow: var(--shadow-glow);
}
.btn.primary:hover {
  transform: translateY(-2px);
  box-shadow: 0 20px 54px rgba(244, 114, 107, 0.42);
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
.chips {
  display: flex;
  flex-wrap: wrap;
  gap: 0.6rem 1.4rem;
  list-style: none;
  margin: 0;
  padding: 0;
}
.chips li {
  position: relative;
  padding-left: 1.3rem;
  color: var(--ink-2);
  font-weight: 600;
  font-size: 0.92rem;
}
.chips li::before {
  content: '';
  position: absolute;
  left: 0;
  top: 50%;
  transform: translateY(-50%);
  width: 9px;
  height: 9px;
  border-radius: 50%;
  background: var(--glow);
}

/* --- signature illuminated lens --- */
.visual {
  position: relative;
  display: grid;
  place-items: center;
  min-height: 360px;
  animation: rise 0.9s var(--ease) both;
}
@keyframes rise {
  from {
    opacity: 0;
    transform: translateY(16px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}
.lens {
  position: relative;
  width: min(88%, 380px);
  aspect-ratio: 1;
  border-radius: 50%;
  display: grid;
  place-items: center;
  background: radial-gradient(circle at 50% 45%, #fffdf9, #fbf1e6);
  box-shadow: var(--shadow-lg), inset 0 0 60px rgba(245, 166, 35, 0.14);
}
.halo {
  position: absolute;
  inset: -14%;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(245, 166, 35, 0.28), rgba(244, 114, 107, 0.12) 45%, transparent 68%);
  filter: blur(6px);
  z-index: -1;
}
.reticle {
  width: 82%;
  height: 82%;
}
.spectrum-ring {
  position: absolute;
  inset: 3%;
  border-radius: 50%;
  padding: 5px;
  background: conic-gradient(
    var(--skin-1),
    var(--skin-2),
    var(--skin-3),
    var(--skin-4),
    var(--skin-5),
    var(--skin-6),
    var(--skin-1)
  );
  -webkit-mask: radial-gradient(farthest-side, transparent calc(100% - 6px), #000 calc(100% - 5px));
  mask: radial-gradient(farthest-side, transparent calc(100% - 6px), #000 calc(100% - 5px));
  opacity: 0.9;
}
.float {
  position: absolute;
  display: inline-flex;
  align-items: center;
  gap: 0.45rem;
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: var(--r-pill);
  padding: 0.5rem 0.85rem;
  font-size: 0.82rem;
  font-weight: 700;
  color: var(--ink-2);
  box-shadow: var(--shadow-md);
}
.f-cam {
  top: 8%;
  right: -2%;
}
.f-tone {
  bottom: 12%;
  left: -4%;
}
.f-odds {
  bottom: 2%;
  right: 8%;
  color: var(--coral-deep);
}
.float .dot {
  width: 9px;
  height: 9px;
  border-radius: 50%;
  background: var(--malignant);
  box-shadow: 0 0 8px var(--malignant);
}
.float .swatches {
  width: 44px;
  height: 10px;
  border-radius: var(--r-pill);
  background: var(--skin-spectrum);
}

@media (max-width: 880px) {
  .grid {
    grid-template-columns: 1fr;
  }
  .visual {
    order: -1;
    min-height: 300px;
  }
  .lens {
    width: min(76%, 320px);
  }
}
</style>
