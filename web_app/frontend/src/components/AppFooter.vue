<template>
  <footer class="footer">
    <hr class="spectrum-rule" />
    <div class="shell inner">
      <div class="col brandcol">
        <BrandLogo :size="28" />
        <p class="tag">
          Skin-lesion analysis that shows its work — every step from photo to
          verdict, out in the open.
        </p>
      </div>

      <div class="col">
        <h4>Explore</h4>
        <a href="#analyzer">Analyzer</a>
        <a href="#moles">My moles</a>
        <a href="#how">How it works</a>
        <a href="#results">Results</a>
      </div>

      <div class="col">
        <h4>Project</h4>
        <a href="https://github.com/datice-lumen/Lumen-i-" target="_blank" rel="noopener">
          GitHub repository
        </a>
        <a
          href="https://github.com/datice-lumen/Lumen-i-/wiki"
          target="_blank"
          rel="noopener"
          >Documentation</a
        >
      </div>

      <div class="col">
        <h4>Team Datice</h4>
        <span>Jurica Jerinić</span>
        <span>Filip Hlup</span>
        <span>Tomislav Matanović</span>
        <span>Karlo Rastegorac</span>
      </div>
    </div>

    <div class="shell base">
      <p class="disclaimer">
        Educational and research use only. Datice is not a medical device and does not
        replace a dermatologist.
      </p>
      <p class="copy">© 2025 Datice</p>
    </div>
  </footer>
</template>

<script setup>
import BrandLogo from './BrandLogo.vue'
</script>

<style scoped>
.footer {
  margin-top: var(--section-y);
}
.spectrum-rule {
  margin: 0;
  opacity: 0.7;
}
.inner {
  display: grid;
  grid-template-columns: 1.8fr 1fr 1fr 1fr;
  gap: 2rem;
  padding-top: 3rem;
  padding-bottom: 2.2rem;
}
.brandcol .tag {
  margin-top: 0.9rem;
  max-width: 30ch;
  color: var(--ink-soft);
  font-size: 0.95rem;
  line-height: 1.55;
}
.col h4 {
  font-family: var(--font-body);
  font-size: 0.78rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--ink-faint);
  margin-bottom: 0.9rem;
}
.col a,
.col span {
  display: block;
  text-decoration: none;
  color: var(--ink-2);
  font-size: 0.95rem;
  padding: 0.28rem 0;
  transition: color 0.2s var(--ease);
}
.col a:hover {
  color: var(--coral-deep);
}
.base {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  flex-wrap: wrap;
  padding-top: 1.4rem;
  padding-bottom: 2.4rem;
  border-top: 1px solid var(--line);
}
.disclaimer {
  color: var(--ink-faint);
  font-size: 0.86rem;
  max-width: 60ch;
}
.copy {
  color: var(--ink-faint);
  font-size: 0.86rem;
}

@media (max-width: 760px) {
  .inner {
    grid-template-columns: 1fr 1fr;
    gap: 1.6rem;
  }
  .brandcol {
    grid-column: 1 / -1;
  }
}
</style>
