<template>
  <header class="header" :class="{ scrolled }">
    <div class="shell bar">
      <a href="#top" class="brand" aria-label="Datice home">
        <BrandLogo :size="30" />
      </a>

      <nav class="nav" aria-label="Sections">
        <a href="#analyzer">Analyzer</a>
        <a href="#moles">My moles</a>
        <a href="#how">How it works</a>
        <a href="#results">Results</a>
        <a href="#about">About</a>
      </nav>

      <div class="actions">
        <a
          class="ghost"
          href="https://github.com/datice-lumen/Lumen-i-"
          target="_blank"
          rel="noopener"
        >
          GitHub
        </a>
        <a class="cta" href="#analyzer">Analyze a photo</a>
      </div>
    </div>
  </header>
</template>

<script setup>
import { onMounted, onUnmounted, ref } from 'vue'
import BrandLogo from './BrandLogo.vue'

const scrolled = ref(false)
const onScroll = () => {
  scrolled.value = window.scrollY > 12
}
onMounted(() => {
  window.addEventListener('scroll', onScroll, { passive: true })
  onScroll()
})
onUnmounted(() => window.removeEventListener('scroll', onScroll))
</script>

<style scoped>
.header {
  position: sticky;
  top: 0;
  z-index: 50;
  transition: background 0.3s var(--ease), box-shadow 0.3s var(--ease),
    border-color 0.3s var(--ease);
  border-bottom: 1px solid transparent;
}
.header.scrolled {
  background: rgba(251, 246, 241, 0.82);
  backdrop-filter: saturate(1.4) blur(14px);
  -webkit-backdrop-filter: saturate(1.4) blur(14px);
  border-bottom-color: var(--line);
  box-shadow: 0 6px 24px rgba(90, 55, 40, 0.06);
}
.bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  height: 68px;
}
.brand {
  text-decoration: none;
  border-radius: var(--r-pill);
}
.nav {
  display: flex;
  gap: 0.35rem;
  align-items: center;
}
.nav a {
  text-decoration: none;
  color: var(--ink-soft);
  font-weight: 600;
  font-size: 0.94rem;
  padding: 0.5rem 0.7rem;
  border-radius: var(--r-pill);
  transition: color 0.2s var(--ease), background 0.2s var(--ease);
}
.nav a:hover {
  color: var(--ink);
  background: rgba(244, 114, 107, 0.08);
}
.actions {
  display: flex;
  align-items: center;
  gap: 0.6rem;
}
.ghost {
  text-decoration: none;
  color: var(--ink-2);
  font-weight: 700;
  font-size: 0.92rem;
  padding: 0.55rem 0.9rem;
  border-radius: var(--r-pill);
  transition: background 0.2s var(--ease);
}
.ghost:hover {
  background: rgba(42, 33, 29, 0.06);
}
.cta {
  text-decoration: none;
  color: #fff;
  font-weight: 700;
  font-size: 0.92rem;
  padding: 0.6rem 1.15rem;
  border-radius: var(--r-pill);
  background: var(--glow);
  box-shadow: var(--shadow-glow);
  transition: transform 0.2s var(--ease), box-shadow 0.2s var(--ease);
}
.cta:hover {
  transform: translateY(-1px);
  box-shadow: 0 18px 50px rgba(244, 114, 107, 0.4);
}

@media (max-width: 860px) {
  .nav {
    display: none;
  }
}
@media (max-width: 520px) {
  .ghost {
    display: none;
  }
  .bar {
    height: 60px;
  }
}
</style>
