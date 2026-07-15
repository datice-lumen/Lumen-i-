<template>
  <span class="logo" :style="{ '--sz': size + 'px' }">
    <svg
      class="mark"
      viewBox="0 0 40 40"
      role="img"
      aria-label="Datice"
      focusable="false"
    >
      <defs>
        <radialGradient id="brandCore" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stop-color="#ffe6b8" />
          <stop offset="45%" stop-color="#f5a623" />
          <stop offset="100%" stop-color="#f4726b" />
        </radialGradient>
      </defs>
      <!-- dermatoscope ring -->
      <circle cx="20" cy="20" r="16.5" fill="none" stroke="#e7d7c8" stroke-width="2.4" />
      <!-- luminous core -->
      <circle cx="20" cy="20" r="9" fill="url(#brandCore)" />
      <!-- light rays -->
      <g stroke="#f4726b" stroke-width="2.2" stroke-linecap="round" opacity="0.9">
        <line x1="20" y1="1.5" x2="20" y2="5.5" />
        <line x1="20" y1="34.5" x2="20" y2="38.5" />
        <line x1="1.5" y1="20" x2="5.5" y2="20" />
        <line x1="34.5" y1="20" x2="38.5" y2="20" />
      </g>
    </svg>
    <span v-if="withWordmark" class="word">Datice</span>
  </span>
</template>

<script setup>
defineProps({
  size: { type: Number, default: 30 },
  withWordmark: { type: Boolean, default: true },
})
</script>

<style scoped>
.logo {
  display: inline-flex;
  align-items: center;
  gap: 0.55rem;
  line-height: 1;
}
.mark {
  width: var(--sz);
  height: var(--sz);
  flex: none;
  filter: drop-shadow(0 2px 6px rgba(244, 114, 107, 0.28));
}
.word {
  font-family: var(--font-display);
  font-weight: 800;
  font-size: calc(var(--sz) * 0.72);
  letter-spacing: -0.02em;
  color: var(--ink);
}
</style>
