<template>
  <section id="moles" class="moles">
    <div class="shell">
      <header class="sec-head">
        <div>
          <p class="eyebrow">Your history · on this device</p>
          <h2>Track a mole over time</h2>
          <p class="lead">
            Save any analysis to a mole and watch how it changes across weeks and months.
            Everything stays in this browser — nothing is uploaded.
          </p>
        </div>

        <div v-if="state.moles.length" class="clear">
          <button v-if="!confirmClear" type="button" class="ghost-sm" @click="confirmClear = true">
            Clear all
          </button>
          <span v-else class="clear-confirm">
            Delete everything?
            <button type="button" class="del-sm" @click="wipe">Yes, clear</button>
            <button type="button" class="ghost-sm" @click="confirmClear = false">Cancel</button>
          </span>
        </div>
      </header>

      <MoleDetail v-if="selected" :mole="selected" @back="selectedId = ''" />

      <template v-else>
        <div v-if="!state.moles.length" class="empty">
          <span class="ic" aria-hidden="true">
            <svg viewBox="0 0 24 24" width="26" height="26"><path
              d="M3 12s3.5-7 9-7 9 7 9 7M8 18l2-2m4 4 2-2M12 21v-4" fill="none"
              stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" />
              <circle cx="12" cy="12" r="2.6" fill="currentColor" /></svg>
          </span>
          <h3>No moles tracked yet</h3>
          <p>Analyze a photo, then choose <strong>Save to a mole</strong> to start a history.</p>
          <a class="btn primary" href="#analyzer">Analyze a photo</a>
        </div>

        <ul v-else class="grid">
          <li v-for="m in state.moles" :key="m.id">
            <button type="button" class="card" @click="selectedId = m.id">
              <span class="thumb">
                <img v-if="latest(m)" :src="latest(m).thumb" :alt="`Latest photo of ${m.label}`" />
              </span>
              <span class="cbody">
                <span class="clabel">{{ m.label }}</span>
                <span class="csub">
                  {{ m.entries.length }} check{{ m.entries.length === 1 ? '' : 's' }} ·
                  last {{ fmt(latest(m)?.date) }}
                </span>
                <span v-if="latest(m)" class="cbadge" :class="latest(m).predictedClass === 1 ? 'mal' : 'ben'">
                  {{ pct(latest(m)) }} · {{ latest(m).predictedClass === 1 ? 'worth checking' : 'benign' }}
                </span>
              </span>
              <svg class="chev" viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
                <path d="M9 6l6 6-6 6" fill="none" stroke="currentColor" stroke-width="2"
                  stroke-linecap="round" stroke-linejoin="round" />
              </svg>
            </button>
          </li>
        </ul>
      </template>

      <p v-if="state.error" class="store-err">{{ state.error }}</p>
    </div>
  </section>
</template>

<script setup>
import { computed, ref } from 'vue'
import { useMoleTracker, sortedEntries } from '../../composables/useMoleTracker'
import MoleDetail from '../MoleDetail.vue'

const { state, clearAll } = useMoleTracker()

const selectedId = ref('')
const confirmClear = ref(false)

const selected = computed(() => state.moles.find((m) => m.id === selectedId.value) || null)

function latest(mole) {
  const s = sortedEntries(mole)
  return s.length ? s[s.length - 1] : null
}
function pct(e) {
  return e && e.probability != null ? `${Math.round(e.probability * 100)}%` : '—'
}
function fmt(dateStr) {
  if (!dateStr) return '—'
  const [y, m, d] = dateStr.split('-').map(Number)
  return new Date(y, (m || 1) - 1, d || 1).toLocaleDateString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  })
}
function wipe() {
  clearAll()
  confirmClear.value = false
  selectedId.value = ''
}
</script>

<style scoped>
.moles {
  padding-block: var(--section-y);
  scroll-margin-top: 84px;
}
.sec-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 1.2rem;
  flex-wrap: wrap;
  margin-bottom: 2rem;
}
.sec-head h2 {
  font-size: var(--fs-h1);
  margin: 0.7rem 0 0.9rem;
}
.lead {
  font-size: var(--fs-lead);
  color: var(--ink-soft);
  line-height: 1.6;
  max-width: 56ch;
}
.clear {
  flex: none;
  font-size: 0.86rem;
  color: var(--ink-soft);
}
.clear-confirm {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  flex-wrap: wrap;
}
.ghost-sm {
  border: 1px solid var(--line-strong);
  background: var(--surface);
  color: var(--ink-2);
  font-weight: 700;
  font-size: 0.84rem;
  cursor: pointer;
  padding: 0.4rem 0.8rem;
  border-radius: var(--r-pill);
}
.ghost-sm:hover {
  border-color: var(--coral);
}
.del-sm {
  border: none;
  background: var(--malignant);
  color: #fff;
  font-weight: 700;
  font-size: 0.84rem;
  cursor: pointer;
  padding: 0.42rem 0.8rem;
  border-radius: var(--r-pill);
}

/* empty state */
.empty {
  text-align: center;
  padding: clamp(2rem, 1.5rem + 3vw, 3.4rem);
  border: 1px dashed var(--line-strong);
  border-radius: var(--r-xl);
  background: var(--surface-warm);
}
.empty .ic {
  display: inline-grid;
  place-items: center;
  width: 58px;
  height: 58px;
  border-radius: 50%;
  color: var(--coral-deep);
  background: var(--coral-wash);
  margin-bottom: 1rem;
}
.empty h3 {
  font-size: var(--fs-h3);
  margin-bottom: 0.5rem;
}
.empty p {
  color: var(--ink-soft);
  margin-bottom: 1.4rem;
}
.btn {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  text-decoration: none;
  font-weight: 700;
  border-radius: var(--r-pill);
  padding: 0.75rem 1.4rem;
}
.btn.primary {
  color: #fff;
  background: var(--glow);
  box-shadow: var(--shadow-glow);
}

/* grid of moles */
.grid {
  list-style: none;
  margin: 0;
  padding: 0;
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
  gap: 1.1rem;
}
.card {
  width: 100%;
  display: flex;
  align-items: center;
  gap: 1rem;
  text-align: left;
  padding: 0.9rem;
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: var(--r-lg);
  box-shadow: var(--shadow-sm);
  cursor: pointer;
  transition: transform 0.2s var(--ease), box-shadow 0.2s var(--ease), border-color 0.2s var(--ease);
}
.card:hover {
  transform: translateY(-2px);
  box-shadow: var(--shadow-md);
  border-color: var(--line-strong);
}
.thumb {
  width: 64px;
  height: 64px;
  flex: none;
  border-radius: var(--r-md);
  overflow: hidden;
  background: var(--sand-deep);
}
.thumb img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}
.cbody {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 0.3rem;
  min-width: 0;
}
.clabel {
  font-family: var(--font-display);
  font-weight: 700;
  font-size: 1.05rem;
  color: var(--ink);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.csub {
  font-size: 0.82rem;
  color: var(--ink-faint);
}
.cbadge {
  align-self: flex-start;
  font-weight: 700;
  font-size: 0.78rem;
  padding: 0.2rem 0.55rem;
  border-radius: var(--r-pill);
  margin-top: 0.15rem;
}
.cbadge.ben {
  color: var(--benign-ink);
  background: var(--benign-wash);
}
.cbadge.mal {
  color: var(--malignant-ink);
  background: var(--malignant-wash);
}
.chev {
  flex: none;
  color: var(--ink-faint);
}
.store-err {
  margin-top: 1.2rem;
  color: var(--malignant-ink);
  background: var(--malignant-wash);
  border-radius: var(--r-sm);
  padding: 0.7rem 0.9rem;
  font-size: 0.9rem;
}

@media (max-width: 560px) {
  .grid {
    grid-template-columns: 1fr;
  }
}
</style>
