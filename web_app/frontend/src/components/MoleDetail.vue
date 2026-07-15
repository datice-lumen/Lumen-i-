<template>
  <div class="detail">
    <div class="bar">
      <button type="button" class="back" @click="$emit('back')">
        <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
          <path d="M15 6l-6 6 6 6" fill="none" stroke="currentColor" stroke-width="2"
            stroke-linecap="round" stroke-linejoin="round" />
        </svg>
        All moles
      </button>

      <div v-if="!confirmDel" class="del-wrap">
        <button type="button" class="del" @click="confirmDel = true">Delete mole</button>
      </div>
      <div v-else class="del-wrap confirming">
        <span>Delete “{{ mole.label }}” and its {{ entries.length }} check(s)?</span>
        <button type="button" class="del" @click="removeMole">Delete</button>
        <button type="button" class="ghost-sm" @click="confirmDel = false">Cancel</button>
      </div>
    </div>

    <header class="head">
      <h3>{{ mole.label }}</h3>
      <p class="meta">
        {{ entries.length }} check{{ entries.length === 1 ? '' : 's' }} ·
        first logged {{ fmt(entries[0]?.date) }}
      </p>
    </header>

    <TrendSparkline :points="trendPoints" />

    <!-- side-by-side compare -->
    <section v-if="entries.length > 1" class="compare">
      <h4>Compare two dates</h4>
      <div class="cols">
        <div class="col">
          <select v-model="aId" aria-label="First date to compare">
            <option v-for="e in entries" :key="e.id" :value="e.id">{{ fmt(e.date) }}</option>
          </select>
          <figure v-if="entryA">
            <img :src="entryA.thumb" :alt="`Photo from ${fmt(entryA.date)}`" />
            <figcaption>
              <span class="pct" :class="entryA.predictedClass === 1 ? 'mal' : 'ben'">
                {{ pct(entryA) }}
              </span>
            </figcaption>
          </figure>
        </div>

        <div class="delta" v-if="entryA && entryB">
          <span class="darrow" :class="deltaClass">{{ deltaLabel }}</span>
          <span class="dnote">change in malignancy %</span>
        </div>

        <div class="col">
          <select v-model="bId" aria-label="Second date to compare">
            <option v-for="e in entries" :key="e.id" :value="e.id">{{ fmt(e.date) }}</option>
          </select>
          <figure v-if="entryB">
            <img :src="entryB.thumb" :alt="`Photo from ${fmt(entryB.date)}`" />
            <figcaption>
              <span class="pct" :class="entryB.predictedClass === 1 ? 'mal' : 'ben'">
                {{ pct(entryB) }}
              </span>
            </figcaption>
          </figure>
        </div>
      </div>
      <p class="hint">
        Compare the photos with your own eyes — colour, border, and size. The % is the
        model’s read, not a measurement of the mole.
      </p>
    </section>

    <!-- full timeline -->
    <section class="timeline">
      <h4>All checks</h4>
      <ul>
        <li v-for="e in entriesDesc" :key="e.id">
          <img :src="e.thumb" :alt="`Photo from ${fmt(e.date)}`" />
          <div class="tinfo">
            <span class="tdate">{{ fmt(e.date) }}</span>
            <span class="badges">
              <span class="pct sm" :class="e.predictedClass === 1 ? 'mal' : 'ben'">{{ pct(e) }}</span>
              <span v-if="e.skinGroup" class="muted">Tone {{ roman(e.skinGroup) }}</span>
              <span v-if="metaOf(e)" class="muted">{{ metaOf(e) }}</span>
              <span v-if="e.sample" class="sample">sample</span>
            </span>
          </div>
          <button
            v-if="confirmEntry !== e.id"
            type="button"
            class="trash"
            :aria-label="`Delete check from ${fmt(e.date)}`"
            @click="confirmEntry = e.id"
          >
            <svg viewBox="0 0 24 24" width="17" height="17"><path
              d="M4 7h16M9 7V5h6v2m-7 0 1 13h6l1-13" fill="none" stroke="currentColor"
              stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" /></svg>
          </button>
          <span v-else class="entry-confirm">
            <button type="button" class="del sm" @click="removeEntry(e.id)">Delete</button>
            <button type="button" class="ghost-sm" @click="confirmEntry = ''">Cancel</button>
          </span>
        </li>
      </ul>
    </section>
  </div>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import { useMoleTracker, sortedEntries } from '../composables/useMoleTracker'
import { metadataSummary } from '../composables/metadata'
import TrendSparkline from './TrendSparkline.vue'

const props = defineProps({
  mole: { type: Object, required: true },
})
const emit = defineEmits(['back'])

const { deleteEntry, deleteMole } = useMoleTracker()

const confirmDel = ref(false)
const confirmEntry = ref('')

const entries = computed(() => sortedEntries(props.mole)) // oldest → newest
const entriesDesc = computed(() => [...entries.value].reverse())
const trendPoints = computed(() => entries.value.map((e) => ({ date: e.date, value: e.probability })))

const aId = ref('')
const bId = ref('')
watch(
  entries,
  (list) => {
    const ids = list.map((e) => e.id)
    if (!ids.includes(aId.value)) aId.value = ids[0] || ''
    if (!ids.includes(bId.value)) bId.value = ids[ids.length - 1] || ''
  },
  { immediate: true },
)

const entryA = computed(() => entries.value.find((e) => e.id === aId.value) || null)
const entryB = computed(() => entries.value.find((e) => e.id === bId.value) || null)

const deltaVal = computed(() => {
  if (!entryA.value || !entryB.value) return null
  if (entryA.value.probability == null || entryB.value.probability == null) return null
  return Math.round((entryB.value.probability - entryA.value.probability) * 100)
})
const deltaLabel = computed(() => {
  if (deltaVal.value == null) return '—'
  const v = deltaVal.value
  return `${v > 0 ? '+' : ''}${v} pts`
})
const deltaClass = computed(() =>
  deltaVal.value == null ? '' : deltaVal.value > 0 ? 'up' : deltaVal.value < 0 ? 'down' : 'flat',
)

function pct(e) {
  return e.probability == null ? '—' : `${Math.round(e.probability * 100)}%`
}
function roman(sg) {
  return (sg.split('(')[0] || '').trim()
}
function metaOf(e) {
  return metadataSummary(e.metadata)
}
function fmt(dateStr) {
  if (!dateStr) return '—'
  const [y, m, d] = dateStr.split('-').map(Number)
  const dt = new Date(y, (m || 1) - 1, d || 1)
  return dt.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' })
}

function removeMole() {
  deleteMole(props.mole.id)
  emit('back')
}
function removeEntry(id) {
  confirmEntry.value = ''
  deleteEntry(props.mole.id, id)
}
</script>

<style scoped>
.detail {
  animation: fade 0.3s var(--ease);
}
@keyframes fade {
  from {
    opacity: 0;
    transform: translateY(6px);
  }
}
.bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  flex-wrap: wrap;
  margin-bottom: 1.4rem;
}
.back {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  border: none;
  background: transparent;
  color: var(--ink-2);
  font-weight: 700;
  font-size: 0.94rem;
  cursor: pointer;
  padding: 0.4rem 0.2rem;
  border-radius: var(--r-pill);
}
.back:hover {
  color: var(--coral-deep);
}
.del-wrap {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  font-size: 0.88rem;
  color: var(--ink-soft);
}
.del {
  border: 1px solid transparent;
  background: transparent;
  color: var(--malignant-ink);
  font-weight: 700;
  font-size: 0.88rem;
  cursor: pointer;
  padding: 0.4rem 0.7rem;
  border-radius: var(--r-pill);
}
.del:hover {
  background: var(--malignant-wash);
}
.del.sm {
  padding: 0.25rem 0.6rem;
  font-size: 0.82rem;
}
.ghost-sm {
  border: 1px solid var(--line-strong);
  background: var(--surface);
  color: var(--ink-2);
  font-weight: 700;
  font-size: 0.82rem;
  cursor: pointer;
  padding: 0.25rem 0.65rem;
  border-radius: var(--r-pill);
}
.head {
  margin-bottom: 1.3rem;
}
.head h3 {
  font-size: var(--fs-h3);
}
.meta {
  color: var(--ink-faint);
  font-size: 0.9rem;
  margin-top: 0.3rem;
}
h4 {
  font-size: 1.05rem;
  margin: 1.7rem 0 0.9rem;
}
.compare .cols {
  display: grid;
  grid-template-columns: 1fr auto 1fr;
  gap: 1rem;
  align-items: start;
}
.col select {
  width: 100%;
  font-family: var(--font-body);
  font-size: 0.92rem;
  padding: 0.55rem 0.7rem;
  border: 1px solid var(--line-strong);
  border-radius: var(--r-md);
  background: var(--surface);
  margin-bottom: 0.7rem;
}
.col figure {
  margin: 0;
}
.col img {
  width: 100%;
  aspect-ratio: 1;
  object-fit: cover;
  border-radius: var(--r-md);
  border: 1px solid var(--line);
}
.col figcaption {
  margin-top: 0.6rem;
  text-align: center;
}
.pct {
  font-weight: 800;
  font-size: 0.9rem;
  padding: 0.25rem 0.6rem;
  border-radius: var(--r-pill);
}
.pct.sm {
  font-size: 0.8rem;
}
.pct.ben {
  color: var(--benign-ink);
  background: var(--benign-wash);
}
.pct.mal {
  color: var(--malignant-ink);
  background: var(--malignant-wash);
}
.delta {
  align-self: center;
  text-align: center;
  padding-top: 2.6rem;
}
.darrow {
  display: block;
  font-family: var(--font-display);
  font-weight: 800;
  font-size: 1.1rem;
  font-variant-numeric: tabular-nums;
}
.darrow.up {
  color: var(--malignant-ink);
}
.darrow.down {
  color: var(--benign-ink);
}
.darrow.flat {
  color: var(--ink-soft);
}
.dnote {
  font-size: 0.68rem;
  color: var(--ink-faint);
}
.hint {
  margin-top: 0.9rem;
  font-size: 0.85rem;
  color: var(--ink-faint);
  line-height: 1.5;
}
.timeline ul {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 0.7rem;
}
.timeline li {
  display: flex;
  align-items: center;
  gap: 0.9rem;
  padding: 0.6rem;
  border: 1px solid var(--line);
  border-radius: var(--r-md);
  background: var(--surface);
}
.timeline img {
  width: 56px;
  height: 56px;
  border-radius: var(--r-sm);
  object-fit: cover;
  flex: none;
}
.tinfo {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
}
.tdate {
  font-weight: 700;
  color: var(--ink);
}
.badges {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex-wrap: wrap;
}
.muted {
  font-size: 0.8rem;
  color: var(--ink-faint);
}
.sample {
  font-size: 0.66rem;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--amber);
  border: 1px dashed var(--amber);
  border-radius: var(--r-pill);
  padding: 0.1rem 0.45rem;
}
.trash {
  border: none;
  background: transparent;
  color: var(--ink-faint);
  cursor: pointer;
  padding: 0.4rem;
  border-radius: 50%;
  flex: none;
}
.trash:hover {
  color: var(--malignant);
  background: var(--malignant-wash);
}
.entry-confirm {
  display: flex;
  gap: 0.4rem;
  flex: none;
}

@media (max-width: 620px) {
  .compare .cols {
    grid-template-columns: 1fr;
  }
  .delta {
    padding-top: 0;
  }
}
</style>
