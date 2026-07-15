import { computed, reactive, watch } from 'vue'

// Single source of truth for tracked moles, persisted to localStorage only.
// Shared across components via a module-scoped reactive singleton.

const KEY = 'datice.moles.v1'

const state = reactive({
  moles: [],
  error: '',
})

let initialized = false

function uid() {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) return crypto.randomUUID()
  return 'id-' + Math.random().toString(36).slice(2) + Date.now().toString(36)
}

function todayISO() {
  return new Date().toISOString().slice(0, 10)
}

function load() {
  try {
    const raw = localStorage.getItem(KEY)
    if (!raw) return
    const data = JSON.parse(raw)
    if (data && Array.isArray(data.moles)) state.moles = data.moles
  } catch {
    // corrupt or unavailable storage — start empty rather than crash
    state.moles = []
  }
}

function persist() {
  try {
    localStorage.setItem(KEY, JSON.stringify({ version: 1, moles: state.moles }))
    state.error = ''
  } catch {
    state.error = 'Storage is full — delete some older checks to save new ones.'
  }
}

function init() {
  if (initialized) return
  initialized = true
  load()
  watch(() => state.moles, persist, { deep: true })
}

export function useMoleTracker() {
  init()

  function createMole(label) {
    const mole = {
      id: uid(),
      label: (label || '').trim() || 'Untitled mole',
      createdAt: new Date().toISOString(),
      entries: [],
    }
    state.moles.push(mole)
    return mole
  }

  function addEntry(moleId, entry) {
    const mole = state.moles.find((m) => m.id === moleId)
    if (!mole) return null
    const record = {
      id: uid(),
      date: entry.date || todayISO(),
      thumb: entry.thumb,
      probability: entry.probability ?? null,
      predictedClass: entry.predictedClass ?? null,
      skinGroup: entry.skinGroup || '',
      // metadata the model reasoned with, e.g. { age, sex, anatom_site }; null if none
      metadata: entry.metadata || null,
      sample: !!entry.sample,
    }
    mole.entries.push(record)
    return record
  }

  function deleteEntry(moleId, entryId) {
    const mole = state.moles.find((m) => m.id === moleId)
    if (!mole) return
    mole.entries = mole.entries.filter((e) => e.id !== entryId)
  }

  function deleteMole(moleId) {
    state.moles = state.moles.filter((m) => m.id !== moleId)
  }

  function clearAll() {
    state.moles = []
  }

  const moleCount = computed(() => state.moles.length)

  return {
    state,
    moleCount,
    createMole,
    addEntry,
    deleteEntry,
    deleteMole,
    clearAll,
    todayISO,
  }
}

// entries sorted oldest → newest (for trend/compare)
export function sortedEntries(mole) {
  return [...(mole?.entries || [])].sort((a, b) => (a.date < b.date ? -1 : a.date > b.date ? 1 : 0))
}
