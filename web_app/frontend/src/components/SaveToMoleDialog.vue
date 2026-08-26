<template>
  <Teleport to="body">
    <transition name="dlg">
      <div v-if="open" class="backdrop" @click.self="close">
        <div class="dialog" role="dialog" aria-modal="true" aria-labelledby="save-mole-title">
          <button class="x" type="button" aria-label="Close" @click="close">
            <svg viewBox="0 0 24 24" width="20" height="20"><path d="M6 6l12 12M18 6 6 18"
              fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" /></svg>
          </button>

          <h3 id="save-mole-title">Save this check</h3>
          <p class="sub">Keep this photo and result so you can watch the mole over time.</p>

          <div class="preview">
            <img v-if="imageSrc" :src="imageSrc" alt="Photo being saved" />
            <div class="pmeta">
              <span class="badge" :class="predictedClass === 1 ? 'mal' : 'ben'">
                {{ percent }}% · {{ predictedClass === 1 ? 'worth checking' : 'benign' }}
              </span>
              <span v-if="skinGroup" class="muted">Skin tone {{ roman }}</span>
              <span v-if="metaSummary" class="muted">{{ metaSummary }}</span>
              <span v-if="isDemo" class="sample">sample data</span>
            </div>
          </div>

          <label class="field">
            <span>Add to</span>
            <select v-model="target">
              <option v-for="m in state.moles" :key="m.id" :value="m.id">
                {{ m.label }} ({{ m.entries.length }})
              </option>
              <option value="__new">＋ New mole…</option>
            </select>
          </label>

          <label v-if="target === '__new'" class="field">
            <span>Mole name</span>
            <input
              v-model="label"
              type="text"
              placeholder="e.g. left forearm"
              maxlength="40"
              @keydown.enter.prevent="save"
            />
          </label>

          <label class="field">
            <span>Date of this photo</span>
            <input v-model="date" type="date" :max="today" />
          </label>

          <p v-if="error" class="err">{{ error }}</p>

          <div class="actions">
            <button type="button" class="btn ghost" @click="close">Cancel</button>
            <button type="button" class="btn primary" :disabled="saving || !canSave" @click="save">
              {{ saving ? 'Saving…' : 'Save check' }}
            </button>
          </div>
        </div>
      </div>
    </transition>
  </Teleport>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import { useMoleTracker } from '../composables/useMoleTracker'
import { metadataSummary } from '../composables/metadata'
import { downscaleToThumb } from '../utils/image'

const props = defineProps({
  open: { type: Boolean, default: false },
  imageSrc: { type: String, default: '' },
  probability: { type: Number, default: null },
  predictedClass: { type: Number, default: null },
  skinGroup: { type: String, default: '' },
  metadata: { type: Object, default: null },
  isDemo: { type: Boolean, default: false },
})
const emit = defineEmits(['close', 'saved'])

const { state, createMole, addEntry, todayISO } = useMoleTracker()

const metaSummary = computed(() => metadataSummary(props.metadata))

const today = todayISO()
const target = ref('__new')
const label = ref('')
const date = ref(today)
const saving = ref(false)
const error = ref('')

const percent = computed(() =>
  props.probability == null ? '–' : Math.round(props.probability * 100),
)
const roman = computed(() => (props.skinGroup.split('(')[0] || '').trim())
const canSave = computed(() =>
  target.value === '__new' ? label.value.trim().length > 0 : !!target.value,
)

// reset the form each time the dialog opens
watch(
  () => props.open,
  (isOpen) => {
    if (!isOpen) return
    error.value = ''
    date.value = today
    label.value = ''
    target.value = state.moles.length ? state.moles[0].id : '__new'
  },
)

function close() {
  if (saving.value) return
  emit('close')
}

async function save() {
  if (!canSave.value || saving.value) return
  saving.value = true
  error.value = ''
  try {
    const thumb = await downscaleToThumb(props.imageSrc)
    const moleId = target.value === '__new' ? createMole(label.value).id : target.value
    const mole = state.moles.find((m) => m.id === moleId)
    addEntry(moleId, {
      date: date.value,
      thumb,
      probability: props.probability,
      predictedClass: props.predictedClass,
      skinGroup: props.skinGroup,
      metadata: props.metadata,
      sample: props.isDemo,
    })
    if (state.error) {
      error.value = state.error
      return
    }
    emit('saved', { moleId, label: mole?.label || 'mole' })
  } catch (e) {
    error.value = e?.message || 'Could not save this check.'
  } finally {
    saving.value = false
  }
}
</script>

<style scoped>
.backdrop {
  position: fixed;
  inset: 0;
  z-index: 100;
  display: grid;
  place-items: center;
  padding: 1.2rem;
  background: rgba(42, 33, 29, 0.42);
  backdrop-filter: blur(3px);
}
.dialog {
  position: relative;
  width: min(440px, 100%);
  background: var(--surface);
  border-radius: var(--r-xl);
  padding: clamp(1.4rem, 1rem + 1.5vw, 2rem);
  box-shadow: var(--shadow-lg);
  max-height: 90vh;
  overflow-y: auto;
}
.x {
  position: absolute;
  top: 12px;
  right: 12px;
  display: grid;
  place-items: center;
  width: 36px;
  height: 36px;
  border: none;
  border-radius: 50%;
  background: transparent;
  color: var(--ink-soft);
  cursor: pointer;
  transition: background 0.2s var(--ease);
}
.x:hover {
  background: rgba(42, 33, 29, 0.07);
}
h3 {
  font-size: 1.4rem;
}
.sub {
  color: var(--ink-soft);
  font-size: 0.94rem;
  margin: 0.4rem 0 1.3rem;
}
.preview {
  display: flex;
  gap: 0.9rem;
  align-items: center;
  padding: 0.7rem;
  border: 1px solid var(--line);
  border-radius: var(--r-md);
  background: var(--surface-warm);
  margin-bottom: 1.3rem;
}
.preview img {
  width: 62px;
  height: 62px;
  border-radius: var(--r-sm);
  object-fit: cover;
  flex: none;
}
.pmeta {
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
  align-items: flex-start;
}
.badge {
  font-weight: 700;
  font-size: 0.85rem;
  padding: 0.25rem 0.6rem;
  border-radius: var(--r-pill);
}
.badge.ben {
  color: var(--benign-ink);
  background: var(--benign-wash);
}
.badge.mal {
  color: var(--malignant-ink);
  background: var(--malignant-wash);
}
.muted {
  font-size: 0.82rem;
  color: var(--ink-faint);
}
.sample {
  font-size: 0.7rem;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--amber);
  border: 1px dashed var(--amber);
  border-radius: var(--r-pill);
  padding: 0.15rem 0.5rem;
}
.field {
  display: block;
  margin-bottom: 1rem;
}
.field > span {
  display: block;
  font-weight: 700;
  font-size: 0.82rem;
  color: var(--ink-2);
  margin-bottom: 0.4rem;
}
.field select,
.field input {
  width: 100%;
  font-family: var(--font-body);
  font-size: 0.98rem;
  color: var(--ink);
  padding: 0.7rem 0.85rem;
  border: 1px solid var(--line-strong);
  border-radius: var(--r-md);
  background: var(--surface);
}
.field select:focus,
.field input:focus {
  outline: none;
  border-color: var(--coral);
  box-shadow: 0 0 0 3px var(--coral-wash);
}
.err {
  color: var(--malignant-ink);
  background: var(--malignant-wash);
  border-radius: var(--r-sm);
  padding: 0.6rem 0.8rem;
  font-size: 0.88rem;
  margin-bottom: 1rem;
}
.actions {
  display: flex;
  justify-content: flex-end;
  gap: 0.7rem;
  margin-top: 0.5rem;
}
.btn {
  font-family: var(--font-body);
  font-weight: 700;
  font-size: 0.95rem;
  border-radius: var(--r-pill);
  padding: 0.7rem 1.4rem;
  cursor: pointer;
  border: 1px solid transparent;
  transition: transform 0.2s var(--ease), background 0.2s var(--ease), border-color 0.2s var(--ease);
}
.btn.primary {
  color: #fff;
  background: var(--glow);
  box-shadow: var(--shadow-glow);
}
.btn.primary:disabled {
  opacity: 0.5;
  cursor: not-allowed;
  box-shadow: none;
}
.btn.primary:not(:disabled):hover {
  transform: translateY(-1px);
}
.btn.ghost {
  color: var(--ink);
  background: var(--surface);
  border-color: var(--line-strong);
}
.btn.ghost:hover {
  border-color: var(--coral);
}

.dlg-enter-active,
.dlg-leave-active {
  transition: opacity 0.2s var(--ease);
}
.dlg-enter-from,
.dlg-leave-to {
  opacity: 0;
}
.dlg-enter-active .dialog,
.dlg-leave-active .dialog {
  transition: transform 0.25s var(--ease);
}
.dlg-enter-from .dialog,
.dlg-leave-to .dialog {
  transform: translateY(12px) scale(0.98);
}
</style>
