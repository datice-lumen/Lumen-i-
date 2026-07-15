// Metadata option lists for the predictor.
// Values MUST match the model's SEX_CATEGORIES / SITE_CATEGORIES exactly
// (from the najbolji_10_6.pt checkpoint config) — the backend one-hot encodes
// against these strings, so a mismatch silently degrades to "unknown".

export const SEX_OPTIONS = [
  { value: 'unknown', label: 'Prefer not to say' },
  { value: 'female', label: 'Female' },
  { value: 'male', label: 'Male' },
]

export const SITE_OPTIONS = [
  { value: 'unknown', label: 'Not sure' },
  { value: 'torso', label: 'Torso' },
  { value: 'upper_extremity', label: 'Arm / upper limb' },
  { value: 'lower_extremity', label: 'Leg / lower limb' },
  { value: 'head_neck', label: 'Head or neck' },
  { value: 'palms_soles', label: 'Palms or soles' },
]

const SEX_LABEL = Object.fromEntries(SEX_OPTIONS.map((o) => [o.value, o.label]))

// Concise forms used in one-line summaries.
const SITE_SHORT = {
  torso: 'torso',
  upper_extremity: 'upper limb',
  lower_extremity: 'lower limb',
  head_neck: 'head/neck',
  palms_soles: 'palms/soles',
}

export function sexLabel(v) {
  return SEX_LABEL[v] || SEX_LABEL.unknown
}

// One-line summary of the metadata a run used, e.g. "age 34 · female · upper limb".
// Returns '' when nothing meaningful was provided (all unknown/missing).
export function metadataSummary(meta) {
  if (!meta) return ''
  const parts = []
  if (meta.age !== null && meta.age !== undefined && meta.age !== '') parts.push(`age ${meta.age}`)
  if (meta.sex && meta.sex !== 'unknown') parts.push(SEX_LABEL[meta.sex].toLowerCase())
  if (meta.anatom_site && meta.anatom_site !== 'unknown') {
    parts.push(SITE_SHORT[meta.anatom_site] || '')
  }
  return parts.filter(Boolean).join(' · ')
}
