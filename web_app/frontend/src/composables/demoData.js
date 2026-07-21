// Sample-run imagery for the "Watch a sample run" demo.
// Everything here is clearly-synthetic placeholder art — no real patient data
// and no real inference. The UI labels it as a demo so it can't be mistaken
// for a genuine analysis.

function svg(inner, w = 1200, h = 1200) {
  const doc = `<svg xmlns='http://www.w3.org/2000/svg' width='${w}' height='${h}' viewBox='0 0 400 400'>${inner}</svg>`
  return 'data:image/svg+xml;utf8,' + encodeURIComponent(doc)
}

// warm skin background + a soft irregular lesion
const skinBg = `
  <defs>
    <radialGradient id='skin' cx='50%' cy='42%' r='75%'>
      <stop offset='0%' stop-color='#f3cba6'/>
      <stop offset='60%' stop-color='#e6b489'/>
      <stop offset='100%' stop-color='#d59d72'/>
    </radialGradient>
    <radialGradient id='lesion' cx='48%' cy='46%' r='60%'>
      <stop offset='0%' stop-color='#6b4a3a'/>
      <stop offset='55%' stop-color='#4f3226'/>
      <stop offset='100%' stop-color='#3a241b'/>
    </radialGradient>
    <filter id='soft'><feGaussianBlur stdDeviation='3'/></filter>
  </defs>
  <rect width='400' height='400' fill='url(#skin)'/>
  <g opacity='0.25'>
    <circle cx='70' cy='90' r='2.4' fill='#c98f63'/>
    <circle cx='320' cy='120' r='2' fill='#c98f63'/>
    <circle cx='120' cy='330' r='2.2' fill='#c98f63'/>
    <circle cx='300' cy='320' r='1.8' fill='#c98f63'/>
  </g>
`

const lesionShape = `
  <g filter='url(#soft)'>
    <ellipse cx='200' cy='196' rx='96' ry='84' fill='url(#lesion)'/>
    <ellipse cx='168' cy='176' rx='34' ry='30' fill='#5c3a2b' opacity='0.8'/>
    <ellipse cx='236' cy='214' rx='40' ry='34' fill='#2f1d15' opacity='0.85'/>
    <ellipse cx='214' cy='168' rx='22' ry='18' fill='#7a5340' opacity='0.6'/>
  </g>
`

const hairs = `
  <g stroke='#241812' fill='none' stroke-linecap='round' opacity='0.9'>
    <path d='M40 60 C 140 120, 240 150, 360 130' stroke-width='3'/>
    <path d='M60 340 C 160 300, 250 250, 350 260' stroke-width='2.4'/>
    <path d='M30 210 C 120 200, 210 230, 330 300' stroke-width='2.2'/>
    <path d='M210 30 C 190 130, 220 230, 180 360' stroke-width='2'/>
  </g>
`

const original = svg(skinBg + lesionShape + hairs)

const hairRemoved = svg(skinBg + lesionShape)

const processed = svg(
  skinBg +
    lesionShape +
    // subtle sampling ring showing where skin tone is read (lesion-free border)
    `<circle cx='200' cy='200' r='150' fill='none' stroke='#ffffff' stroke-width='2' stroke-dasharray='6 8' opacity='0.55'/>`,
)

const hairMask = svg(`
  <rect width='400' height='400' fill='#0a0a0a'/>
  <g stroke='#ffffff' fill='none' stroke-linecap='round'>
    <path d='M40 60 C 140 120, 240 150, 360 130' stroke-width='3'/>
    <path d='M60 340 C 160 300, 250 250, 350 260' stroke-width='2.4'/>
    <path d='M30 210 C 120 200, 210 230, 330 300' stroke-width='2.2'/>
    <path d='M210 30 C 190 130, 220 230, 180 360' stroke-width='2'/>
  </g>
`)

// jet-style Grad-CAM heatmap concentrated over the lesion
const gradcam = svg(`
  <defs>
    <radialGradient id='cam' cx='50%' cy='50%' r='50%'>
      <stop offset='0%' stop-color='#ff2d2d'/>
      <stop offset='28%' stop-color='#ff9a1f'/>
      <stop offset='52%' stop-color='#f2e836'/>
      <stop offset='72%' stop-color='#37d67a'/>
      <stop offset='100%' stop-color='#1b3fd6'/>
    </radialGradient>
  </defs>
  <rect width='400' height='400' fill='#101a3a'/>
  <ellipse cx='202' cy='198' rx='150' ry='140' fill='url(#cam)' opacity='0.95'/>
`)

// Sample metadata for the walkthrough. `metadata` is what the user "picked";
// `metadata_used` mirrors what the backend would echo back (resolved values).
const SAMPLE_META = { age: 34, sex: 'female', anatom_site: 'upper_extremity' }

export function buildDemoRun(metadata) {
  // Reflect the user's picks if they set any; otherwise fall back to the sample.
  const used = {
    age: metadata?.age ?? SAMPLE_META.age,
    sex: metadata?.sex && metadata.sex !== 'unknown' ? metadata.sex : SAMPLE_META.sex,
    anatom_site:
      metadata?.anatom_site && metadata.anatom_site !== 'unknown'
        ? metadata.anatom_site
        : SAMPLE_META.anatom_site,
  }
  return [
    { step: 'load_image', width: 1200, height: 1200, image_base64: original },
    { step: 'remove_hair', hair_mask: hairMask, inpainted_image: hairRemoved },
    { step: 'preprocess', skin_group: 'III (Intermediate)', processed_image: processed },
    { step: 'model_prediction', probability: 0.12, predicted_class: 0, metadata_used: used },
    { step: 'gradcam', gradcam: gradcam },
    { step: 'done' },
  ]
}
