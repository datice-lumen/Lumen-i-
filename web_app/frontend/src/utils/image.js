// Downscale an image (data URL or object URL) to a small JPEG thumbnail so it
// fits comfortably in localStorage. Full-res base64 photos are 1-3 MB each and
// would blow the ~5 MB quota after a couple of saves; a 512px JPEG is ~40-80 KB.

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => resolve(img)
    img.onerror = () => reject(new Error('Could not load image for thumbnailing'))
    img.src = src
  })
}

export async function downscaleToThumb(src, max = 512, quality = 0.72) {
  const img = await loadImage(src)
  const w0 = img.naturalWidth || img.width || max
  const h0 = img.naturalHeight || img.height || max
  const scale = Math.min(1, max / Math.max(w0, h0))
  const w = Math.max(1, Math.round(w0 * scale))
  const h = Math.max(1, Math.round(h0 * scale))

  const canvas = document.createElement('canvas')
  canvas.width = w
  canvas.height = h
  const ctx = canvas.getContext('2d')
  // JPEG has no alpha, paint a white base so transparent PNGs don't go black
  ctx.fillStyle = '#ffffff'
  ctx.fillRect(0, 0, w, h)
  ctx.drawImage(img, 0, 0, w, h)

  return canvas.toDataURL('image/jpeg', quality)
}
