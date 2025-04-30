// src/api/api.js

export async function uploadImage(file) {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch('/image/process', {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error('Upload failed');
  }

  // Since the backend streams SSE, you need to handle this differently
  const reader = response.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let result = '';
  let done = false;

  while (!done) {
    const { value, done: doneReading } = await reader.read();
    done = doneReading;
    result += decoder.decode(value);
  }

  return result;
}
