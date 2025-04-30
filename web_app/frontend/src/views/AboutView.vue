<!-- src/views/About.vue -->
<template>
  <div class="about-page">
    <n-card title="About This Project">
      <div v-html="renderedMarkdown" class="markdown-content" />
    </n-card>
  </div>
</template>

<script setup>
import MarkdownIt from 'markdown-it'
import { ref } from 'vue'
import { NCard } from 'naive-ui'

const md = new MarkdownIt()

const markdownText = `
# Melanoma Detector

This is a web-based tool to detect **melanoma** in skin images using deep learning.

## Features

- Upload an image of a skin lesion
- See preprocessing steps like hair removal
- View Grad-CAM visualizations
- Get model predictions

## Disclaimer

This tool is for educational purposes only and **not** a substitute for medical advice.
`

const renderedMarkdown = ref(md.render(markdownText))
</script>

<style scoped>
.markdown-content h1 {
  font-size: 1.5em;
  margin-bottom: 0.5em;
}
.markdown-content h2 {
  font-size: 1.2em;
  margin-top: 1em;
}
.markdown-content p {
  margin: 0.5em 0;
}
</style>
