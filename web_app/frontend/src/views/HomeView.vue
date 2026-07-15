<template>
  <div class="home">
    <HeroSection />
    <LiveAnalyzer />
    <MyMoles />
    <hr class="spectrum-rule shell divider" />
    <HowItWorks />
    <FairnessSection />
    <MetricsSection />
    <AboutSection />
  </div>
</template>

<script setup>
import HeroSection from '../components/sections/HeroSection.vue'
import LiveAnalyzer from '../components/LiveAnalyzer.vue'
import MyMoles from '../components/sections/MyMoles.vue'
import HowItWorks from '../components/sections/HowItWorks.vue'
import FairnessSection from '../components/sections/FairnessSection.vue'
import MetricsSection from '../components/sections/MetricsSection.vue'
import AboutSection from '../components/sections/AboutSection.vue'
</script>

<style scoped>
.divider {
  margin: 0 auto;
  opacity: 0.5;
}
</style>
