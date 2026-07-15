import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '../views/HomeView.vue'

const routes = [
  {
    path: '/',
    name: 'Home',
    component: HomeView,
  },
  // legacy /about now scrolls to the About section on the single-page site
  {
    path: '/about',
    redirect: { path: '/', hash: '#about' },
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
  scrollBehavior(to) {
    if (to.hash) {
      return { el: to.hash, behavior: 'smooth', top: 84 }
    }
    return { top: 0 }
  },
})

export default router
