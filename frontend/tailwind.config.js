/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        atelier: {
          dark: '#08090d',
          charcoal: '#111217',
          walnut: '#1b1716',
          panel: '#151314',
          border: 'rgba(180, 130, 90, 0.18)',
          accent: '#f59e0b', // Amber glow
          sage: '#789d78',
          amethyst: '#8b5cf6',
          cyan: '#06b6d4',
          silver: '#e2e8f0',
        }
      },
      fontFamily: {
        serif: ['Georgia', 'Cambria', 'serif'],
        mono: ['Menlo', 'Monaco', 'Courier New', 'monospace'],
      }
    },
  },
  plugins: [],
}
