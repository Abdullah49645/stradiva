/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./app/**/*.{js,ts,jsx,tsx,mdx}", "./components/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        asphalt: "#0B0D0E",     // page background - dark road at night, not pure black
        graphite: "#15181A",    // panel surface
        graphite2: "#1D2124",   // raised/hover surface
        hairline: "#2A2F33",    // dividers, panel edges
        paper: "#EDEAE2",       // primary text - warm off-white, headlamp-lit paper
        muted: "#8B9296",       // secondary text
        signal: "#E8A33D",      // amber hazard/caution - primary accent (avoidable verdict)
        hazard: "#C1443C",      // brick-red hazard - unavoidable verdict
        insufficient: "#6B7280", // slate - insufficient evidence
        hud: "#7FD1C6",         // HUD cyan-teal - measurement/telemetry accent, reticles, grid
      },
      fontFamily: {
        display: ["var(--font-display)"],
        body: ["var(--font-body)"],
        mono: ["var(--font-mono)"],
      },
    },
  },
  plugins: [],
};
