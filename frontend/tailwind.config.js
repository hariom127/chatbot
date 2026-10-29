/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Poppins", "system-ui", "sans-serif"],
      },
      colors: {
        // Palette taken from the ReelTrak web app (steel blue primary, pale blue-grey surfaces).
        brand: {
          50: "#eef3f7",
          100: "#dce8f0",
          500: "#5089ac",
          600: "#43779a",
          700: "#386482",
        },
        canvas: "#eef3f7",
        ink: "#111927",
        muted: "#6c737f",
      },
      boxShadow: {
        card: "0 2px 12px rgba(17, 24, 39, 0.06)",
      },
    },
  },
  plugins: [],
};
