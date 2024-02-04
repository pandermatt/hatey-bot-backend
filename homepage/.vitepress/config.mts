import { defineConfig } from 'vitepress'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "Hatey Bot",
  description: "Hate Speech Detection",
  themeConfig: {
    socialLinks: [
      { icon: 'github', link: 'https://github.com/pandermatt/hatey-bot-backend' },
    ],

    footer: {
      message: "Made with ❤️ in Switzerland",
      copyright: "Pascal, Tobias and Denis"
    }
  }
})
