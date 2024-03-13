import { defineConfig } from 'vitepress'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "Hatey Bot",
  description: "Hate Speech Detection",
  base: "/hatey-bot/",
  themeConfig: {
    socialLinks: [
      { icon: 'github', link: 'https://github.com/pandermatt/hatey-bot-backend' },
    ],
    footer: {
      copyright: "<a href='https://pandermatt.ch' target='_blank'>Pascal Andermatt</a> | <a href='https://pandermatt.ch/privacy-policy' target='_blank'>Privacy Policy</a>",
      message: "Made with ❤️ in Switzerland"
    }
  }
})
