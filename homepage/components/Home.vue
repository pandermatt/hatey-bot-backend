<template>
    <div class="main-container vp-doc">
        <div class="info custom-block">
            <p class="custom-block-title">INFO</p>
            <p>
                This project was developed as part of the <a
                    href="https://studentservices.uzh.ch/uzh/anonym/vvz/?sap-language=EN&sap-ui-language=EN#/details/2022/003/SM/51079434">Essentials
                    in Text and Speech Processing</a> course at the University of Zurich (UZH).
                We received a grade of <b>6.0</b> 🥇, the highest grade in the Swiss grading system.
            </p>
        </div>


        <p class="made-by">Made with ❤️ by</p>
        <VPTeamMembers size="small" :members="members" />

        <img src="/webpage.png" alt="Webpage" />
    </div>
</template>

<script setup>
import { VPTeamMembers } from 'vitepress/theme'

const members = [
    {
        avatar: 'https://avatars.githubusercontent.com/u/20790833?v=4',
        name: 'Pascal Andermatt',
        links: [
            { icon: 'github', link: 'https://github.com/pandermatt' },
            { icon: 'linkedin', link: 'https://www.linkedin.com/in/pandermatt/' },
        ]
    },
    {
        avatar: 'https://avatars.githubusercontent.com/u/87606238?v=4',
        name: 'Tobias Fankhauser',
        links: [
            { icon: 'github', link: 'https://github.com/TobiFank' },
            { icon: 'linkedin', link: 'https://www.linkedin.com/in/tobias-fankhauser-b536a0b7/' },
        ]
    },
    {
        avatar: 'https://avatars.githubusercontent.com/u/15248306?v=4',
        name: 'Denis Müller',
        links: [
            { icon: 'github', link: 'https://github.com/denis-mueller' },
            { icon: 'linkedin', link: 'https://www.linkedin.com/in/denis-mueller/' },
        ]
    },
]
</script>

<style scoped lang="scss">
.main-container {
    margin: 15px auto 0;
    width: 100%;
    max-width: 1280px;
    padding: 0 24px;
    position: relative;

    .info {
        margin-bottom: 30px;
    }

    .made-by {
        text-align: center;
        font-size: 18px;
        font-weight: 500;
    }
}

@media (min-width: 640px) {
    .main-container {
        padding: 0 48px;
    }
}

@media (min-width: 960px) {
    .main-container {
        width: 100%;
        padding: 0 64px;
    }
}

img {
    max-width: 800px;
    width: 100%;
    border-radius: 8px;
    margin: 50px auto 0;
}
</style>
