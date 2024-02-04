<template>
    <div class="main-container vp-doc">
        <div class="info custom-block">
            <p class="custom-block-title">INFO</p>
            <p>This project was created for the course <a href="https://studentservices.uzh.ch/uzh/anonym/vvz/?sap-language=EN&sap-ui-language=EN#/details/2022/003/SM/51079434">Essentials in Text and Speech Processing</a> at the University of Zurich (UZH).</p>
        </div>

        <img src="/webpage.png" alt="Webpage" />
    </div>
</template>

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
    margin: 0 auto 30px;
}
</style>
