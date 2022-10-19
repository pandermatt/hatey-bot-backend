# 🤬🤖 Hatey Bot Backend

## Important Links

- [Website hatey.monster](https://hatey.monster)
- [Swagger Documentation api.hatey.monster](https://api.hatey.monster)
- [Frontend GitHub](https://github.com/pandermatt/hatey-bot-frontend) [![React app deployement](https://github.com/pandermatt/hatey-bot-frontend/actions/workflows/deploy.yml/badge.svg)](https://github.com/pandermatt/hatey-bot-frontend/actions/workflows/deploy.yml)
- [Backend GitHub](https://github.com/pandermatt/hatey-bot-backend)
- [LaTeX Documentation](https://www.overleaf.com/project/633592679211c2009c8bce96)


## Development

### Setup

```bash
git clone git@github.com:pandermatt/hatey-bot-backend.git
cd hatey-bot-backend

# Install dependencies
pip install -r requirements.txt

# Create application config
cp application.example.yml application.yml
```

#### Application config

The application config is a YAML file that contains all the configuration for the application. 
It is located at `application.yml` and is ignored by git. 
The file `application.example.yml` contains all the possible configuration options with their default values.
See [application.example.yml](application.example.yml) for more information.


### Run

- Run the Flask application with `python app.py`
- Run the production server with `python waitress_server.py`

If `CAN_GENERATE_FILES` in the application config is set to `True`, the application will generate all the AI models and files on startup.
Thus, the first startup will take a while.
All needed files will be generated in the [data](data) folder.
The app downloads the data from several sources, like GitHub or HuggingFace.
Pre-generated files are available in the [data/input](data/input) folder.


### Tests

- Run the tests with `python -m pytest py_tests/`


### Research

The [research](research) folder contains all the research and experiments that were done for this project.
The filename describes the topic of the research and they can be run with `python filename.py` or with jupyter notebooks.

## Slackbot Integration

Create a new Slack App via [api.slack.com/apps](https://api.slack.com/apps) and click on <Your Slack App Name>

- Find the ENV var `SLACK_BOT_TOKEN` under OAuth & Permissions $\rightarrow$ Bot User OAuth Access Token
- Find the ENV var `SLACK_SIGNING_SECRET` under Basic Information $\rightarrow$ Signing Secret


## Server deployment

Instructions for deploying the app on a server.
We use [asdf](https://asdf-vm.com/#/) to manage the different versions of python and [caddy](https://caddyserver.com/) as a reverse proxy.

### Setup

```bash
# install asdf to easily install python
git clone https://github.com/asdf-vm/asdf.git ~/.asdf --branch v0.10.2
echo ". $HOME/.asdf/asdf.sh" >> ~/.bashrc
echo ". $HOME/.asdf/completions/asdf.bash" >> ~/.bashrc
source ~/.bashrc

# install python
sudo apt-get update; sudo apt-get install --no-install-recommends make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
asdf plugin-add python
asdf install python 3.7.14
asdf global python 3.7.14

# clone repo
git clone git@github.com:pandermatt/hatey-bot-backend.git
cd hatey-bot-backend
pip install -r requirements.txt
cp application.example.yml application.yml

# install caddy to serve the app
sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt update
sudo apt install caddy

# start caddy
caddy start
```
