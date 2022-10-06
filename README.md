# Hatey Bot Backend

## Development

### Setup

```bash
git clone git@github.com:pandermatt/hatey-bot-backend.git
cd hatey-bot-backend
pip install -r requirements.txt
cp application.example.yml application.yml
```



## Server deployment

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