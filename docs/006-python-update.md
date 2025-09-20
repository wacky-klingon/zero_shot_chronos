Here are the best ways to upgrade Python to 3.9+ on Ubuntu WSL for AutoGluon compatibility:

## Option 1: Using deadsnakes PPA (Recommended)
```bash
# Add the deadsnakes PPA
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update

# Install Python 3.9, 3.10, or 3.11
sudo apt install python3.9 python3.9-venv python3.9-dev
# or
sudo apt install python3.10 python3.10-venv python3.10-dev
# or
sudo apt install python3.11 python3.11-venv python3.11-dev

# Update alternatives to set as default
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1
# or for 3.10/3.11, use the appropriate version
```

## Option 2: Using pyenv (Most Flexible)
```bash
# Install dependencies
sudo apt update
sudo apt install -y make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev \
libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python3-openssl

# Install pyenv
curl https://pyenv.run | bash

# Add to your ~/.bashrc
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc

# Reload shell
source ~/.bashrc

# Install Python 3.9.18 (latest 3.9)
pyenv install 3.9.18
pyenv global 3.9.18
```

## Option 3: Using conda/mamba (If you prefer conda)
```bash
# Install miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Create environment with Python 3.9
conda create -n chronos-raw python=3.9
conda activate chronos-raw
```

## Option 4: Using Poetry's Python management
```bash
# If you have Python 3.9+ installed elsewhere
poetry env use python3.9
# or
poetry env use python3.10
```

## After upgrading Python:
1. Verify the version: `python3 --version`
2. Install Poetry if not already installed: `curl -sSL https://install.python-poetry.org | python3 -`
3. Run `poetry install` in your project directory
