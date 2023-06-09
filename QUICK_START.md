# Quick Start

## Install required packages on Mac
```
# install homebrew, skip this step if already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# upgrade pip
python3 -m pip install --upgrade pip setuptools

# install swig which is required for box-2d
brew install swig

# install ffmpeg for recording agent self-play
brew install ffmpeg

# install snappy for compress numpy.array on M1 mac
brew install snappy
CPPFLAGS="-I/opt/homebrew/include -L/opt/homebrew/lib" pip3 install python-snappy


# download the project
git clone https://github.com/michaelnny/deep_rl_zoo.git

cd deep_rl_zoo

pip3 install -r requirements.txt

# optional, install pre-commit and hooks
pip3 install pre-commit

pre-commit install
```

## Install required packages on Ubuntu Linux
```
# install swig which is required for box-2d
sudo apt install swig

# install ffmpeg for recording agent self-play
sudo apt-get install ffmpeg

# upgrade pip
python3 -m pip install --upgrade pip setuptools


# download the project
git clone https://github.com/michaelnny/deep_rl_zoo.git

cd deep_rl_zoo

pip3 install -r requirements.txt

# optional, install pre-commit and hooks
pip3 install pre-commit

pre-commit install
```

## Install required packages on openSUSE 15 Tumbleweed Linux
```
# install required dev packages
sudo zypper install gcc gcc-c++ python3-devel

# install swig which is required for box-2d
sudo zypper install swig

# install ffmpeg for recording agent self-play
sudo zypper install ffmpeg

# upgrade pip
python3 -m pip install --upgrade pip setuptools


# download the project
git clone https://github.com/michaelnny/deep_rl_zoo.git

cd deep_rl_zoo

pip3 install -r requirements.txt

# optional, install pre-commit and hooks
pip3 install pre-commit

pre-commit install
```
