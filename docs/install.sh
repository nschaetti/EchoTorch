#!/bin/bash

# Install Sphinx, rst2pdf, latex
sudo apt-get install python3-sphinx
sudo apt-get install rst2pdf
sudo apt-get install texlive-latex-base
sudo apt install texlive-formats-extra
sudo apt-get install texlive-lang-cyrillic

# Install torch, scipy, matplotlib, etc
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip3 install scikit-learn
pip3 install scipy
pip3 install matplotlib
pip3 install networkx

# Build documentation
sphinx-build -E -b html -d build/doctrees . build/html
