#!/bin/bash
set -e

# Assumes KenLM is installed
lmplz -o 5 < data/lm/lm_corpus.txt > data/lm/my_corpus.arpa
build_binary data/lm/my_corpus.arpa data/lm/my_corpus.bin

echo "KenLM model built → data/lm/my_corpus.bin"
