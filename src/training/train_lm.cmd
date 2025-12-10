@echo off
setlocal

rem Assumes KenLM is installed and available on PATH

if not exist "data\lm" (
    echo Missing data\lm directory.
    exit /b 1
)

if not exist "data\lm\lm_corpus.txt" (
    echo Missing data\lm\lm_corpus.txt input corpus.
    exit /b 1
)

lmplz -o 5 < "data\lm\lm_corpus.txt" > "data\lm\my_corpus.arpa"
if errorlevel 1 (
    echo lmplz failed to generate ARPA file.
    exit /b %errorlevel%
)

build_binary "data\lm\my_corpus.arpa" "data\lm\my_corpus.bin"
if errorlevel 1 (
    echo build_binary failed to create binary LM.
    exit /b %errorlevel%
)

echo KenLM model built -> data\lm\my_corpus.bin
