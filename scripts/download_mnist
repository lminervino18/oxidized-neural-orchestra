#!/usr/bin/env bash

TARGET_URL="https://drive.google.com/uc?export=download&id=18kPPrx97omjSkVltx_INglQPvylsbhbI"
DOWNLOAD_PATH=datasets/mnist.zip
UNZIP_PATH=datasets/

curl -L -o $DOWNLOAD_PATH $TARGET_URL

unzip $DOWNLOAD_PATH -d $UNZIP_PATH
rm $DOWNLOAD_PATH
