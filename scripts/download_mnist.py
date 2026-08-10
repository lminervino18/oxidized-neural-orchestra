#!/usr/bin/env python3

import subprocess

TARGET_URL = "https://drive.google.com/uc?export=download&id=18kPPrx97omjSkVltx_INglQPvylsbhbI"
DOWNLOAD_DIR = "datasets"

zip_download_path = f"{DOWNLOAD_DIR}/mnist.zip"

subprocess.run(["mkdir", "-p", DOWNLOAD_DIR])
subprocess.run(["curl", "-L", "-o", zip_download_path, TARGET_URL])
subprocess.run(["unzip", zip_download_path, "-d", DOWNLOAD_DIR])
