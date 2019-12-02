#!/usr/bin/env bash

set -e

DATA_DIR="$PWD"/data

# Create data directory. Exit script if failed.
mkdir -p "$DATA_DIR"
cd "$DATA_DIR" || exit

# Download data
if [ -d wikitext-2 ]; then
   echo "Found WikiText-2, skipping download."
else
  echo "Downloading WikiText-2..."
  wget --continue https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip
  unzip -q wikitext-2-v1.zip
fi

if [ -d wikitext-103 ]; then
  echo "Found WikiText-103, skipping download."
else
  echo "Downloading WikiText-103..."
  wget --continue https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip
  unzip -q wikitext-103-v1.zip
fi
