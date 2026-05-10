#!/bin/bash
set -euo pipefail

echo "Installing bundler 2.7.2..."
gem install bundler -v 2.7.2 --no-document

cd /work

echo "Running bundle install..."
bundle _2.7.2_ install --jobs 1 --verbose

echo "Running script/update-ids..."
bundle _2.7.2_ exec ruby script/update-ids

echo "Running rake test..."
bundle _2.7.2_ exec rake test

echo "Done."
