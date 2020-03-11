#!/bin/bash

#  Keep this as we have this a a good baseline on sunquant
# floyd run --env tensorflow-2.1 --gpu --message "Dermatologist" \
# --data justinsolms/datasets/ham10000:ham10000_images \
# --max-runtime 3600 --follow \
# "python -m dermatologist --generate -e 40 -d 0.25 -l 0.0001"

python -m dermatologist -n 512 -d 0.80 -l 0.000002
python -m dermatologist -n 512 -d 0.20 -l 0.000002
python -m dermatologist -n 512 -d 0.80 -l 0.000004
python -m dermatologist -n 512 -d 0.20 -l 0.000004
