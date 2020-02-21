#!/bin/bash

floyd run --env tensorflow-1.10.0 --cpu --message "Dermatologist" \
    --data justinsolms/datasets/ham10000:ham10000_images \
    --max-runtime 1800 --follow \
    "python dermatologist -s 2 -e 2 --generate --dropout 0.25 --learn_rate 0.0001"

