#!/bin/bash

floyd run --env theano --gpu --message "Dermatologist" \
    --data justinsolms/datasets/ham10000:ham10000_images \
    --max-runtime 1800 --follow \
    "python -m dermatologist --generate --dropout 0.25 --learn_rate 0.0001"

