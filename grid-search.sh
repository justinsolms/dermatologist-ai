#!/bin/bash

floyd run --env tensorflow-2.0 --gpu --message "Dermatologist" \
--data justinsolms/datasets/ham10000:ham10000_images \
--max-runtime 1800 --follow \
"python -m dermatologist --generate --dropout 0.20 --learn_rate 0.0001"
