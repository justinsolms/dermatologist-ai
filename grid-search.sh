#!/bin/bash

floyd run --env tensorflow-2.0 --gpu --message "Dermatologist" \
--data justinsolms/datasets/ham10000:ham10000_images \
--max-runtime 3600 --follow \
"python -m dermatologist --generate -e 40 -d 0.25 -l 0.0001"
