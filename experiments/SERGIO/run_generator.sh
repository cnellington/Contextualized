#!/bin/bash

K=(1)
e=5
n=(1000)
nh=(0.05)

for ((i = 0; i < 1; i++)); do
        echo "K-${K[i]} e-${e[j]}"
        python generator.py \
        -K ${K[i]} \
        -e ${e} \
        -n ${n[i]} \
        -nh ${nh}
done
