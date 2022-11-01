#!/bin/bash

source bin/activate

cines=( ./data2/*.cine )

for vidname in cines
do
    echo $vidname | python3 meniscus.py
done

