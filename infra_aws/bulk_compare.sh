#!/bin/bash

for i in $(seq 10); do
  ./main.sh COMPARE "../components/nachito9_$i.jpg"
done
