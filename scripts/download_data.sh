#!/bin/bash

echo "Downloading data for tiny nerf to the ./data directory"

wget https://people.eecs.berkeley.edu/~bmild/nerf/tiny_nerf_data.npz -P ./data
