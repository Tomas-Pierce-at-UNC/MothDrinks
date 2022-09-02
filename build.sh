#!/usr/bin/bash

cd cine_median
cargo build --release
cd ..
cp cine_median/target/release/libcine_median.so .
