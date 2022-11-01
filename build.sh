#!/usr/bin/bash

cd cine_median
cargo build --release
cd ..
cp cine_median/target/release/libcine_median.so .

cd sift2
cargo build --release
cd ..
cp sift2/target/release/libsift2.so .