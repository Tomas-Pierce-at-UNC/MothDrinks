#!/usr/bin/bash

cd cine_median
cargo build --release
cd ..
cp cine_median/target/release/libcine_median.so .

cd sift2
cargo build
cd ..
cp sift2/target/debug/libsift2.so .