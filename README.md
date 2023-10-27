# MothDrinks
This repository contains code and data for the paper submission titled "Feeding rate in *Manduca sexta* is unaffected by proboscis
submersion depth"

## Goals
* measure rate of nectar ingestion from video data
* measure proboscis submergence depth from video data
* evaluate whether nectar ingestion rate and proboscis submergence are related

## Data and Artifacts
List of 

### Videos
For space reasons the raw video data is not included in the repository. It would be stored in a data2/ folder and the following list of filenames would be included. Division into subfolders is related to earlier revisions of the software and not reflective of present design constraints. It is also the case that the data/ subfolder contains duplicates. Video data is in CINE format.
* moth22_2022-01-26.cine
* moth22_2022-01-31_Cine1.cine
* moth22_2022-02-01b_Cine1.cine
* moth22_2022-02-01_Cine1.cine
* moth22_2022-02-02a_maybe_Standing.cine
* moth22_2022-02-08_Cine1.cine
* moth22_2022-02-09_okay.cine
* moth22_manual_2022-01-28_Cine1_Cine1.cine
* moth23_2022-02-09_meh.cine
* moth23_2022-02-14_Cine1.cine
* moth25_2022-02-15_nodrink.cine
* moth26_2022-02-15_freeflight.cine
* moth26_2022-02-17_hover-feeding_Cine1.cine
* moth26_2022-02-21_Cine1.cine
* moth26_2022-02-22_Cine1.cine
* moth26_2022-02-25_Cine1.cine
* moth26_2022-02-25_Cine2.cine
* mothM1_2022-09-21_Cine1.cine
* mothM1_2022-09-22_Cine1.cine
* mothM1_2022-09-23_Cine1.cine
* mothM1_2022-09-26_Cine1.cine
* mothM1_2022-09-27_Cine1.cine
* mothM3_2022-09-19_Cine1.cine
* mothM3_2022-09-21_Cine1.cine
* mothM3_2022-09-22_Cine1.cine
* mothM3_2022-09-22_Cine2.cine
* mothM3_2022-09-23_Cine1.cine
* mothM4_2022-09-19_Cine1.cine
* mothM4_2022-09-22_Cine1.cine
* mothM4_2022-09-23_Cine1.cine
* mothM5_2022-09-21_Cine1.cine
* mothM5_2022-09-23_Cine1.cine
* mothM5_2022-09-26_Cine1.cine
* mothM5_2022-09-26_Cine2.cine
* mothM5_2022-09-27_Cine1.cine
* mothM6_2022-09-22_Cine1.cine
* mothM6_2022-09-22_Cine2.cine
* mothM6_2022-09-23_Cine1.cine
* mothM6_2022-09-26_Cine1.cine
* mothM6_2022-09-27_Cine1.cine
* mothM8_2022-09-20_Cine2.cine
* mothM8_2022-09-23_Cine1.cine
* mothM9_2022-09-20_Cine1.cine
* unsuitableVideos/moth22_2022-02-04_Cine1.cine
* unsuitableVideos/moth22_2022_02_04_Cine1.cine
* unsuitableVideos/moth22_2022-02-07_Cine1.cine
* unsuitableVideos/moth22_2022_02_09_bad_Cine1.cine
* unsuitableVideos/moth22_2022-02-11.cine
* unsuitableVideos/moth22_manual2_2022-01-28_Cine1.cine
* unsuitableVideos/moth23_2022-02-11_a.cine
* unsuitableVideos/moth23_2022-02-15_Cine1.cine
* unsuitableVideos/moth26_2022-02-23_Cine1.cine
* unsuitableVideos/mothM8_2022-09-20_Cine1.cine
* data/deadMothTest_2022-01-28_Cine1.cine
* data/moth22_2022-01-26.cine
* data/moth22_2022-01-31_Cine1.cine
* data/moth22_2022-02-01b_Cine1.cine
* data/moth22_2022-02-01_Cine1.cine
* data/moth22_2022_02_02a_maybe_Standing.cine
* data/moth22_2022-02-04_Cine1.cine
* data/moth22_2022_02_04_Cine1.cine
* data/moth22_2022-02-07_Cine1.cine
* data/moth22_2022-02-08_Cine1.cine
* data/moth22_2022_02_09_bad_Cine1.cine
* data/moth22_2022-02-09_okay.cine
* data/moth22_2022-02-11.cine
* data/moth22_manual_2022-01-28_Cine1_Cine1.cine
* data/moth22_manual2_2022-01-28_Cine1.cine
* data/moth23_2022-02-09_meh.cine
* data/moth23_2022-02-11_a.cine
* data/moth23_2022-02-14_Cine1.cine
* data/moth23_2022-02-15_Cine1.cine
* data/moth25_2022-02-15_nodrink.cine
* data/moth26_2022-02-15_freeflight.cine
* data/moth26_2022-02-16_wideflower_hover.cine
* data/moth26_2022-02-17_hover-feeding_Cine1.cine
* data/moth26_2022-02-21_Cine1.cine
* data/moth26_2022-02-22_Cine1.cine
* data/moth26_2022-02-23_Cine1.cine
* data/moth26_2022-02-25_Cine1.cine
* data/moth26_2022-02-25_Cine2.cine

### Extracted Frame Measurements
Measurements of traits extracted by neural networks are present in the meniscus_measurements.json and (WIP) files.
The schema followed in meniscus_measurements.json is (WIP). The schema followed in (WIP) is (WIP).

### Neural Networks
The neural network responsible for locating nectar meniscuses is in a compressed archive in the file (WIP).
The neural network responsible for locating the proboscis tips is in a compressed archive in the file (WIP).

## Important Scripts
* aggregate_figures.py - used to generate figures from the data in meniscus_measurements.json and (WIP)
* (WIP) - used to measure the meniscus position from video data
* (WIP) - used to measure the proboscis tip position from video data

## How to use
Install. Acquire data in CINE format and put it in a data2/ directory inside the cloned repository. Run (WIP) to get measurements of meniscus position in (WIP). Run (WIP) to get measurements of proboscis tip position in (WIP). Run the aggregate_figures.py script to show visualizations of data including drinking rates and proboscis submergence. 

### Installation
Download and install all dependencies. Clone the repository and extract the neural network files. 


## Dependencies
You will want to have the stable channel Rust compiler, the Python 3.10 interpreter, and pip intalled. Having gcc installed may be useful as well.
There should be a requirements.txt file specifying which packages to install, however the direct dependencies are:
pandas, numpy, matplotlib, tensorflow, loess, scikit-image, scikit-learn. CUDA-compatible hardware is recommended.

## Notes
Formatting video file names in the format of moth<id>_<date>_Cine<number>.cine is likely to increase the chance of the code working correctly.
