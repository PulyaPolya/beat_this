# Beat Tracking

This repository contains the code developed for the beat tracking component of the Master's thesis:

**"Beat Tracking for Genre-Related Datasets Obtained by Clustering"**

## Overview

This project focuses on training and evaluating beat tracking models on datasets constructed via clustering-based approaches.

The implementation builds upon the work of  
Francesco Foscarin, Jan Schlüter, and Gerhard Widmer:

> *"[Beat This! Accurate Beat Tracking Without DBN Postprocessing](https://arxiv.org/abs/2407.21658)"*

Their model serves as the baseline and foundation for all experiments conducted in this project.

## Approach

- The original *Beat This!* model is used as a **baseline system**
- The model is fine-tuned on **cluster-specific datasets** obtained from the genre-classification pipeline
- Performance is evaluated in terms of beat and downbeat tracking metrics

This setup enables the investigation of whether **training separate models on clustered data** improves performance compared to a general model trained on the full dataset.
