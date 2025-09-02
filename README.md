
# Bodycam Transcript Search Classifier

This repository contains resources for a project aimed at classifying transcripts that contain police searches. The project includes a notebook with steps to train and save a private model.

## Required Data

A `search_labels.csv` file containing labeled data used for training the model. It should include two columns:
- `transcript`: The text of the transcript.
- `label`: The label indicating whether the transcript contains a police search (1) or not (0).
