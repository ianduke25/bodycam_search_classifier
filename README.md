
# Bodycam Transcript Search Classifier

This repository contains resources for a project aimed at classifying transcripts that contain police searches. The project includes a notebook with steps to train and save a private model.

## Setup Instructions

### Prerequisites

To run the notebook and use the model, you will need the following:
- Python 3.7+
- Jupyter Notebook
- Required Python packages (listed in `embedding_and_training.ipynb`)

### Installation

1. Clone this repository to your local machine. 

2. Navigate to the project directory:
   ```bash
   cd bodycam_search_classifier
   ```

### Running the Notebook

1. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Open `search_classifier_training.ipynb` in Jupyter Notebook.
3. Follow the steps in the notebook to understand the data preprocessing, embedding, and model training processes.

## Required Data

A `search_labels.csv` file containing labeled data used for training the model. It should include two columns:
- `transcript`: The text of the transcript.
- `label`: The label indicating whether the transcript contains a police search (1) or not (0).
