
# Bodycam Transcript Search Classifier

This repository contains resources for a project aimed at classifying transcripts that contain police searches. The project includes a notebook with steps to train and save a private model (Test Set Accuracy: 88% | Precision: 45% | Recall: 83%).

## Presentation
Information about model training and performance was presented to ACLU attorneys in May, 2024. The presentation can be found here: [https://docs.google.com/presentation/d/1loyEmG-KriFtHOBHD2x-RyCTEJal65KX2RSz9zjXKR0/edit?usp=sharing](https://docs.google.com/presentation/d/1R0gg6ssnXQd461zMOuv7qex55wx8SIudQlDcGeh8zMw/edit?usp=sharing)

## Repository Contents

1. **embedding_and_training.ipynb**: This Jupyter notebook contains the code for embedding the transcripts and training the classification model.

## Setup Instructions

### Prerequisites

To run the notebook and use the model, you will need the following:
- Python 3.7+
- Jupyter Notebook
- Required Python packages (listed in `embedding_and_training.ipynb`)

### Installation

1. Clone this repository to your local machine:
   ```bash
   git clone [https://github.com/ia/ACLU-police-search-classifier.git](https://github.com/ianduke25/search_classifier)
   ```
2. Navigate to the project directory:
   ```bash
   cd search_classifier
   ```

### Running the Notebook

1. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Open `embedding_and_training.ipynb` in Jupyter Notebook.
3. Follow the steps in the notebook to understand the data preprocessing, embedding, and model training processes.

## Required Data

A `search_labels.csv` file containing labeled data used for training the model. It should include two columns:
- `transcript`: The text of the transcript.
- `label`: The label indicating whether the transcript contains a police search (1) or not (0).
