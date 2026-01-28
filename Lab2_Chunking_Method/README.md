# Lab 2: Fine-tuning BLIP for Image Captioning

This lab demonstrates the process of fine-tuning a pre-trained Vision-Language model for the task of image captioning.

## Overview

The notebook [Chunking Method.ipynb](Chunking%20Method.ipynb) guides you through fine-tuning the **BLIP (Bootstrapping Language-Image Pre-training)** model from Salesforce on a custom football-related dataset.

## Workflow

1.  **Environment Setup**: Installing necessary libraries like `transformers`, `datasets`, and `torch`.
2.  **Dataset Loading**: Loading the `ybelkada/football-dataset` from the Hugging Face Hub.
3.  **Data Preprocessing**: Creating a custom PyTorch `Dataset` and `DataLoader` to handle image-text pairs and prepare them for the model using `AutoProcessor`.
4.  **Model Loading**: Initializing the `BlipForConditionalGeneration` model and its corresponding processor.
5.  **Training**: Implementing a training loop using the AdamW optimizer to fine-tune the model parameters over multiple epochs.
6.  **Inference**: Testing the fine-tuned model on sample images to generate descriptive captions.
7.  **Hub Integration**: Instructions on how to push the fine-tuned model to the Hugging Face Hub and load it for future use.

## Requirements

- `transformers`
- `datasets`
- `torch`
- `matplotlib`
- `pillow`

## Usage

Run the [Chunking Method.ipynb](Chunking%20Method.ipynb) notebook. Ensure you have access to a GPU (CUDA) for faster training, although it can also run on CPU.
