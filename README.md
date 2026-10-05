# Basic Neural Network 🌬️⚡

This repository contains a fully modular, config-driven Deep Learning pipeline built with TensorFlow and Keras to predict normalized wind power at various wind farms based on weather forecasts.

## Project Structure

The project follows a standard Data Science/ML engineering structure:

```text
.
├── config.py          # Master configuration (Paths, Hyperparameters, Model Architecture)
├── data.py            # Data loading, missing value imputation, and scaling pipeline
├── model.py           # Dynamic Keras Sequential model builder
├── train.py           # Training script with validation and checkpointing
├── predict.py         # Inference script for generating Kaggle-ready submissions
├── requirements.txt   # Project dependencies
└── README.md          # Project documentation