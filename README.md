# Rock-Paper-Scissors Image Classification with CNNs

This project uses convolutional neural networks to classify hand images as rock, paper, or scissors. It includes CNN parameter calculations, visual examples of hand-crafted convolution filters, and two image classification models trained with PyTorch Lightning.

The goal is to show a complete deep learning workflow: dataset loading, image preprocessing, augmentation, model training, checkpointing, learning curve visualization, and final test evaluation.

## Dataset

The project uses the [Rock-Paper-Scissors dataset from Kaggle](https://www.kaggle.com/datasets/sanikamal/rock-paper-scissors-dataset).

To run the notebook locally, download and unzip the dataset, then place the data in this structure:

```text
data/Rock-Paper-Scissors/train/
data/Rock-Paper-Scissors/test/
```

The notebook uses the training folder for training and validation, reserving 320 images for validation. The test folder is used only for final evaluation.

## Project Files

- `cnn-rock-paper-scissors-classification.ipynb`: Cleaned portfolio notebook.

## Methods

The notebook includes:

- Trainable parameter and memory estimates for a sample convolutional network.
- Hand-crafted convolution filters applied to a grayscale image to visualize feature maps.
- Image loading with `torchvision.datasets.ImageFolder`.
- Data augmentation with resizing, horizontal flipping, random rotation, and color jitter.
- A custom CNN trained from scratch.
- A MobileNetV2 transfer-learning model with a new classification head.
- PyTorch Lightning training loops, checkpointing, early stopping, and CSV metric logging.
- Learning curve plots for loss and accuracy.

## Model Results

| Model | Test Accuracy | Test Loss |
| --- | ---: | ---: |
| Custom CNN from scratch | 88.98% | 0.397 |
| MobileNetV2 transfer learning | 81.99% | 0.581 |

In this run, the smaller custom CNN outperformed MobileNetV2 on the held-out test set. This is a useful reminder that pretrained models are not automatically better for every small, focused image classification task. Model performance still depends on the dataset, preprocessing choices, model capacity, and tuning.

## Tools Used

- Python
- PyTorch
- PyTorch Lightning
- Torchvision
- Torchmetrics
- Torchinfo
- Matplotlib
- Pandas
- Jupyter Notebook

## Key Takeaways

- Convolution filters can be interpreted visually through their feature maps.
- Image augmentation helps improve generalization by exposing the model to realistic variation.
- A compact CNN can perform well on a focused three-class image classification problem.
- Transfer learning should be evaluated empirically rather than assumed to outperform simpler models.
- Clear train/validation/test separation is important for reliable model evaluation.

## GitHub Publishing Notes

The dataset, model checkpoints, training logs, and Jupyter checkpoint files are intentionally excluded from version control. This keeps the repository lightweight and reproducible while avoiding large generated files.
# CNN-image-classification
