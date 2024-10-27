**Description:**  
This project aims to develop a deep learning model capable of accurately predicting a person's age from a facial image. Utilizing a Convolutional Neural Network (CNN) architecture, specifically based on ResNet50, the model is designed to leverage powerful image recognition capabilities to solve this challenging regression task. The project employs TensorFlow and Keras for model implementation and fine-tuning, focusing on enhancing prediction accuracy through data augmentation and transfer learning techniques.

---

# Age Prediction with ResNet50

## Overview
This project focuses on developing a machine learning model that can predict the age of a person based on an image of their face. The project employs a Convolutional Neural Network (CNN) using the ResNet50 architecture, which is well-regarded for its effectiveness in image recognition tasks. By leveraging transfer learning, the model aims to improve its performance in predicting the continuous variable of age.

The dataset used consists of labeled images of faces, each tagged with the corresponding age. The primary approach is to fine-tune a pre-trained ResNet50 model to adapt it for age prediction using these labeled images.

## Table of Contents
- [Objectives](#objectives)
- [Technologies](#technologies)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Objectives
The main goal of this project is to create a model that can:
1. Predict the age of a person from an input image of their face.
2. Utilize transfer learning with ResNet50 to reduce training time while maintaining high accuracy.

## Technologies
- Python
- TensorFlow & Keras
- Pandas
- NumPy
- ImageDataGenerator (for data augmentation)
- ResNet50 (for transfer learning)

## Dataset
The dataset used for training and testing the model consists of facial images labeled with the ages of individuals. Due to privacy concerns, the dataset source is not provided in this repository, but commonly used datasets for this type of task include:
- [IMDB-WIKI dataset](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)
- [UTKFace dataset](https://susanqq.github.io/UTKFace/)

Ensure that the dataset is downloaded and stored in the appropriate directory structure before training the model.

## Model Architecture
The project uses the ResNet50 model, a deep convolutional neural network known for its success in image recognition tasks. Key components include:
- **Transfer Learning:** A pre-trained ResNet50 is used, with its top layers removed.
- **GlobalAveragePooling2D and Dense layers:** Added for customization to predict the age.
- **Adam Optimizer:** Used to train the model with an initial learning rate of 0.001.

## Installation
To run this project locally, ensure you have Python installed along with the following packages:

```sh
pip install tensorflow pandas numpy
```

Clone the repository and navigate to the project directory:

```sh
git clone https://github.com/yourusername/Age-Prediction-with-ResNet50.git
cd Age-Prediction-with-ResNet50
```

## Usage
To train the model, use the Jupyter notebook provided:
1. Open the `proyecto_15.ipynb` file.
2. Ensure that the dataset path is correctly configured.
3. Run the cells sequentially to train and evaluate the model.

For predictions on new images, a separate script (`predict_age.py`) can be used to load the trained model and provide predictions.

## Results
The model's performance is evaluated based on Mean Absolute Error (MAE) between the predicted and actual ages. Techniques like data augmentation were used to enhance generalization, resulting in a robust model capable of dealing with the variability in facial features.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your improvements. Ensure that your contributions are well-documented.

## License
This project is licensed under the MIT License. Feel free to use it in your own projects.

---

### Author
- **Your Name** - [GitHub Profile](https://github.com/yourusername)

Feel free to contact me for any questions or suggestions.


