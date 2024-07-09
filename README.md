# Chocolate Detection Using Faster R-CNN

## Table of Contents
1. [Project Overview](#project-overview)
2. [Project Background](#project-background)
3. [Features](#features)
4. [Contents](#contents)
5. [Requirements](#requirements)
6. [Installation](#installation)
7. [Data Preparation](#data-preparation)
8. [Usage](#usage)
    - [Training the Model](#training-the-model)
    - [Testing the Model](#testing-the-model)
9. [Results](#results)
10. [Detailed Information](#detailed-information)
11. [Contributing](#contributing)
12. [License](#license)
13. [Acknowledgments](#acknowledgments)

## Project Overview

This project aims to utilize Faster R-CNN for the Food Packaging Process, with a specific focus on packaging various types of chocolates in a factory's production line. The objective is to identify chocolates in images captured by a camera to automate the pick-and-place task using a two-fingered gripper and a delta parallel robot. The primary aim is to enhance the precision and efficacy of detecting and packaging vulnerable soft chocolates during the pick-and-place operation without causing any damage while maintaining stringent hygiene standards and reducing the need for manual labor.

## Features

- **Object Detection**: Utilizing Faster R-CNN to accurately detect the position of chocolates.
- **Robotic Integration**: Integration with a two-fingered gripper and Delta Parallel Robot for pick-and-place operation.
- **High Accuracy**: Achieves high accuracy and success rates in detecting and automated pick-and-place operations on practical tests.

------------------------------------------------------------------------------

## Contents

- `Faster R-CNN-[Chocolate Detection].ipynb`: Jupyter notebook containing the implementation of the Faster R-CNN model for chocolate detection.

## Tools and Libraries

- Python 3.7+
- PyTorch
- scikit-learn
- OpenCV
- NumPy
- albumentations
- Roboflow

## Data Preparation

1. **Collect Images**: Gathered images of chocolates in various arrangements.
2. **Annotate Images**: Useed the Roboflow annotation tool to annotate the chocolates in the raw images and saveed the annotations in XML format.
3. **Organize Dataset**: Structured the dataset directory as follows:
   ```
   dataset/
       images/
           img1.jpg
           img2.jpg
           ...
       annotations/
           img1.xml
           img2.xml
           ...
   ```

### Training the Model

1. **Prepare the Dataset**: Ensure your dataset is organized as described in the [Data Preparation](#data-preparation) section.
2. 

### Testing and Evaluation of the Model

1. **Load Trained Model**: Use the trained model saved during the training phase.
2. **Test on New Images**: Run the notebook cells designed for testing and visualize the results.
3. 

## Results

- images ...
- tables ...
- ...


## Contribute

- sarmad zandi goharrizi
- mona mohades mojtahedi

## License

.....

## Acknowledgments

- Special thanks to Dr. Tale Masouleh as the advisor and the Taarlab laboratory members for their guidance and support.
