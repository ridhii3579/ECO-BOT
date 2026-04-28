# ECO-BOT
## Overview
#### EcoBot is an AI-powered garbage classification system that uses deep learning to automatically categorize waste into different classes such as plastic, glass, metal, paper, cardboard, and trash. The project aims to improve waste segregation efficiency and promote sustainable recycling practices.

#### The model is trained using a Convolutional Neural Network and deployed on an embedded system for real-time inference.

## Objectives
#### Automate waste classification using AI
#### Reduce human effort in garbage segregation
#### Enable real-time prediction using camera input
#### Deploy model on an edge device for practical use

## Model Used
### MobileNetV2 (Transfer Learning)
#### Lightweight and efficient CNN architecture
#### Suitable for real-time inference on low-power devices

## Dataset
### Garbage Classification Dataset
### Classes:
#### Cardboard
#### Glass
#### Metal
#### Paper
#### Plastic
#### Trash

## Methodology
### Data Preprocessing
#### Image resizing (224×224)
#### Normalization
#### Data loading using PyTorch DataLoader
### Model Training
#### Pretrained MobileNetV2 used
#### Feature layers frozen
#### Final classification layer modified
#### Trained for 50 epochs
### Evaluation
#### Validation accuracy used for model selection
#### Loss and accuracy graphs generated
### Deployment
#### Model saved as .pth file
#### Deployed on Jetson Nano
#### Real-time prediction using camera

## Tech Stack
#### Python
#### PyTorch
#### OpenCV
#### Torchvision
#### Matplotlib

## Features
#### Image-based garbage classification
#### Real-time camera prediction
#### Lightweight model for embedded systems
#### Deployment on Jetson Nano
#### Graph visualization of training performance

## Usage
### Run on Image
#### python inference.py
##### Enter image path when prompted

### Run on Camera
#### python camera.py

## Results
#### Model successfully classifies waste into 6 categories
#### Achieves good accuracy on test data

## Limitations
#### Accuracy depends on dataset quality
#### Sensitive to lighting and background
#### Limited performance on mixed waste images
 
## Future Improvements
#### Use YOLO for object detection
#### Improve dataset diversity
#### Add robotic sorting mechanism
#### Optimize using TensorRT for faster inference

## Project Structure
### eco_bot/
#### │── train.py
#### │── inference.py
#### │── camera.py
#### │── models/
#### │── utils/
#### │── data/

## Learning Outcomes
### Understanding of CNN and transfer learning
### Experience with PyTorch model training
### Deployment on embedded AI hardware
### Real-time computer vision implementation

## Acknowledgements
### Open-source dataset contributors
### PyTorch and Torchvision libraries

## Conclusion

### EcoBot demonstrates how AI can be applied to real-world environmental problems by automating waste classification and enabling smart recycling solutions.
