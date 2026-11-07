Hand Gesture Recognition System

A Deep Learning computer vision model built with TensorFlow and OpenCV to classify American Sign Language (ASL) hand gestures (24 letter classes) using VGG16 Transfer Learning.

🚀 Quick Start

1. Install Dependencies
pip install numpy tensorflow opencv-python matplotlib scikit-learn seaborn


2. Dataset Setup
Organize your dataset as follows:
Database/
├── train/ (folders A-Y)
└── test/  (folders A-Y)

3. Usage
Update the training and testing directory paths in the script and run:
python main.py

Features & Stack
- Architecture: VGG16 (ImageNet transfer learning) + Custom Dense Classifier
- Preprocessing: Image resizing (100 x 100) , RGB conversion, and [0, 1] normalization
- Augmentation: Real-time ImageDataGenerator (rotation, shift, zoom, flip)
- Evaluation: Training loss/accuracy plots & test set confusion matrices


