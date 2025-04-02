# Plant Disease Detection

## Overview
This project focuses on detecting plant diseases using deep learning. A Convolutional Neural Network (CNN) model is trained to classify plant leaves into diseased or healthy categories based on image data.

## Dataset
The dataset consists of labeled images of healthy and diseased plant leaves. These images are preprocessed and fed into the model for training.
📂 **Download Dataset:** [Click Here](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)

## Technologies Used
- Python
- TensorFlow/Keras
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn

## Project Structure
```
📂 Plant_Disease_Detection
├── 📂 train/  # Training dataset
├── 📂 valid/  # Validation dataset
├── 📂 test/   # Test dataset
├── 📂 venv/   # Virtual environment (excluded in .gitignore)
├── main.py    # Script for inference
├── train-plant-disease-notebook.ipynb  # Model training
├── testing-plant-disease-notebook.ipynb # Model testing
├── trained_model.h5  # Saved model file
├── training_history.pkl  # Model training history
├── Home.jpg  # Sample image (excluded in .gitignore)
├── README.md  # Project documentation
├── .gitignore # Ignored files/folders
└── Screen Recording.mp4  # Project demo
```

## Model Training
The model was trained using a CNN architecture with multiple convolutional layers, batch normalization, and dropout for regularization.

### Steps:
1. **Data Preprocessing:** Image resizing, normalization, and augmentation.
2. **Model Training:** Using a CNN with ReLU activation and Softmax output.
3. **Evaluation:** Checking accuracy and loss using validation data.
4. **Testing:** Running predictions on test images.

## Model Performance
| Model  | Training Accuracy | Validation Accuracy |
|--------|------------------|--------------------|
| CNN    | 98.34%           | 96.3%             |

## Demo Video

<video width="100%" controls>
  <source src="Screen_Recording.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>


## How to Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Plant_Disease_Detection.git
   ```
2. Navigate to the project folder:
   ```bash
   cd Plant_Disease_Detection
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run inference on a sample image:
   ```bash
   python main.py --image sample.jpg
   ```

## Contribution
Feel free to fork and contribute to this project! Open a pull request with improvements or bug fixes.
