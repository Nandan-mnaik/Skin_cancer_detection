# Skin Cancer Classification Project

This project implements a machine learning system for classifying skin lesions into different types of skin cancer trained on the HAM10000 dataset. It includes data preprocessing, model training using different architectures (CNN, MobileNetV3, and ResNet50), and a Streamlit web application for easy use of the trained model. All these different models bring out different accuracy values.

## Project Structure

- `app.py`: Streamlit web application for using the trained model
- `cnnmodel.py`: Implementation of a custom CNN model
- `Data_manip.py`: Data preprocessing and splitting script
- `Mobilenet_model.py`: Implementation of a MobileNetV3Large-based model
- `Resnet_model.py`: Implementation of a ResNet50-based model

## Requirements

- Python 3.7+
- TensorFlow 2.x
- Streamlit
- Pillow
- NumPy
- Matplotlib

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/your-username/skin-cancer-classification.git
   cd skin-cancer-classification
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Prepare your data:
   - Place the HAM10000 dataset in the `data` folder
   - Run the data manipulation script:
     ```
     python Data_manip.py
     ```

## Training Models

To train the different models, run the following scripts:

1. CNN Model: This is a built-from-scratch model with a basic structure. It provided an accuracy range of 59%, which was not sufficient for classification. This model was not used in the final application.
   ```
   python cnnmodel.py
   ```

2. MobileNetV3 Model: The last model trained, returning a training accuracy of 82%. This model performs well at classifying images and generates good output.
   ```
   python Mobilenet_model.py
   ```

3. ResNet50 Model: The second model tried in this project. It provided an accuracy of 71%, which was good enough to classify the test set but not sufficient for identifying other images.
   ```
   python Resnet_model.py
   ```

Each script will save the trained model and generate a plot of the training history. The training history plot is included in the repository.

## Running the Web Application

To run the Streamlit web application:

```
streamlit run app.py
```

This will start a local server, and you can access the application through your web browser.

## Model Information

The project includes three different model architectures:

1. Custom CNN: A simple convolutional neural network
2. MobileNetV3Large: A lightweight model suitable for mobile and embedded vision applications
3. ResNet50: A deep residual network known for its performance on image classification tasks

All models are trained on the HAM10000 dataset, which contains images of seven different types of skin lesions. The MobileNetV3 model is recommended for use as it presents the highest accuracy among the three.

## Usage Notes

- The web application allows users to upload an image of a skin lesion and receive a classification prediction.
- The application provides information about the predicted skin condition and confidence scores for all possible classes.
- Remember that this tool is for educational purposes only and should not be used as a substitute for professional medical advice. This tool was developed as a mini-project.

## Contributing

Contributions to this project are welcome. Feel free to train other models using this code. You can visit the Keras documentation to create new models. Please submit a Pull Request with your improvements.
