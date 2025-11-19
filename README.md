
# Australian Wildlife Classifier

This project classifies Australian wildlife species using a hybrid ML pipeline:
- **ResNet34** for feature extraction
- **XGBoost** for classification

It includes a Flask API backend and can be connected to a front-end website for live predictions.


## Project Structure

wildlife-classifier-backend/
├── app.py                → Flask API for image classification
├── model_utils.py        → Loads models & preprocessing functions
├── classes.json          → Species labels
├── resnet34.pth          → Trained PyTorch ResNet34 model
├── xgb_model.json        → Trained XGBoost model
├── requirements.txt      → Python dependencies
└── README.md             → This file
