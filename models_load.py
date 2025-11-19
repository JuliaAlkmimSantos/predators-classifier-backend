import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import json
import xgboost as xgb

#device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#load classes
with open("classes.json", "r") as f:
    classes = json.load(f)

num_classes = len(classes)

#load resnet model
resnet = models.resnet34(pretrained=True)
resnet.fc = nn.Identity()  # remove final layer to get features
resnet.load_state_dict(torch.load("resnet34.pth", map_location=device))
resnet = resnet.to(device)
resnet.eval()

#load xgboost model
xgb_model = xgb.XGBClassifier()
xgb_model.load_model("xgb_model.json")

#image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


#classificaion function
def classify_image(img):

    


    img = img.convert("RGB")
    
    # preprocess
    x = transform(img).unsqueeze(0).to(device)

    # feature extraction
    with torch.no_grad():
        feats = resnet(x).cpu().numpy()

    # xgboost prediction
    probs = xgb_model.predict_proba(feats)[0]

    pred_class = np.argmax(probs)

    return classes[pred_class], probs
