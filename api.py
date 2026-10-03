import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from fastapi import FastAPI, UploadFile
from fastapi.responses import Response
from PIL import Image
from fastapi.responses import RedirectResponse



import src

app = FastAPI()

model = src.KITTIdepthNET(pretrained=False)
model.load_state_dict(torch.load("weights/kitti.pt", map_location="cpu", weights_only=True)["depth_net"])
model.eval()
normalize = src.dataloaders.NormalizeKittiImageNet()

@app.get("/")
def root():
    return RedirectResponse("/docs")

@app.post("/predict")
def predict(file: UploadFile):
    img = Image.open(file.file).convert("RGB").resize((640, 192), Image.LANCZOS)
    x = normalize(TF.to_tensor(img)).unsqueeze(0)

    with torch.no_grad():
        disp = model(x)[0][0, 0].numpy()

    disp = (disp / disp.max() * 255).astype(np.uint8)
    color = cv2.applyColorMap(disp, cv2.COLORMAP_MAGMA)
    png = cv2.imencode(".png", color)[1].tobytes()

    return Response(png, media_type="image/png")