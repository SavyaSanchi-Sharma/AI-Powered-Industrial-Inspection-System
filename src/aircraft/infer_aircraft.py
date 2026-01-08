# src/aircraft/infer_aircraft.py
import torch
import cv2

from .aircraft_model import AircraftDamageModel
from .gradcam import GradCAM
from ..preprocess import preprocess_image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = AircraftDamageModel().to(DEVICE)
model.load_state_dict(torch.load("aircraft_damage_model.pth"))
model.eval()

gradcam = GradCAM(model, model.model.layer4)

img_path = "C:\\Users\\mdkai\\OneDrive\\Desktop\\AIML PROJECTS\\Aerothon\\Aerothon\\aircraft_damage_dataset_v1\\test\\47_6_JPG_jpg.rf.f806f6f7dd3487c687b6530e588a8618.jpg"
img = preprocess_image(img_path)
if img is None:
    raise FileNotFoundError(
        f"Could not read image at path: {img_path}"
    )

img_tensor = torch.tensor(img).permute(2,0,1).unsqueeze(0).to(DEVICE)
cam = gradcam.generate(img_tensor)[0]

cam = cv2.resize(cam, (224,224))
heatmap = cv2.applyColorMap((cam*255).astype("uint8"), cv2.COLORMAP_JET)

original = cv2.resize(cv2.imread(img_path), (224,224))
output = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

cv2.imshow("Aircraft Damage Localization", output)
cv2.waitKey(0)
