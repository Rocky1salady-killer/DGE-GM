import cv2
import torch
from model import GazeSwinTransformer  # Load the mod script you defined, making sure the mod is consistent with the checkpoint.
from torchvision import transforms
import numpy as np

# Load the trained model
model_path = 'Iter_60_mp2.pt'  # Replace with your checkpoint path
model = GazeSwinTransformer()
model.load_state_dict(torch.load(model_path))
model.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Define preprocessing pipeline
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])






from facenet_pytorch import MTCNN

# Initialize MTCNN face detector£¨You can replace any face detector£©
mtcnn = MTCNN(keep_all=True, device=device)

def process_frame(frame):
    # Detect faces in the frame
    boxes, _ = mtcnn.detect(frame)
    
    # If no faces are detected, return the original frame
    if boxes is None:
        return frame

    # For each detected face
    for box in boxes:
        # Crop the face from the frame
        x1, y1, x2, y2 = [int(coord) for coord in box]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 3)  # Blue color for the face box
        face = frame[y1:y2, x1:x2]

        # Preprocess the face
        input_tensor = transform(face).unsqueeze(0).to(device)

        # Perform inference
        with torch.no_grad():
            gaze, _ = model({"face": input_tensor})

        # Convert gaze to arrow direction
        dx, dy = -gaze[0, 0].item(), gaze[0, 1].item()
        start_point = ((x1 + x2) // 2, (y1 + y2) // 2)
        end_point = (int(start_point[0] + dx * 500), int(start_point[1] - dy * 500))

        # Draw the arrow on the frame
        cv2.arrowedLine(frame, start_point, end_point, (0, 0, 255), 10)

    return frame






# Input can be an image or a video file
input_path = 'demotest.mp4'  # Replace with your input path
output_path = 'output/demotest.mp4'  # Replace with your output path

if input_path.endswith('.jpg') or input_path.endswith('.png'):
    img = cv2.imread(input_path)
    output_img = process_frame(img)
    cv2.imwrite(output_path, output_img)

else:
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        output_frame = process_frame(frame)
        out.write(output_frame)

    cap.release()
    out.release()
