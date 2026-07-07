import tritonclient.grpc as grpcclient
import numpy as np
import cv2
import torch
from  utils import py_cpu_nms
from functions import PriorBox
from config import cfg_re50
model_name = "retina_facetrt"
input_name = "input0"
output_names = ["loc", "conf", "landms"]

# Read and preprocess the image
img_raw = cv2.imread(r"C:\Users\18088\Pictures\2.jpeg", cv2.IMREAD_COLOR)
img_resized = cv2.resize(img_raw, (640, 640))  # Resize the image to 640x640

img = np.float32(img_resized)
img -= (104, 117, 123)
img = img.transpose(2, 0, 1)
img = np.expand_dims(img, axis=0)

# Create Triton client
triton_client = grpcclient.InferenceServerClient(url="192.168.1.115:40008")

# Create the input tensor
inputs = []
inputs.append(grpcclient.InferInput(input_name, img.shape, "FP32"))
inputs[0].set_data_from_numpy(img)

# Create the output tensors
outputs = []
for output_name in output_names:
    outputs.append(grpcclient.InferRequestedOutput(output_name))

# Perform inference
response = triton_client.infer(model_name=model_name, inputs=inputs, outputs=outputs)

# Get the outputs
loc = response.as_numpy("loc")
conf = response.as_numpy("conf")
landms = response.as_numpy("landms")

print(f'loc: {loc.shape}')
print(f'conf: {conf.shape}')
print(f'landms: {landms.shape}')

# Decode boxes, scores, and landmarks
def decode(loc, priors, variances):
    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

def decode_landm(landm, priors, variances):
    landms = torch.cat((
        priors[:, :2] + landm[:, :2] * variances[0] * priors[:, 2:],
        priors[:, :2] + landm[:, 2:4] * variances[0] * priors[:, 2:],
        priors[:, :2] + landm[:, 4:6] * variances[0] * priors[:, 2:],
        priors[:, :2] + landm[:, 6:8] * variances[0] * priors[:, 2:],
        priors[:, :2] + landm[:, 8:10] * variances[0] * priors[:, 2:],
    ), dim=1)
    return landms

# Assuming we have the prior boxes from the PriorBox layer
priorbox = PriorBox(cfg_re50, image_size=(640, 640))
priors = priorbox.forward()
priors = priors.data.numpy()

boxes = decode(torch.tensor(loc[0]), torch.tensor(priors), cfg_re50['variance']).numpy()
scores = conf[0][:, 1]
landms = decode_landm(torch.tensor(landms[0]), torch.tensor(priors), cfg_re50['variance']).numpy()

# Scale boxes and landmarks to the original image size
scale = np.array([img_raw.shape[1], img_raw.shape[0], img_raw.shape[1], img_raw.shape[0]])
boxes = boxes * scale
landms_scale = np.array([img_raw.shape[1], img_raw.shape[0]] * 5)
landms = landms * landms_scale

# Filter boxes with a threshold
threshold = 0.2
inds = np.where(scores > threshold)[0]
boxes = boxes[inds]
landms = landms[inds]
scores = scores[inds]

# Apply NMS
dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
print(dets)
print(scores)
keep = py_cpu_nms(dets, 0.4)
print(keep)
dets = dets[keep, :]
landms = landms[keep]
print(dets)
print(landms)
# Function to align face based on landmarks
def align_face(img, landmarks):
    # Assume the landmarks are in the order of left_eye, right_eye, nose, left_mouth, right_mouth
    left_eye = landmarks[:2]
    right_eye = landmarks[2:4]
    nose = landmarks[4:6]
    left_mouth = landmarks[6:8]
    right_mouth = landmarks[8:10]

    # Calculate the angle for rotation
    eye_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))

    # Get the rotation matrix
    rot_mat = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
    # Rotate the entire image
    aligned_img = cv2.warpAffine(img, rot_mat, (img.shape[1], img.shape[0]))

    # Apply the same transformation to landmarks
    landmarks = np.array(landmarks).reshape(5, 2)
    ones = np.ones(shape=(len(landmarks), 1))
    points_ones = np.hstack([landmarks, ones])
    transformed_landmarks = rot_mat.dot(points_ones.T).T

    return aligned_img, transformed_landmarks

# Crop and align faces from the original image
for i, b in enumerate(dets):
    if b[4] < threshold:
        continue
    b = list(map(int, b))
    
    # Get landmarks for the current face
    landmarks = landms[i]

    # Align the face
    aligned_img, aligned_landmarks = align_face(img_raw, landmarks)

    # Crop the face from the aligned image
    face = aligned_img[b[1]:b[3], b[0]:b[2]]
    cv2.imwrite(f"aligned_face_{i}.jpg", face)
    print(f"Cropped and aligned face {i} saved as aligned_face_{i}.jpg")
