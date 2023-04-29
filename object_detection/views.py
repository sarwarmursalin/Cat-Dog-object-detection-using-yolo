from django.shortcuts import render

# Create your views here.
# Import necessary libraries

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
from rest_framework.response import Response
import cv2
import numpy as np


def load_yolo_model(model_config_path, model_weights_path):
    # Load YOLOv3 model
    net = cv2.dnn.readNetFromDarknet(model_config_path, model_weights_path)

    # Get output layer names
    layer_names = net.getLayerNames()
    output_layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return net, output_layer_names


# Define API endpoint
@api_view(['GET', 'POST'])
@csrf_exempt
def object_detection(request, class_names=2):
    # Load YOLOv3 model
    net, output_layer_names = load_yolo_model('yolov3.cfg', 'yolov3.weights')

    # Load input image
    image = cv2.imdecode(np.fromstring(request.FILES['image'].read(), np.uint8), cv2.IMREAD_UNCHANGED)

    # Perform object detection
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layer_names)

    # Process the outputs
    detections = []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                x, y, w, h = detection[:4] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
                x, y, w, h = int(x - w / 2), int(y - h / 2), int(w), int(h)
                detections.append({
                    'class_id': int(class_id),
                    'class_name': class_names[class_id],
                    'confidence': float(confidence),
                    'bbox': [x, y, w, h]
                })

    # Return the detections as a JSON response
    return JsonResponse({'detections': detections})
