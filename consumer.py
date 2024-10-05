import base64
import json
import logging
import sys
import types

# Following session is to fix the six.moves issue in kafka-python package
m = types.ModuleType("kafka.vendor.six.moves", "Mock module")
setattr(m, "range", range)
sys.modules["kafka.vendor.six.moves"] = m

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils
import torchvision.transforms as transforms
from kafka import KafkaConsumer

from main import Net

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


MODEL_PATH = "./cifar_net.pth"
DATA_PATH = "./data"
BOOTSTRAP_SERVER = "localhost:9092"
TOPIC = "cifar10-data"

# Initialize Kafka consumer
consumer = KafkaConsumer(
    TOPIC, bootstrap_servers=BOOTSTRAP_SERVER, auto_offset_reset="latest"
)


model = Net()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# Pre-processing function
preprocess = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((32, 32)),  # Assuming the model expects 32x32 images
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

# Class labels (example for CIFAR-10)
class_names = [
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def imshow(img):
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()


# Process the images in real-time
for message in consumer:
    message_value = message.value.decode("utf-8")
    data = json.loads(message_value)
    logger.info(f"Consumed message {data["id"]}")

    image_bytes = base64.b64decode(data["image"])
    image = np.frombuffer(image_bytes, dtype=np.float32).reshape(3, 32, 32)
    # imshow(image)
    image = torch.from_numpy(image)
    label = torch.tensor(data["label"])

    img_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)

    logger.info(f"Classified as: {class_names[predicted.item()]}")
