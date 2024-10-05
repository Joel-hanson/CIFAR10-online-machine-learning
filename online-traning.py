import base64
import json
import logging
import sys
import time
import types

# Following session is to fix the six.moves issue in kafka-python package
m = types.ModuleType("kafka.vendor.six.moves", "Mock module")
setattr(m, "range", range)
sys.modules["kafka.vendor.six.moves"] = m

from collections import deque
from threading import Thread

# Kafka and torch imports
import torch
import torch.nn as nn
import torch.optim as optim
from kafka import KafkaConsumer
from torchvision import transforms

# Assuming the Net class is defined as before
from main import Net

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = "./cifar_net.pth"
BOOTSTRAP_SERVER = "localhost:9092"
TOPIC = "cifar10-data"
BATCH_SIZE = 64
UPDATE_INTERVAL = 300  # Update model every 5 minutes

# Initialize Kafka consumer
consumer = KafkaConsumer(
    TOPIC, bootstrap_servers=BOOTSTRAP_SERVER, auto_offset_reset="latest"
)

# Load pre-trained CNN model
model = Net()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# Preprocessing function
preprocess = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

# Class labels
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

# Data accumulator
data_buffer = deque(maxlen=BATCH_SIZE)


def accumulate_data(image, label):
    """Accumulate new data for retraining"""
    data_buffer.append((image, label))


def retrain_model():
    """Retrain the model on accumulated data"""
    global model
    if len(data_buffer) < BATCH_SIZE:
        logger.info("Not enough data for retraining")
        return

    logger.info("Retraining model...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(5):  # Do a few epochs of training
        running_loss = 0.0
        for i, (image, label) in enumerate(data_buffer):
            inputs = preprocess(image).unsqueeze(0)
            labels = torch.tensor([label])

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        logger.info(f"Epoch {epoch+1}, Loss: {running_loss/len(data_buffer):.3f}")

    model.eval()
    logger.info("Model updated")

    # Save the updated model
    torch.save(model.state_dict(), MODEL_PATH)


def update_model_periodically():
    """Periodically retrain the model"""
    while True:
        time.sleep(UPDATE_INTERVAL)
        retrain_model()


def main():
    # Start a thread for periodic model updates
    update_thread = Thread(target=update_model_periodically)
    update_thread.daemon = True
    update_thread.start()

    # Main loop for processing incoming data
    for message in consumer:
        message_value = message.value.decode("utf-8")
        data = json.loads(message_value)
        logger.info(f"Consumed message {data['id']}")

        image_bytes = base64.b64decode(data["image"])
        image = torch.from_numpy(
            torch.frombuffer(image_bytes, dtype=torch.float32).reshape(3, 32, 32)
        )
        label = data["label"]

        # Accumulate new data
        accumulate_data(image, label)

        # Make prediction
        img_tensor = preprocess(image).unsqueeze(0)
        with torch.no_grad():
            output = model(img_tensor)
            _, predicted = torch.max(output, 1)

        logger.info(f"Classified as: {class_names[predicted.item()]}")
        logger.info(f"True label: {class_names[label]}")

        # Check prediction accuracy
        if predicted.item() == label:
            logger.info("Correct prediction!")
        else:
            logger.info("Incorrect prediction.")

        logger.info(f"Current buffer size: {len(data_buffer)}")


if __name__ == "__main__":
    main()
