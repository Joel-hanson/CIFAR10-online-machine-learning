import json

import numpy as np
import torch
from model import Net  # Import our CIFAR-10 model
from pyflink.common.serialization import SimpleStringSchema
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.connectors import FlinkKafkaConsumer, FlinkKafkaProducer

env = StreamExecutionEnvironment.get_execution_environment()

# Define Kafka consumer
kafka_consumer = FlinkKafkaConsumer(
    "cifar10-data",
    SimpleStringSchema(),
    properties={"bootstrap.servers": "localhost:9092", "group.id": "cifar10-group"},
)

# Define Kafka producer for results
kafka_producer = FlinkKafkaProducer(
    "classification-results",
    SimpleStringSchema(),
    properties={"bootstrap.servers": "localhost:9092"},
)

# Add consumer to the streaming environment
stream = env.add_source(kafka_consumer)

# Load the CIFAR-10 model
model = Net()
model.load_state_dict(torch.load("cifar_net.pth"))
model.eval()


# Process function to classify images and update model
def classify_and_update(data):
    data = json.loads(data)
    image = torch.from_numpy(
        np.frombuffer(data["image"], dtype=np.float32).reshape(3, 32, 32)
    )
    label = torch.tensor(data["label"])

    # Perform classification
    with torch.no_grad():
        output = model(image.unsqueeze(0))
        _, predicted = torch.max(output, 1)

    # Update the model (online learning)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss = criterion(output, label.unsqueeze(0))
    loss.backward()
    optimizer.step()

    return json.dumps({"true_label": label.item(), "predicted": predicted.item()})


# Apply the processing function and write results back to Kafka
stream.map(classify_and_update).add_sink(kafka_producer)

# Execute the Flink job
env.execute("CIFAR-10 Classification Job")
