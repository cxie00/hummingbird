import torch
import numpy as np
import lightgbm as lgb

import onnxruntime as ort
from onnxmltools.convert import convert_lightgbm
from onnxconverter_common.data_types import FloatTensorType

from hummingbird.ml import convert
from hummingbird.ml import constants
import time
# Create some random data for binary classification.
num_classes = 2
X = np.array(np.random.rand(10000, 28), dtype=np.float32)
y = np.random.randint(num_classes, size=10000)

# Create and train a model (LightGBM in this case).
model = lgb.LGBMClassifier()
model.fit(X, y)

# Use ONNXMLTOOLS to convert the model to ONNXML.
initial_types = [("input", FloatTensorType([X.shape[0], X.shape[1]]))] # Define the inputs for the ONNX
onnx_ml_model = convert_lightgbm(
    model, initial_types=initial_types, target_opset=9
)
# Run the ONNX model on CPU 
file1 = open("metrics/LGBM-ONNXML-example_metrics.txt", "w") 

pre_conv = []
n = 1
for i in range(0, n):
    start = time.perf_counter()
    model.predict(X)
    end = time.perf_counter()
    pre_conv.append(end - start)
pre_avg_ms = sum(pre_conv) / n * 1000
pre_metric = f"Pre-conversion average: {pre_avg_ms:.2f} ms over {n} runs \n"
file1.write(pre_metric)


# Use Hummingbird to convert the ONNXML model to ONNX.
onnx_model = convert(onnx_ml_model, "onnx")

# Run the ONNX model on CPU 
post_conv = []
for i in range(0, n):
    start = time.perf_counter()
    onnx_model.predict(X)
    end = time.perf_counter()
    post_conv.append(end - start)
post_avg_ms = sum(post_conv) / n * 1000
post_metric = f"Post-conversion average: {post_avg_ms:.2f} ms over {n} runs \n"
file1.write(post_metric)