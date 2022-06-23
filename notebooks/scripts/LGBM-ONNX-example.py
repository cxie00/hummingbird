
import numpy as np
import lightgbm as lgb

from hummingbird.ml import convert

import time 
# Create some random data for binary classification.
num_classes = 2
X = np.array(np.random.rand(10000, 28), dtype=np.float32)
y = np.random.randint(num_classes, size=10000)

# Create and train a model (LightGBM in this case).
model = lgb.LGBMClassifier()
model.fit(X, y)

file1 = open("metrics/LGBM-ONNX-example_metrics.txt", "w") 

# predict on the original model
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
onnx_model = convert(model, "onnx", X)

# Run the ONNX model on CPU (if you have onnxruntime installed) or GPU (if you have onnxruntime-gpu installed)
post_conv = []
for i in range(0, n):
    start = time.perf_counter()
    onnx_model.predict(X)
    end = time.perf_counter()
    post_conv.append(end - start)
post_avg_ms = sum(post_conv) / n * 1000
post_metric = f"Post-conversion average: {post_avg_ms:.2f} ms over {n} runs \n"
file1.write(post_metric)