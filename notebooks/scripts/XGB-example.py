import numpy as np
import xgboost as xgb
from hummingbird.ml import convert

# Create some random data for binary classification.
num_classes = 2
X = np.random.rand(100000, 28)
y = np.random.randint(num_classes, size=100000)

# Create and train a model (XGBoost in this case).
model = xgb.XGBRegressor()
model.fit(X, y)

# predict on xgboost model
import time
file1 = open("metrics/XGB-example_metrics.txt", "w") 

pre_conv = []
n = 10000
for i in range(0, n):
    start = time.perf_counter()
    model.predict(X)
    end = time.perf_counter()
    pre_conv.append(end - start)
pre_avg_ms = sum(pre_conv) / n * 1000
pre_metric = f"Pre-conversion average: {pre_avg_ms:.2f} ms over {n} runs \n"
file1.write(pre_metric)


# Use Hummingbird to convert the model to PyTorch
# Note that XGBRegressor requires us to pass it some sample data.
hb_model = convert(model, 'torch', X[0:1])

# Run Hummingbird on CPU - By default CPU execution is used in Hummingbird.
post_conv = []
for i in range(0, n):
    start = time.perf_counter()
    hb_model.predict(X)
    end = time.perf_counter()
    post_conv.append(end - start)
post_avg_ms = sum(post_conv) / n * 1000
post_metric = f"Post-conversion average: {post_avg_ms:.2f} ms over {n} runs \n"
file1.write(post_metric)


# Run Hummingbird on GPU (Note that you must have a GPU-enabled machine).
hb_model.to('cuda')
post_gpu_conv = []
for i in range(0, n):
    start = time.perf_counter()
    hb_model.predict(X)
    end = time.perf_counter()
    post_gpu_conv.append(end - start)
post_gpu_avg_ms = sum(post_gpu_conv) / n * 1000
post_gpu_metric = f"Post-conversion GPU average: {post_gpu_avg_ms:.2f} ms over {n} runs \n"
file1.write(post_gpu_metric)
