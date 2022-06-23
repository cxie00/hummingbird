import pandas as pd
import numpy as np
from prophet import Prophet
import hummingbird.ml
import time 
# Read the data and train a Prophet model
DATA = "../resources/example_wp_log_peyton_manning.csv"
df = pd.read_csv(DATA)

model = Prophet()
model.fit(df)

# Convert into PyTorch using Hummingbird
hb_model = hummingbird.ml.convert(model, "torch")

# Generate future dataset.
future = model.make_future_dataframe(periods=365)

# Trend prediction with prophet
prophet_trend = model.predict(future)["trend"]

# Trend prediction with Hummingbird
file1 = open("metrics/prohpet-example_metrics.txt", "w") 
n = 10000
post_conv = []
for i in range(0, n):
    start = time.perf_counter()
    hb_trend = hb_model.predict(future)
    end = time.perf_counter()
    post_conv.append(end - start)
post_avg_ms = sum(post_conv) / n * 1000
post_metric = f"Post-conversion average: {post_avg_ms:.2f} ms over {n} runs \n"
file1.write(post_metric)

# Check that the predictions match.
np.testing.assert_allclose(prophet_trend, hb_trend, rtol=1e-06, atol=1e-06)

pre_conv = []
for i in range(0, n):
    start = time.perf_counter()
    prophet_trend = model.predict(future)
    end = time.perf_counter()
    pre_conv.append(end - start)
pre_avg_ms = sum(pre_conv) / n * 1000
pre_metric = f"Pre-conversion average: {pre_avg_ms:.2f} ms over {n} runs \n"
file1.write(pre_metric)


# Generate onnx model. ONNX model requires inputs in numpy format.
future_np = (future.values - np.datetime64("1970-01-01T00:00:00.000000000")).astype(np.int64) / 1000000000
hb_onnx_model = hummingbird.ml.convert(model, "onnx", future_np)

hb_trend = hb_onnx_model.predict(future_np)

# Test on GPU
hb_model.to('cuda')
post_gpu_conv = []
for i in range(0, n):
    start = time.perf_counter()
    hb_trend = hb_model.predict(future)
    end = time.perf_counter()
    post_gpu_conv.append(end - start)
post_gpu_avg_ms = sum(post_gpu_conv) / n * 1000
post_gpu_metric = f"Post-conversion GPU average: {post_gpu_avg_ms:.2f} ms over {n} runs \n"
file1.write(post_gpu_metric)

