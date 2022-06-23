import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from hummingbird.ml import convert

# We are going to use the breast cancer dataset from scikit-learn for this example.
X, y = load_breast_cancer(return_X_y=True)
nrows=15000
X = X[0:nrows]
y = y[0:nrows]

# Create and train a random forest model.
model = RandomForestClassifier(n_estimators=10, max_depth=10)
model.fit(X, y)

# Time for scikit-learn.
import time
file1 = open("metrics/sklearn-random-forest-example_metrics.txt", "w") 

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

model = convert(model, 'torch', extra_config={"tree_implementation":"gemm"})

# Time for HB.
post_conv = []
for i in range(0, n):
    start = time.perf_counter()
    model.predict(X)
    end = time.perf_counter()
    post_conv.append(end - start)
post_avg_ms = sum(post_conv) / n * 1000
post_metric = f"Post-conversion average: {post_avg_ms:.2f} ms over {n} runs \n"
file1.write(post_metric)


model.to('cuda')

# Time for HB GPU.
post_gpu_conv = []
for i in range(0, n):
    start = time.perf_counter()
    model.predict(X)
    end = time.perf_counter()
    post_gpu_conv.append(end - start)
post_gpu_avg_ms = sum(post_gpu_conv) / n * 1000
post_gpu_metric = f"Post-conversion GPU average: {post_gpu_avg_ms:.2f} ms over {n} runs \n"
file1.write(post_gpu_metric)
