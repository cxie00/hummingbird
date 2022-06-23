from sklearn.metrics import accuracy_score
import numpy as np
import lightgbm as lgb
from hummingbird.ml import convert

import time
# Create some random data for binary classification.
num_classes = 2
X = np.random.rand(200000, 28)
y = np.random.randint(num_classes, size=200000)
# Create and train a model (LightGBM in this case).
model = lgb.LGBMClassifier()
model.fit(X, y)
file2 = open("old_metrics/LGBM-example_metrics.txt", "w") 
file2.write("\nLGBM-example_metrics.txt\n")

pre_conv = []
pre_acc = []
n = 1
for i in range(0, n):
    start = time.perf_counter()
    pred = model.predict(X)
    end = time.perf_counter()
    pre_conv.append(end - start)
    accuracy = accuracy_score(y,pred)
    pre_acc.append(accuracy)
pre_avg_ms = sum(pre_conv) / n * 1000
pre_acc_avg = sum(pre_acc) / n 
pre_metric = f"Pre-conversion pred average: {pre_avg_ms:.2f} ms over {n} runs \n Pre-conversion accuracy average: {pre_acc_avg:.5f} \n"
file2.write(pre_metric)

# Use Hummingbird to convert the model to PyTorch.
hb_model = convert(model, 'torch')


# Run Hummingbird on CPU - By default CPU execution is used in Hummingbird.
post_conv = []
post_acc = []
for i in range(0, n):
    start = time.perf_counter()    
    pred_cpu_hb = hb_model.predict(X)
    end = time.perf_counter()
    post_conv.append(end - start)
    post_accuracy = accuracy_score(y, pred_cpu_hb)
    post_acc.append(post_accuracy)
post_avg_ms = sum(post_conv) / n * 1000
post_acc_avg = sum(post_acc) / n 
post_metric = f"Post-conversion average: {post_avg_ms:.2f} ms over {n} runs \n Post-conversion accuracy average: {post_acc_avg:.5f} \n"
file2.write(post_metric)

# Run Hummingbird on GPU (Note that you must have a GPU-enabled machine).
hb_model.to('cuda')

post_gpu_conv = []
post_gpu_acc = []
for i in range(0, n):
    start = time.perf_counter()
    pred_gpu_hb = hb_model.predict(X)
    end = time.perf_counter()
    post_gpu_conv.append(end - start)
    post_gpu_accuracy = accuracy_score(y, pred_gpu_hb)
    post_gpu_acc.append(post_gpu_accuracy)
post_gpu_avg_ms = sum(post_gpu_conv) / n * 1000
post_acc_gpu_avg = sum(post_gpu_acc) / n
post_gpu_metric = f"Post-conversion GPU average: {post_gpu_avg_ms:.2f} ms over {n} runs \n Post-conversion GPU accuracy average: {post_acc_gpu_avg:.5f} \n"
file2.write(post_gpu_metric)

file2.close()