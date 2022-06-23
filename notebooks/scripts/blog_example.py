from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from hummingbird.ml import convert
from sklearn.metrics import accuracy_score

# Create and train a RandomForestClassifier model
X, y = load_breast_cancer(return_X_y=True)
skl_model = RandomForestClassifier(n_estimators=500, max_depth=7)
skl_model.fit(X, y)

import time
# file1 = open("metrics/blog_example_metrics.txt", "w") 

pre_conv = []
pre_acc = []
n = 1
for i in range(0, n):
    start = time.perf_counter()
    pred = skl_model.predict(X)
    end = time.perf_counter()
    pre_conv.append(end - start)
    accuracy = accuracy_score(y,pred)
    pre_acc.append(accuracy)
pre_avg_ms = sum(pre_conv) / n * 1000
pre_acc_avg = sum(pre_acc) / n 
pre_metric = f"Pre-conversion pred average: {pre_avg_ms:.2f} ms over {n} runs \n Pre-conversion accuracy average: {pre_acc_avg:.5f} \n"
# file1.write(pre_metric)
print("\n")
print(pre_metric)
print("\n")

model = convert(skl_model, 'torch')

post_conv = []
post_acc = []
for i in range(0, n):
    start = time.perf_counter()
    pred_cpu_hb = model.predict(X)
    end = time.perf_counter()
    post_conv.append(end - start)
    post_accuracy = accuracy_score(y, pred_cpu_hb)
    post_acc.append(post_accuracy)
post_avg_ms = sum(post_conv) / n * 1000
post_acc_avg = sum(post_acc) / n 
post_metric = f"Post-conversion average: {post_avg_ms:.2f} ms over {n} runs \n Post-conversion accuracy average: {post_acc_avg:.5f} \n"
# file1.write(post_metric)
print("\n")
print(post_metric)
print("\n")

# model.to('cuda')

# post_gpu_conv = []
# post_gpu_acc = []
# for i in range(0, n):
#     start = time.perf_counter()
#     pred_gpu_hb = model.predict(X)
#     end = time.perf_counter()
#     post_gpu_conv.append(end - start)
#     post_gpu_accuracy = accuracy_score(y, pred_gpu_hb)
#     post_gpu_acc.append(post_gpu_accuracy)
# post_gpu_avg_ms = sum(post_gpu_conv) / n * 1000
# post_acc_gpu_avg = sum(post_gpu_acc) / n
# post_gpu_metric = f"Post-conversion GPU average: {post_gpu_avg_ms:.2f} ms over {n} runs \n Post-conversion GPU accuracy average: {post_acc_gpu_avg:.5f} \n"
# file1.write(post_gpu_metric)

# file1.close()

# file1 = open("metrics/blog_example_pre_conv.txt", "w") 
# for t in pre_conv:
#     file1.write(f"{t}\n")

# file1.close()

# file1 = open("metrics/blog_example_post_conv.txt", "w") 
# for t in post_conv:
#     file1.write(f"{t}\n")

# file1.close()

# file1 = open("metrics/blog_example_post_conv_gpu.txt", "w") 
# for t in post_gpu_conv:
#     file1.write(f"{t}\n")

# file1.close()