from sklearn.metrics import accuracy_score
from hummingbird.ml import convert

import zipfile
import urllib.request as urllib
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip'

filehandle, _ = urllib.urlretrieve(url)
zip_file_object = zipfile.ZipFile(filehandle, 'r')
filename = zip_file_object.namelist()[0]
bytes_data = zip_file_object.open(filename).read()


import pandas as pd
from io import BytesIO
from sklearn.model_selection import train_test_split

year = pd.read_csv(BytesIO(bytes_data), header = None)

X = year.iloc[:, 1:]
y = year.iloc[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, train_size=463715, test_size=51630)

# Store the test data as numpy by pulling the values out of the pandas dataframe
data = X_test.values

import lightgbm as lgb
lgbm_model = lgb.LGBMRegressor(max_depth=8,n_estimators=500, num_leaves=256, objective="regression", reg_lambda=1, n_jobs=6)
lgbm_model.fit(X_train, y_train)

import time
file1 = open("old_metrics/LGBM_year_with_train_metrics.txt", "w") 
file1.write("\nLGBM_year_with_train_metrics.txt\n")

pre_conv = []
pre_acc = []
n = 1
for i in range(0, n):
    start = time.perf_counter()
    lgbm_time = lgbm_model.predict(data)
    end = time.perf_counter()
    pre_conv.append(end - start)
    accuracy = accuracy_score(y,lgbm_time)
    pre_acc.append(accuracy)
pre_avg_ms = sum(pre_conv) / n * 1000
pre_acc_avg = sum(pre_acc) / n 
pre_metric = f"Pre-conversion pred average: {pre_avg_ms:.2f} ms over {n} runs \n Pre-conversion accuracy average: {pre_acc_avg:.5f} \n"
file1.write(pre_metric)

model = convert(lgbm_model, 'torch')

post_conv = []
post_acc = []
for i in range(0, n):
    start = time.perf_counter()
    pred_cpu_hb = model.predict(data)
    end = time.perf_counter()
    post_conv.append(end - start)
    post_accuracy = accuracy_score(y, pred_cpu_hb)
    post_acc.append(post_accuracy)
post_avg_ms = sum(post_conv) / n * 1000
post_acc_avg = sum(post_acc) / n 
post_metric = f"Post-conversion average: {post_avg_ms:.2f} ms over {n} runs \n Post-conversion accuracy average: {post_acc_avg:.5f} \n"
file1.write(post_metric)

model.to('cuda')

post_gpu_conv = []
post_gpu_acc = []
for i in range(0, n):
    start = time.perf_counter()
    pred_gpu_hb = model.predict(data)
    end = time.perf_counter()
    post_gpu_conv.append(end - start)
    post_gpu_accuracy = accuracy_score(y, pred_gpu_hb)
    post_gpu_acc.append(post_gpu_accuracy)
post_gpu_avg_ms = sum(post_gpu_conv) / n * 1000
post_acc_gpu_avg = sum(post_gpu_acc) / n
post_gpu_metric = f"Post-conversion GPU average: {post_gpu_avg_ms:.2f} ms over {n} runs \n Post-conversion GPU accuracy average: {post_acc_gpu_avg:.5f} \n"
file1.write(post_gpu_metric)

file1.close()