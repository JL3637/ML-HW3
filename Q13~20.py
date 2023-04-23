import numpy as np
import random
import math

filename1 = 'train_data.txt'
train_N = 100
x_vector_array = np.ones([train_N, 11])
y_array = np.zeros(train_N)
with open(filename1) as file:
    a = 0
    for line in file:
        line = line.strip().split()
        for i in range(10):
            x_vector_array[a][i+1] = line[i]
        x_vector_array[a][0] = 1
        y_array[a] = line[10]
        a += 1
file.close()

filename2 = 'test_data.txt'
test_N = 400
x_vector_array_test = np.ones([test_N, 11])
y_array_test = np.zeros(test_N)
with open(filename2) as file:
    a = 0
    for line in file:
        line = line.strip().split()
        for i in range(10):
            x_vector_array_test[a][i+1] = line[i]
        x_vector_array_test[a][0] = 1
        y_array_test[a] = line[10]
        a += 1
file.close()


w_lin = np.matmul(np.matmul(np.linalg.inv(np.matmul(x_vector_array.transpose(), x_vector_array)), x_vector_array.transpose()), y_array)
E_sqr_in = (1/train_N) * (np.linalg.norm(np.matmul(x_vector_array, w_lin) - y_array) ** 2)
print(E_sqr_in)    #Q13


eta = 0.001
E_in_sum = 0
for i in range(1000):
    w_sgd = np.zeros(11)
    for j in range(800):
        tmp = random.randint(0, train_N-1)
        w_sgd = w_sgd - eta * 2 * (np.matmul(w_sgd.transpose(), x_vector_array[tmp]) - y_array[tmp]) * x_vector_array[tmp]
    E_in_sum += (1/train_N) * (np.linalg.norm(np.matmul(x_vector_array, w_sgd) - y_array) ** 2)
E_sqr_in_sgd = E_in_sum / 1000
print(E_sqr_in_sgd)    #Q14


E_in_sum = 0
for i in range(1000):
    w_ce = np.zeros(11)
    for j in range(800):
        tmp = random.randint(0, train_N-1)
        w_ce = w_ce + eta * (1 / (1 + np.exp(y_array[tmp] * np.matmul(w_ce.transpose(), x_vector_array[tmp])))) * (y_array[tmp] * x_vector_array[tmp])
    E_in_tmp = 0
    for j in range(train_N):
        E_in_tmp += np.log(1 + np.exp(-y_array[tmp] * np.matmul(w_ce.transpose(), x_vector_array[tmp])))
    E_in_sum += E_in_tmp / train_N
E_ce_in = E_in_sum / 1000
print(E_ce_in)    #Q15


E_in_sum = 0
for i in range(1000):
    w_ce = w_lin
    for j in range(800):
        tmp = random.randint(0, train_N-1)
        w_ce = w_ce + eta * (1 / (1 + np.exp(y_array[tmp] * np.matmul(w_ce.transpose(), x_vector_array[tmp])))) * (y_array[tmp] * x_vector_array[tmp])
    E_in_tmp = 0
    for j in range(train_N):
        E_in_tmp += np.log(1 + np.exp(-y_array[tmp] * np.matmul(w_ce.transpose(), x_vector_array[tmp])))
    E_in_sum += E_in_tmp / train_N
E_ce_in = E_in_sum / 1000
print(E_ce_in)    #Q16


E_in_minus_E_out_sum = 0
for i in range(1000):
    w_ce = w_lin
    for j in range(800):
        tmp = random.randint(0, train_N-1)
        w_ce = w_ce + eta * (1 / (1 + np.exp(y_array[tmp] * np.matmul(w_ce.transpose(), x_vector_array[tmp])))) * (y_array[tmp] * x_vector_array[tmp])
    wrong_data = 0
    for j in range(train_N):
        if np.sign(np.matmul(w_ce.transpose(), x_vector_array[j])) != y_array[j]:
            wrong_data += 1
    E_in_tmp = wrong_data / train_N
    wrong_data = 0
    for j in range(test_N):
        if np.sign(np.matmul(w_ce.transpose(), x_vector_array_test[j])) != y_array_test[j]:
            wrong_data += 1
    E_out_tmp = wrong_data / test_N
    E_in_minus_E_out_sum += abs(E_in_tmp - E_out_tmp)
E_in_minus_E_out = E_in_minus_E_out_sum / 1000
print(E_in_minus_E_out)    #Q17


wrong_data = 0
for i in range(train_N):
    if np.sign(np.matmul(w_lin.transpose(), x_vector_array[i])) != y_array[i]:
        wrong_data += 1
E_in_tmp = wrong_data / train_N
wrong_data = 0
for i in range(test_N):
    if np.sign(np.matmul(w_lin.transpose(), x_vector_array_test[i])) != y_array_test[i]:
        wrong_data += 1
E_out_tmp = wrong_data / test_N
E_in_minus_E_out = abs(E_in_tmp - E_out_tmp)
print(E_in_minus_E_out)    #Q18


Q = 2    #change Q for Q19 and Q20
x_transform_array = np.ones([train_N, 1 + 10*Q])
for i in range(train_N):
    x_transform_array[i][0] = 1
    for j in range(Q):
        for k in range(1, 11):
            x_transform_array[i][k + 10 * j] = x_vector_array[i][k] ** (j+1)
x_transform_array_test = np.ones([test_N, 1 + 10*Q])
for i in range(test_N):
    x_transform_array_test[i][0] = 1
    for j in range(Q):
        for k in range(1, 11):
            x_transform_array_test[i][k + 10 * j] = x_vector_array_test[i][k] ** (j+1)
w_lin_tran = np.matmul(np.matmul(np.linalg.inv(np.matmul(x_transform_array.transpose(), x_transform_array)), x_transform_array.transpose()), y_array)
wrong_data = 0
for i in range(train_N):
    if np.sign(np.matmul(w_lin_tran.transpose(), x_transform_array[i])) != y_array[i]:
        wrong_data += 1
E_in_tmp = wrong_data / train_N
wrong_data = 0
for i in range(test_N):
    if np.sign(np.matmul(w_lin_tran.transpose(), x_transform_array_test[i])) != y_array_test[i]:
        wrong_data += 1
E_out_tmp = wrong_data / test_N
E_in_minus_E_out = abs(E_in_tmp - E_out_tmp)
print(E_in_minus_E_out)    #Q19 and Q20