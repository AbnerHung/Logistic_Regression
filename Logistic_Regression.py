# -*- coding: UTF-8 -*-

import torch
import numpy as np

'''超参数设定'''
BATCH_SIZE = 32

'''数据生成'''
n_data = torch.ones(1000, 2)
xx = torch.ones(2000)
# print(xx)
x0 = torch.normal(2 * n_data, 1)  # 生成均值为2.标准差为1的随机数组成的矩阵 shape=(100, 2)
y0 = torch.zeros(1000)
x1 = torch.normal(-2 * n_data, 1)  # 生成均值为-2.标准差为1的随机数组成的矩阵 shape=(100, 2)
y1 = torch.ones(1000)


#合并数据x,y
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)

# print(x)
y = torch.cat((y0, y1), 0).type(torch.FloatTensor)
y = torch.unsqueeze(y, 1)
# 转numpy数组，手动划分训练集测试集
x = x.numpy()
x = np.insert(x, 0, xx, axis=1)
x_t = x[1500:2000, :]
x = x[0:1499, :]
# print(x)
y = y.numpy()
y_t = y[1500:2000, :]
y = y[0:1499, :]
#train_loader = torch.utils.data.DataLoader(train_set, batch_size = BATCH_SIZE, shuffle = True)

'''定义sigmoid函数'''
def sigmoid(x):
    return (1/ (1+ np.exp(-x )))

'''定义model函数'''
def model(x,theta):
    return sigmoid(np.dot(x,theta.T))

'''定义损失函数'''
def cost(x, y, theta):
    left = np.multiply(-y, np.log(model(x, theta)))
    right = np.multiply((1 - y), np.log(1 - model(x, theta)))
    num = len(x)
    return (np.sum(left - right) / num)

'''定义gradient函数'''
def gradient(x, y, theta):
    num = len(x)
    grad = np.zeros(theta.shape)
    # print("grad.shape=", grad.shape)
    # print("model.shape=", model(x,theta).shape)
    error = model(x, theta) - y
    error = error.ravel()
    # print("error.shape=", error.shape)
    for i in range(len(theta.ravel())):
        # print("x[:,i].shape=", x[:,i].shape)
        grad[0, i] = np.sum(np.multiply(error, x.T[i])) / num
    return grad

'''定义descent函数'''
def descent(X, y, theta, batchsize, alpha, epoch):
    costs = [cost(X, y, theta)]
    grad = np.zeros(theta.shape)
#    dataloader = torch.utils.data.DataLoader(data, batch_size=batchsize, shuffle = True)

    for j in range(epoch):
        # print("x shape = ", X.shape)
        # print("y shape = ", y.shape)
        # print("theta shape = ", theta.shape)
        # print("theta.ravel() shape = ", len(theta.ravel()))
        print("epoch: {0}".format(j))

        costs.append(cost(X, y, theta))
        grad = gradient(X, y, theta)
        theta = theta - alpha * grad
        #dataloader = torch.utils.data.DataLoader(data, batch_size=batchsize, shuffle=True)
    return theta, costs, grad

theta = np.zeros([1,3])
theta, costs, grad = descent(x, y, theta, BATCH_SIZE, 0.001, 20)

def predict(X, theta):
    return [1 if x >= 0.5 else 0 for x in model(X, theta)]

print("theta = ", theta)
preds = predict(x_t, theta)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(preds, y_t)]
accuracy = sum(map(int, correct))
accuracy = accuracy / len(correct)

print('acc = {0}'.format(accuracy))
