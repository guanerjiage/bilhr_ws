#!/usr/bin/env python
# Author: Erjiage GUAN, Zhifan NI, Lei WANG, Zhaoxiong WEI
import numpy as np
import matplotlib.pyplot as plt
from fnn.fnn import FNN

lr = 0.1
nn = FNN(discrete=False, learning_rate=lr, batch_size=10)
nn.add_layer(3, 20, nn.ReLU, nn.RandnInitializer, nn.ZeroInitializer, 2, 0.001, "hidden1")
# nn.add_layer(20, 12, nn.Sigmoid, nn.RandnInitializer, nn.ZeroInitializer, 2, 0.001, "hidden2")
nn.add_layer(20, 1, nn.NoActivation, nn.RandnInitializer, nn.ZeroInitializer, 2, 0.001, "output")
print(nn)


t = np.linspace(-40, 40, 5000)
s = np.sin(1/2 * t)
v = 1/2 * np.cos(1/2 * t)
a = -1/4 * np.sin(1/2 * t)

train_x_set = np.array([s[0:4800], v[0:4800], a[0:4800]])
train_y_set = np.array([s[50:4850]])
test_x_set = np.array([s[0:4800:50], v[0:4800:50], a[0:4800:50]])
test_y_set = np.array([s[50:4850:50]])
print(train_x_set.shape)
print(train_y_set.shape)
print(test_x_set.shape)
print(test_y_set.shape)

loss_trace = []
for i in range(1000):
    # if i > 30000:
    #     lr *= 0.999
    # nn.lr = lr
    if i % 100 == 0:
        print(i)
    loss_trace.append(nn.train_one_step(train_x_set[:, i].reshape((-1, 1)), train_y_set[:, i].reshape((-1, 1))))
    

y_pred = nn.predict(test_x_set)
# y_pred1 = y_pred[0, :]
# y_pred2 = y_pred[1, :]
# y1 = test_y_set[0, :]
# y2 = test_y_set[1, :]

plt.figure()
plt.plot(y_pred.flatten(), label="pre1")
# plt.plot(y_pred2, label="pre2")
plt.plot(test_y_set.flatten(), label="real1")
# plt.plot(y2, label="real2")
plt.legend()

plt.figure()
plt.plot(loss_trace)

plt.show()

