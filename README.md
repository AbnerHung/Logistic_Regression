# 从0开始的逻辑回归python实现
## 使用pytorch生成数据示例：
```python
    n_data = torch.ones(1000, 2)
    xx = torch.ones(2000)
    # print(xx)
    x0 = torch.normal(2 * n_data, 1)  # 生成均值为2.标准差为1的随机数组成的矩阵 shape=(100, 2)
    y0 = torch.zeros(1000)
    x1 = torch.normal(-2 * n_data, 1)  # 生成均值为-2.标准差为1的随机数组成的矩阵 shape=(100, 2)
    y1 = torch.ones(1000)


    #合并数据x,y
    x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
    y = torch.cat((y0, y1), 0).type(torch.FloatTensor)
    y = torch.unsqueeze(y, 1) # 使y的shape=（2000，2）
    x = x.numpy() # 转numpy
    x = np.insert(x, 0, xx, axis=1) # 在所有行第一列前插入1
    x_t = x[1500:2000, :] # 手动划分测试集，下同
    x = x[0:1499, :]
    y = y.numpy()
    y_t = y[1500:2000, :]
    y = y[0:1499, :]
```
## 定义sigmoid函数
![sigmoid](https://cdn.nlark.com/yuque/0/2020/png/514680/1605181164656-bbbfa3ea-0c5c-4036-9c4e-3cdd732d3391.png)
