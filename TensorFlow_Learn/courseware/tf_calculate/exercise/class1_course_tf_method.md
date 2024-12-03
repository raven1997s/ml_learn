# 使用tf模拟观察梯度下降的过程


```python
import tensorflow as tf

w = tf.Variable(tf.constant(5, dtype=tf.float32))
lr = 0.2
epoch = 40

for epoch in range(epoch):  # for epoch 定义顶层循环，表示对数据集循环epoch次，此例数据集数据仅有1个w,初始化时候constant赋值为5，循环40次迭代。
    with tf.GradientTape() as tape:  # with结构到grads框起了梯度的计算过程。
        loss = tf.square(w + 1)
    grads = tape.gradient(loss, w)  # .gradient函数告知谁对谁求导

    w.assign_sub(lr * grads)  # .assign_sub 对变量做自减 即：w -= lr*grads 即 w = w - lr*grads
    print("After %s epoch,w is %f,loss is %f" % (epoch, w.numpy(), loss))

# lr初始值：0.2   请自改学习率  0.001  0.999 看收敛过程
# 最终目的：找到 loss 最小 即 w = -1 的最优参数w

```

    After 0 epoch,w is 2.600000,loss is 36.000000
    After 1 epoch,w is 1.160000,loss is 12.959999
    After 2 epoch,w is 0.296000,loss is 4.665599
    After 3 epoch,w is -0.222400,loss is 1.679616
    After 4 epoch,w is -0.533440,loss is 0.604662
    After 5 epoch,w is -0.720064,loss is 0.217678
    After 6 epoch,w is -0.832038,loss is 0.078364
    After 7 epoch,w is -0.899223,loss is 0.028211
    After 8 epoch,w is -0.939534,loss is 0.010156
    After 9 epoch,w is -0.963720,loss is 0.003656
    After 10 epoch,w is -0.978232,loss is 0.001316
    After 11 epoch,w is -0.986939,loss is 0.000474
    After 12 epoch,w is -0.992164,loss is 0.000171
    After 13 epoch,w is -0.995298,loss is 0.000061
    After 14 epoch,w is -0.997179,loss is 0.000022
    After 15 epoch,w is -0.998307,loss is 0.000008
    After 16 epoch,w is -0.998984,loss is 0.000003
    After 17 epoch,w is -0.999391,loss is 0.000001
    After 18 epoch,w is -0.999634,loss is 0.000000
    After 19 epoch,w is -0.999781,loss is 0.000000
    After 20 epoch,w is -0.999868,loss is 0.000000
    After 21 epoch,w is -0.999921,loss is 0.000000
    After 22 epoch,w is -0.999953,loss is 0.000000
    After 23 epoch,w is -0.999972,loss is 0.000000
    After 24 epoch,w is -0.999983,loss is 0.000000
    After 25 epoch,w is -0.999990,loss is 0.000000
    After 26 epoch,w is -0.999994,loss is 0.000000
    After 27 epoch,w is -0.999996,loss is 0.000000
    After 28 epoch,w is -0.999998,loss is 0.000000
    After 29 epoch,w is -0.999999,loss is 0.000000
    After 30 epoch,w is -0.999999,loss is 0.000000
    After 31 epoch,w is -1.000000,loss is 0.000000
    After 32 epoch,w is -1.000000,loss is 0.000000
    After 33 epoch,w is -1.000000,loss is 0.000000
    After 34 epoch,w is -1.000000,loss is 0.000000
    After 35 epoch,w is -1.000000,loss is 0.000000
    After 36 epoch,w is -1.000000,loss is 0.000000
    After 37 epoch,w is -1.000000,loss is 0.000000
    After 38 epoch,w is -1.000000,loss is 0.000000
    After 39 epoch,w is -1.000000,loss is 0.000000


# 创建一个简单的TensorFlow


```python
# 创建一个TensorFlow
import tensorflow as tf

a = tf.constant(1, dtype=tf.float32)
b = tf.constant([1,2], dtype=tf.float32)
c = tf.constant([[1,1],[2,2],[3,3]], dtype=tf.float32)

print(a)
print(f'a = {a}')
print(f'a.dtype = {a.dtype}')
print(f'a.shape = {a.shape}')
print('=============================================')

print(b)
print(f'b = {b}')
print(f'b.dtype = {b.dtype}')
print(f'b.shape = {b.shape}')
print('=============================================')

print(c)
print(f'c = {c}')
print(f'c.dtype = {c.dtype}')
print(f'c.shape = {c.shape}')

```

    tf.Tensor(1.0, shape=(), dtype=float32)
    a = 1.0
    a.dtype = <dtype: 'float32'>
    a.shape = ()
    =============================================
    tf.Tensor([1. 2.], shape=(2,), dtype=float32)
    b = [1. 2.]
    b.dtype = <dtype: 'float32'>
    b.shape = (2,)
    =============================================
    tf.Tensor(
    [[1. 1.]
     [2. 2.]
     [3. 3.]], shape=(3, 2), dtype=float32)
    c = [[1. 1.]
     [2. 2.]
     [3. 3.]]
    c.dtype = <dtype: 'float32'>
    c.shape = (3, 2)


# 将numpy的数据类型转换为tensor数据类型


```python
# 将numpy的数据类型转换为tensor数据类型
import numpy as np

a = np.arange(0,5)
print(a)
b = tf.convert_to_tensor(a,dtype=tf.float32)
print(b)

```

    [0 1 2 3 4]
    tf.Tensor([0. 1. 2. 3. 4.], shape=(5,), dtype=float32)


# 使用TensorFlow创建不同纬度的的数组


```python
# 使用TensorFlow创建不同纬度的的数组

# param 数组的纬度  值全部为0
# 纬度
# 1维： 直接写个数
# 2维： 用(行，列)  输出[行，列]
# 3维： 用(行，列，层) 输出[行，列，层]
# n维： 用(行，列，层，...) 输出[行，列，层...]
a1 = tf.zeros(1)
a2 = tf.zeros((2,2))
a3 = tf.zeros((3,3,3))
# print('------------zeros-----------------------------')
# print(a1)
# print('=================================================')
# print(a2)
# print('=================================================')
# print(a3)

# print('-------------ones--------------------------------')
# # param 数组的纬度  值全部为1
# b1 = tf.ones(1)
# b2 = tf.ones((2,2))
# print(b1)
# print('=================================================')
# print(b2)

print('----------------fill-------------------------------')
# 创建全为指定值的数组
c1 = tf.fill(1,9)
c2 = tf.fill((2,2),8)
c3 = tf.fill((3,3,3),7)
print(c1)
print('=================================================')
print(c2)
print('=================================================')
print(c3)
```

    ----------------fill-------------------------------
    tf.Tensor([9], shape=(1,), dtype=int32)
    =================================================
    tf.Tensor(
    [[8 8]
     [8 8]], shape=(2, 2), dtype=int32)
    =================================================
    tf.Tensor(
    [[[7 7 7]
      [7 7 7]
      [7 7 7]]
    
     [[7 7 7]
      [7 7 7]
      [7 7 7]]
    
     [[7 7 7]
      [7 7 7]
      [7 7 7]]], shape=(3, 3, 3), dtype=int32)


# 使用TensorFlow 生成正太分布的随机数


```python
# 使用TensorFlow 生成正太分布的随机数 所有值都可能出现。
# parameters: shape = 形状, mean = 均值, stddev = 标准差 结果都在俩倍标准差内 ，数据都在均值附近。
a = tf.random.normal(shape=(2, 3), mean=1.0, stddev=1.0)
print(a)

# 使用TensorFlow 生成截断正太分布的随机数 舍弃了极端值，更平滑。
# parameters: shape = 形状, mean = 均值, stddev = 标准差
b = tf.random.truncated_normal(shape=(3, 3), mean=1.0, stddev=1.0)
print(b)

c = tf.random.normal(shape= (3,3,3), mean=0.0, stddev=1.0)
print(c)
```

    tf.Tensor(
    [[1.6755232 1.3096004 2.626256 ]
     [3.099818  1.5037482 1.2535437]], shape=(2, 3), dtype=float32)
    tf.Tensor(
    [[2.3492804  0.30397093 1.5604835 ]
     [1.3046471  1.1749543  1.5334079 ]
     [0.14099407 0.12230831 2.4702039 ]], shape=(3, 3), dtype=float32)
    tf.Tensor(
    [[[-1.1929271  -0.5396427  -0.51999027]
      [ 0.03832126 -0.9764158   2.1505182 ]
      [-0.20639803  0.00585017  0.69992405]]
    
     [[ 1.8584644  -0.9749002   1.1374246 ]
      [ 0.2115156  -0.63308686 -1.3756505 ]
      [ 0.7488925  -0.6113612   0.9755997 ]]
    
     [[ 1.2926517  -0.5451891   0.29370302]
      [-0.38894567  0.4151528  -0.44211608]
      [-0.4689476  -1.4655814  -0.5997934 ]]], shape=(3, 3, 3), dtype=float32)


# 使用tf生成均匀分布的随机数


```python
# 生成均匀分布的随机数 [max,min)
a = tf.random.uniform(shape=(2,2),minval=0,maxval=1)
print(a)

b = tf.random.uniform(shape=(3,3,3),minval=0,maxval=1)
print(b)
```

    tf.Tensor(
    [[0.8786297  0.40170407]
     [0.4544171  0.03021514]], shape=(2, 2), dtype=float32)
    tf.Tensor(
    [[[6.0782564e-01 6.6189969e-01 6.4920962e-01]
      [1.2656271e-01 4.3745995e-01 5.0298190e-01]
      [1.9666672e-01 7.8926504e-01 8.4545839e-01]]
    
     [[1.9708371e-01 4.2974949e-04 9.9448299e-01]
      [8.0319953e-01 5.4251420e-01 4.2206776e-01]
      [8.4026217e-01 2.5527239e-02 5.9680498e-01]]
    
     [[7.5571537e-03 2.4450099e-01 2.3040354e-01]
      [2.6115513e-01 7.6110709e-01 9.8826993e-01]
      [2.2702920e-01 2.7402782e-01 5.2210724e-01]]], shape=(3, 3, 3), dtype=float32)


# 强制TensorFlow 转换为该数据类型


```python
# 强制TensorFlow 转换为该数据类型

# define  constant tensor
a = tf.constant([1.0, 2.0,3.0], dtype=tf.float32)
print(a)

# cast the tensor to another data type
b = tf.cast(a, tf.int32)
print(b)

# get the max and min value of tensor
print(tf.reduce_max(b))
print(tf.reduce_min(b))

```

    tf.Tensor([1. 2. 3.], shape=(3,), dtype=float32)
    tf.Tensor([1 2 3], shape=(3,), dtype=int32)
    tf.Tensor(3, shape=(), dtype=int32)
    tf.Tensor(1, shape=(), dtype=int32)


# 理解axis参数


```python
# 理解axis参数
# axis = 0，沿着列进行操作 （经度，跨行，dwon）
# axis = 1，沿着行进行操作 （纬度，跨列，across）
# 不指定axis ，确认对整个矩阵进行操作

a = tf.fill((2,2),1)
print(f'a is : {a}')

# 求和
sum_all = tf.reduce_sum(a)
print(f'sum_all is : {sum_all}')

sum_down_all = tf.reduce_sum(a,axis=0)
print(f'sum_down_all is : {sum_down_all}')

sum_across_all = tf.reduce_sum(a,axis=1)
print(f'sum_across_all is : {sum_across_all}')


# 求平均值
mean_all = tf.reduce_mean(a)
print(f'mean_all is : {mean_all}')

mean_down_all = tf.reduce_mean(a,axis=0)
print(f'mean_down_all is : {mean_down_all}')

mean_across_all = tf.reduce_mean(a,axis=1)
print(f'mean_across_all is : {mean_across_all}')
```

    a is : [[1 1]
     [1 1]]
    sum_all is : 4
    sum_down_all is : [2 2]
    sum_across_all is : [2 2]
    mean_all is : 1
    mean_down_all is : [1 1]
    mean_across_all is : [1 1]


# Variable函数


```python
# tf.Variable 函数将变量标记为可训练，被标记的变量会在反向传播中计算梯度
# 在神经网络中，常用该函数来定义权重和偏置

w = tf.Variable(tf.random.normal([3, 3], mean = 0, stddev = 1))
print(f'w: {w}')
```

    w: <tf.Variable 'Variable:0' shape=(3, 3) dtype=float32, numpy=
    array([[ 1.73464   ,  0.935121  , -1.7392381 ],
           [-0.44500777,  1.32529   ,  0.39678475],
           [ 1.241507  , -0.7415922 , -1.8283793 ]], dtype=float32)>


# TensorFlow中的数学计算


```python
# 四则运算 add subtract multiply divide
# 平方、次方、开方 square power sqrt
# 矩阵乘 muamul

# 只有纬度相等的矩阵才能进行乘法运算
# 注意：tensorflow 四则运算时，俩个矩阵的数据类型要相同
# 四则运算时，如果形状不同，会通过广播机制调整形状

import tensorflow as tf
a = tf.ones([1,2])
b = tf.fill([1,2], 2.)
print(f'\na = {a}')
print(f'\nb = {b}')

print(f'\na + b = {tf.add(a,b)}')
print(f'\na * b = {tf.multiply(a,b)}')
print(f'\na / b = {tf.divide(a,b)}')
print(f'\na - b = {tf.subtract(a,b)}')

# 平方
c = tf.square(b)
print(f'\nc = {c}')

# 次方 此时为3次方
print(f'\nc ** 3 = {tf.pow(c,3)}')

# 开方
d = tf.sqrt(c)
print(f'\nd = {d}')


# 矩阵乘
# 注意：tensorflow 在进行矩阵乘时，要求两个矩阵的纬度必须满足条件：
# 1. 第一个矩阵的列数 = 第二个矩阵的行数
# 2. 两个矩阵的数据类型必须相同
e1 = tf.ones([3,2])
e2 = tf.fill([2,3], 3.)
e = tf.matmul(e1,e2)
print(f'\ne1 = {e1}')
print(f'\ne2 = {e2}')
print(f'\ne = {e}')
# e3 = tf.matmul(e2,e1)
# print(f'\ne3 = {e3}')
```

    
    a = [[1. 1.]]
    
    b = [[2. 2.]]
    
    a + b = [[3. 3.]]
    
    a * b = [[2. 2.]]
    
    a / b = [[0.5 0.5]]
    
    a - b = [[-1. -1.]]
    
    c = [[4. 4.]]
    
    c ** 3 = [[64. 64.]]
    
    d = [[2. 2.]]
    
    e1 = [[1. 1.]
     [1. 1.]
     [1. 1.]]
    
    e2 = [[3. 3. 3.]
     [3. 3. 3.]]
    
    e = [[6. 6. 6.]
     [6. 6. 6.]
     [6. 6. 6.]]


multiply 和 matmul 有什么区别？

| 特性           | `tf.multiply`                        | `tf.matmul`                               |
|----------------|--------------------------------------|-------------------------------------------|
| 运算类型        | 逐元素相乘                          | 矩阵乘法                                  |
| 广播支持        | 支持                                 | 不支持（但支持高维批量矩阵乘法）          |
| 输入形状要求    | 形状相同或可广播                    | 矩阵乘法规则（前者的列数等于后者的行数）  |
| 常用场景        | 逐元素计算，点积                    | 线性代数运算，深度学习模型的权重操作      |

# 切片传入张量的第一维度，生成输入特征/标签 构建数据集


```python
# 切片传入张量的第一维度，生成输入特征/标签 构建数据集
# data = tf.data.Dataset.from_tensor_slices((x, y))

feauture = tf.constant([1,2,3,4,5])
label = tf.constant([0,1,0,1,0])
result = tf.data.Dataset.from_tensor_slices((feauture, label))
for i in result:
    print(i)
```

    (<tf.Tensor: shape=(), dtype=int32, numpy=1>, <tf.Tensor: shape=(), dtype=int32, numpy=0>)
    (<tf.Tensor: shape=(), dtype=int32, numpy=2>, <tf.Tensor: shape=(), dtype=int32, numpy=1>)
    (<tf.Tensor: shape=(), dtype=int32, numpy=3>, <tf.Tensor: shape=(), dtype=int32, numpy=0>)
    (<tf.Tensor: shape=(), dtype=int32, numpy=4>, <tf.Tensor: shape=(), dtype=int32, numpy=1>)
    (<tf.Tensor: shape=(), dtype=int32, numpy=5>, <tf.Tensor: shape=(), dtype=int32, numpy=0>)


    2024-11-28 17:15:38.888271: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_1' with dtype int32 and shape [5]
    	 [[{{node Placeholder/_1}}]]


# 使用GradientTape 实现某个函数对指定的变量求导


```python
import tensorflow as tf

# tf.GradientTape()  在with结构中，使用tf.GradientTape()实现某个函数对制定参数的求导运算
# with tf.GradientTape() as tape:
# grad = tape.gradient(y, x)

with tf.GradientTape() as tape:
    # 定义一个变量 w，初始值为常量 3.0。
	# 使用 tf.Variable 表明 w 是一个可训练变量（即可以对它求导和更新）。
    w = tf.Variable(tf.constant(3.0))
    # 定义损失函数 loss，这里是  w^2 
    loss = tf.pow(w, 2)
# 用 tape.gradient(target, sources) 计算目标值 loss 对源变量 w 的梯度
grad = tape.gradient(loss, w)
print(grad)
```

    tf.Tensor(6.0, shape=(), dtype=float32)



```python
class MyContext:
    def __enter__(self):
        print("进入上下文")
        return "资源"

    def __exit__(self, exc_type, exc_value, traceback):
        print("退出上下文")
        return True  # 如果返回 True，抑制异常传播

with MyContext() as resource:
    print("使用", resource)
    raise ValueError("抛出异常")
# 输出:
# 进入上下文
# 使用 资源
# 退出上下文
```

    进入上下文
    使用 资源
    退出上下文



```python
# enumerate 函数 用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
array = ['a', 'b', 'c']
for i, value in enumerate(array):
    print(i, value)
```

    0 a
    1 b
    2 c


# tf.one_hot 独热编码 


```python
# tf.one_hot 独热编码 在分类问题中，我们通常会使用独热编码来表示一个类别。
# 将待转换数据 转换为 one_hot形式的数据输出
# tf.one_hot(data, depth= 几分类)
classes =3
labels = tf.constant([1, 2, 0])
one_hot = tf.one_hot(labels, classes)
print(one_hot)

```

    tf.Tensor(
    [[0. 1. 0.]
     [0. 0. 1.]
     [1. 0. 0.]], shape=(3, 3), dtype=float32)


# 使用tf.nn.softmax(x) 使结果符合概率分布


```python
# 使用tf.nn.softmax(x) 使结果符合概率分布
# 当n分类的n个输出（y0，y1，y2，y3，y4） 通过tf.nn.softmax(x)函数后符合概率分布 y0+y1+y2+y3+y4=1

y = tf.constant([1.2213, 2.33312, 3.332])
y_pro = tf.nn.softmax(y)
print(y_pro)
```

    tf.Tensor([0.08134112 0.24726778 0.67139107], shape=(3,), dtype=float32)


# 使用assign_sub 函数更新参数的值并返回


```python
# 使用assign_sub 函数更新参数的值并返回
# 调用assign_sub时，需要先用tf.Variable()函数创建变量，定义变量为可训练的

w = tf.Variable(10, name='weight')
w.assign_sub(1)
print(w)
```

    <tf.Variable 'weight:0' shape=() dtype=int32, numpy=9>


# 使用tf.argmax()函数返回张量沿指定轴的最大值的索引


```python
import numpy as np
array = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
print(array)
# 返回每一列（经度）最大值的索引
print(tf.argmax(array,axis=0))
# 返回每一行（纬度）最大值的索引
print(tf.argmax(array,axis=1))

```

    [[ 1  2  3  4]
     [ 5  6  7  8]
     [ 9 10 11 12]]
    tf.Tensor([2 2 2 2], shape=(4,), dtype=int64)
    tf.Tensor([3 3 3], shape=(3,), dtype=int64)

