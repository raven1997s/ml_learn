# tf.where (boolean_tensor, x, y) 条件语句真返回x 假返回y


```python
# tf.where (boolean_tensor, x, y) 条件语句真返回x，假返回y
import tensorflow as tf
a = tf.constant([1, 2, 3,1,1])
b = tf.constant([0,1,3,4,5])

# tf.greater(a,b) 返回a>b的布尔值 a>b为真返回true，假返回false
c = tf.where(tf.greater(a, b), a, b)
print(c)

```

    tf.Tensor([1 2 3 4 5], shape=(5,), dtype=int32)


# np.random.RandomState.rand() 返回一个[0,1)之间的随机数 


```python

```


```python
# np.random.RandomState.rand(纬度)
import numpy as np
rdm = np.random.RandomState(seed = 1) # seed = 常数时，每次运行结果不变
# 返回一个随机数
a = rdm.rand()

# 返回一个2行3列的随机数
b = rdm.rand(2,3)
print(f'a is {a}')
print(f'b is \n{b}')
print(f'b shape is {b.shape}')



```

    a is 0.417022004702574
    b is 
    [[7.20324493e-01 1.14374817e-04 3.02332573e-01]
     [1.46755891e-01 9.23385948e-02 1.86260211e-01]]
    b shape is (2, 3)


# np.vstack() 将多个数组按垂直方向叠加


```python
# np.vstack() 函数 将多个数组按垂直方向堆叠起来

a = np.array([1,2])
b = np.array([3,4])
c = np.array([5,6])
d = np.vstack((a,b,c))
print(f'd is \n{d}')
print(f'd shape is {d.shape}')

```

    d is 
    [[1 2]
     [3 4]
     [5 6]]
    d shape is (3, 2)


#  生成网格数据
 - np.mgird[]    
 - np.ravel()     
 - np.c_[]


```python
# np.mgird[] 左闭右开 [1,3)
# np.mgird[起始值1:结束值1:步长1,起始值2:结束值2:步长2 ..]

a = np.mgrid[1:3:1]
b = np.mgrid[2:4:0.5]
print(f'a is \n {a}' )
print(f'b is \n {b}')

x,y = np.mgrid[1:3:1, 2:4:0.5]
print(f'x is \n {x}')
print(f'y is \n {y}')


# np.ravel() 将多维数组降为一维
x1 = x.ravel()
print(f'x1 is \n {x1}')
y1 = y.ravel()
print(f'y1 is \n {y1}')

# np.c_[] 使返回的间隔数值点配对
# 将一维数组转置成多维数组
grid = np.c_[x1, y1]
print(f'grid is \n {grid}')


# 总结
#	1.	np.mgrid 用于生成多维网格点。
#	2.	np.ravel 将多维数组展平成一维。
#	3.	np.c_[] 将一维数组合并为二维配对形式（坐标点）。
# 你最终通过这些步骤，将 (1:3) 和 (2:4) 的网格点展开为坐标对：

```

    a is 
     [1 2]
    b is 
     [2.  2.5 3.  3.5]
    x is 
     [[1. 1. 1. 1.]
     [2. 2. 2. 2.]]
    y is 
     [[2.  2.5 3.  3.5]
     [2.  2.5 3.  3.5]]
    x1 is 
     [1. 1. 1. 1. 2. 2. 2. 2.]
    y1 is 
     [2.  2.5 3.  3.5 2.  2.5 3.  3.5]
    grid is 
     [[1.  2. ]
     [1.  2.5]
     [1.  3. ]
     [1.  3.5]
     [2.  2. ]
     [2.  2.5]
     [2.  3. ]
     [2.  3.5]]


 `np.mgrid` 的行为取决于具体的调用方式：

1. **一维情况** (`np.mgrid[start:stop:step]`)： 
   - 它会生成一个一维数组，其中包含从 `start` 到（但不包括）`stop`，步长为 `step` 的值。
   - 例如，`np.mgrid[1:3:1]` 生成的是 `[1, 2]`。

2. **多维情况** (`np.mgrid[start1:stop1:step1, start2:stop2:step2]`)：
   - 它会生成**网格数据**，用来表示笛卡尔坐标系中的点。每个维度上的范围用一个切片表示。
   - 例如，`np.mgrid[1:3:1, 2:4:0.5]` 会生成两个 2D 数组：
     - `x` 表示所有点在第一个维度上的坐标。
     - `y` 表示所有点在第二个维度上的坐标。

### 对比 `a` 和 `x`：
- `a` 是一维数组：`np.mgrid[1:3:1]` 只沿一个维度生成数值。
  ```python
  a = [1, 2]
  ```

- `x` 是二维网格的结果：`np.mgrid[1:3:1, 2:4:0.5]` 的第一部分。
  - `x` 会重复每一行，让数组的形状与 `y` 保持一致。
  - 它的形状为 `(len(1:3:1), len(2:4:0.5))`，即 `(2, 4)`。

具体到你的代码，`x` 和 `a` 的值不同是因为 `x` 包含多维网格的坐标，而 `a` 是简单的一维范围值。

### 代码输出说明：
```python
a = np.mgrid[1:3:1]
# a: [1 2]

b = np.mgrid[2:4:0.5]
# b: [2.  2.5 3.  3.5]

x, y = np.mgrid[1:3:1, 2:4:0.5]
# x:
# [[1 1 1 1]
#  [2 2 2 2]]
# y:
# [[2.  2.5 3.  3.5]
#  [2.  2.5 3.  3.5]]
```

- `x` 表示每一行的值对应 `1:3:1`，每一列重复这些值以形成网格。
- `y` 表示每一列的值对应 `2:4:0.5`，每一行重复这些值以形成网格。

这样，`x` 和 `y` 共同构成了笛卡尔网格中的所有坐标点。
