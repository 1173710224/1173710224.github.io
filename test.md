# 回调函数callbacks

### 前言

回调函数是在训练阶段可以执行各种活动的对象，训练阶段比如：每一个$epoch$的开头或者结尾，每一个$batch$之前或者之后等。接下来的讲解顺序大致就是：API的调用方法，十一个API的分析，以及自定义回调函数的例子。

使用回调函数，我们可以做到一下几件事情：

- 每几个$batch$之后，将$metrics$信息输出到日志文件中从而对训练过程进行监控
- 定期将当前的模型保存到磁盘中
- 执行$early\ stopping$
- 训练期间查看模型的内部状态和统计信息
- $\dots\ \dots$

### API调用方法

一般是在$fit()$函数中调用，使用函数的$callbacks$参数进行传递，可以传递一个回调函数的对象，也可以是一个回调函数的列表，一个简单的例子如下所示，具体的每个回调函数都会在之后讲解。

```python
my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2),
    tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
]
model.fit(dataset, epochs=10, callbacks=my_callbacks)
```

### API

#### Base Callback class

所有回调函数的抽象基类，如果是自定义的话，应该继承这个类，但是在API中没有直接的用处。该类有两个参数，一个是$params$，包含了$batch size,\ number\ of\ epochs$等，另一个是$model$。

```python
tf.keras.callbacks.Callback()
```

#### Model Checkpoint

以一定的频率保存模型或者模型的参数，这些信息将会被保存在一个$check\ point$文件中。被保存的模型后面可以被读取出来，继续进行训练。

更加具体的，如下所示：

- 是否只保存到目前为止性能最好的一组参数，或者忽视性能，每个$epoch$结束的时候都保存一下
- 在上一个功能中涉及到“性能最好”这一说法，因此就要对这个“最好”，进行定义，可以选择一个指标的最大或者最小作为最好
- 指定保存的频率，目前支持的频率为每个$epoch$结束之后或者在固定的训练批次之后进行保存
- 指定只保存模型的参数或者整个模型都保存下来

```python
tf.keras.callbacks.ModelCheckpoint(
    filepath,
    monitor="val_loss",
    verbose=0,
    save_best_only=False,
    save_weights_only=False,
    mode="auto",
    save_freq="epoch",
    **kwargs
)
```

各个参数的含义如下所示：

- $filepath$：模型存储的路径，同时可以包含一些格式化选项，比如$weights.{epoch:02d}-{val_loss:.2f}.hdf5$，对于这样的一个路径名，$02d,.2f$将会被替换为相应的指标值
- $monitor$：监控指标，决定了”最好“

- $verbose$：设置运行时输出信息的详细程度，0或者1
- $save\_best\_only$：布尔变量，如果设置为真，则最终结果中只保存效果最好的一组参数
- $mode$：字符串，$auto,min,max$中的一个，如果是监控的指标是正确率的话，一般是最大，损失函数的话一般是最小化，不过这个一般用$auto$，程序可以自己判断
- $save\_weights\_only$：布尔变量，字面意思
- $save\_freq$：$'epoch'$或者一个整数，前者对应每个$epoch$保存一次，后者对应若干个$batch$之后保存

在这里给出一个简单的例子

```python
EPOCHS = 10
checkpoint_filepath = '/tmp/checkpoint'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_acc',
    mode='max',
    save_best_only=True)

# Model weights are saved at the end of every epoch, if it's the best seen
# so far.
model.fit(epochs=EPOCHS, callbacks=[model_checkpoint_callback])

# The model weights (that are considered the best) are loaded into the model.
model.load_weights(checkpoint_filepath)
```

#### Tensor Board

对训练过程中的数据或者结果进行可视化，具体的如下所示：

- 指标摘要图
- 训练图可视化
- 激活直方图
- 采样分析

关于tensorboard，在后面会进行更加详细的分析。

#### Early Stopping

当某个指标已经停止优化时，不过这里用到的指标必须是调用compile函数时必须包含的指标，对于一些没有默认包含的指标，一定要用metrics参数指明。

```python
tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=0,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=False,
)
```

- $monitor$：监控的指标
- $min\_delta$：认为在某一指标上提升多少才算一次优化
- $patience$：连续多少次没有优化之后就停止
- $verbose$：同上
- $mode$：同上
- $baseline$：阈值，如果结果没有达到这个值，即使$patience$到了也不停
- $restore\_best\_weights$：是否保存最优结果，与上面的功能有些重叠

#### Learning Rate Scheduler

可以在训练的过程中使用变化的学习率

```python
tf.keras.callbacks.LearningRateScheduler(schedule, verbose=0)
```

这里的第一个参数是一个函数，函数的传入参数是$epoch,lr$，其中$epoch$是当前的训练轮数，$lr$是当前的学习率，返回一个新的学习率，可以自定义函数，根据当前的状况，决定学习率，如果想要实现一个根据loss的变化确定的学习率，也可以设置一个全局变量来解决这个问题。

```python
def scheduler(epoch, lr):
  if epoch < 10:
    return lr
  else:
    return lr * tf.math.exp(-0.1)
model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
model.compile(tf.keras.optimizers.SGD(), loss='mse')
callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
history = model.fit(np.arange(100).reshape(5, 20), np.zeros(5),
                    epochs=15, callbacks=[callback], verbose=0)
```

