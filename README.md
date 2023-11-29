# Example of Tensorflow Operation in C++

This repository contains an example of a simple Tensorflow operation and its gradient both implemented in C++, as described in [this article](http://davidstutz.de/implementing-tensorflow-operations-in-c-including-gradients/).

Gonen Raveh Says: I have tailored this codebase to Tensorflow2 and added a sample training procedure with this CustomOP as one of the layers. The network is very simple:
```
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
N=121; M=64;BATCH=N
x = Input(shape=(1), name='x')  # will be at training [N,1]
w = Input(shape=(N), name='w')  # will be at training [M,N]
y = inner_product_module.inner_product(x,w,name='inner1')  # [M,1]
model = Model([x,w], y)
```

## The Training procedure
```
import keras
model, N, M = self.genModel()
# train...
epochs = 5
optimizer = keras.optimizers.Adam(learning_rate=1e-3)
loss_fn = keras.losses.MeanSquaredError()
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))
    with tf.GradientTape() as tape:
        x = np.random.randint(10, size=(N,1))  # [M,N]*[N,1] = [M,1]
        w = np.random.randint(10, size=(M,N))
        y_true = np.random.randint(10, size=(M,1))  # not really "training". just an example
        y_pred = model([x, w])
        loss = loss_fn(y_true, y_pred)
        print(f'loss={loss}')
    #
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```
## The 8'th Unit Test Output should be as follows
```
Model: "model"
__________________________________________________________________________________________________
 Layer (type)                Output Shape                 Param #   Connected to                  
==================================================================================================
 x (InputLayer)              [(None, 1)]                  0         []                            
                                                                                                  
 w (InputLayer)              [(None, 121)]                0         []                            
                                                                                                  
 tf.inner_product (TFOpLamb  (None, 1)                    0         ['x[0][0]',                   
 da)                                                                 'w[0][0]']                   
                                                                                                  
==================================================================================================
Total params: 0 (0.00 Byte)
Trainable params: 0 (0.00 Byte)
Non-trainable params: 0 (0.00 Byte)
__________________________________________________________________________________________________
None

Start of epoch 0
loss=5896649.5

Start of epoch 1
loss=6279839.0

Start of epoch 2
loss=6196921.0

Start of epoch 3
loss=5386027.0

Start of epoch 4
loss=6036963.0
.
```
## Building

The operation is built using [CMake](https://cmake.org/) and requires an appropriate version of Tensorflow to be installed. In order to get the necessary include directories containing the Tensorflow header files, the following trick is used (also see the [Tensorflow documentation](https://www.tensorflow.org/how_tos/adding_an_op/)):

    import tensorflow
    print(tensorflow.sysconfig.get_include())

In the `CMakeLists.txt` this is used as follows:

    execute_process(COMMAND python3 -c "import tensorflow; print(tensorflow.sysconfig.get_include())" OUTPUT_VARIABLE Tensorflow_INCLUDE_DIRS)

The remaining contents are pretty standard. Building is now done using:

    $ mkdir build
    $ cd build
    $ cmake ..
    $ make
    Scanning dependencies of target inner_product
    [ 50%] Building CXX object CMakeFiles/inner_product.dir/inner_product.cc.o
    Linking CXX shared library libinner_product.so
    [ 50%] Built target inner_product
    Scanning dependencies of target inner_product_grad
    [100%] Building CXX object CMakeFiles/inner_product_grad.dir/inner_product_grad.cc.o
    Linking CXX shared library libinner_product_grad.so
    [100%] Built target inner_product_grad

`libinner_product.so` and `libinner_product_grad.so` can be found in `build` and need to be included in order to load the module in Python:

    import tensorflow as tf
    inner_product_module = tf.load_op_library('build/libinner_product.so')

See `inner_product_tests.py` for usage examples.

## License

Copyright (c) 2016 David Stutz

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
