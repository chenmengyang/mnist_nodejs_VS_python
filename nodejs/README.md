# mnist in nodejs

## time cost

54 seconds per epoch

## memory cost

## code

170 lines of codes in 3 files (mnist.js + mnistData.js + mnistModel.js)

## files description:
### mnistData.js
load train and test data&labels from files (please download files by yourself from MNIST website, after download please unzip and put all files to ./data/mnist under root dir)

you should notice the MnistData class feed the models by batches


### model.js
define 3 (convelotions) layers neural network, 

### mnist.js
the main program, including: train, test, run