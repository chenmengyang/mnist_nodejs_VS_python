const tf = require('@tensorflow/tfjs');
const fs = require('fs');

// load the training set from file
const trainingSetImages  = fs.readFileSync('../data/mnist/train-images-idx3-ubyte');
_ = trainingSetImages.readUInt32BE(0)
const m_train_images = trainingSetImages.readUInt32BE(4)
const train_width = trainingSetImages.readUInt32BE(8)
const train_height = trainingSetImages.readUInt32BE(12)
const HEAD_TRAIN_SIZE = 16
let index1 = HEAD_TRAIN_SIZE
const IMAGE_SIZE = train_width * train_height // 728
train_images = []
while(index1< (m_train_images * IMAGE_SIZE)) {
    let imgBuf = []
    for (let i=0; i<IMAGE_SIZE; i++) {
        imgBuf.push((trainingSetImages.readUInt8(index1++)) * (1.0 / 255.0))
    }
    train_images.push(imgBuf)
}

const trainSetLabels  = fs.readFileSync('../data/mnist/train-labels-idx1-ubyte');
_ = trainSetLabels.readUInt32BE(0)
const m_train_labels = trainSetLabels.readUInt32BE(4)
const train_labels = [];
let indexl1 = 8;
while (indexl1 < trainSetLabels.byteLength) {
  const array = new Int32Array(1);
  for (let i = 0; i < 1; i++) {
    array[i] = trainSetLabels.readUInt8(indexl1++);
  }
  train_labels.push(array);
}

// load the test set from file
const testSetImages  = fs.readFileSync('../data/mnist/t10k-images-idx3-ubyte');
_ = testSetImages.readUInt32BE(0)
const m_test_images = testSetImages.readUInt32BE(4)
const test_width = testSetImages.readUInt32BE(8)
const test_height = testSetImages.readUInt32BE(12)
let index2 = 16
test_images = []
while(index2 < (m_test_images * IMAGE_SIZE)) {
    let imgBuf = []
    for (let i=0; i<IMAGE_SIZE; i++) {
        imgBuf.push((testSetImages.readUInt8(index2++)) * (1.0 / 255.0))
    }
    test_images.push(imgBuf)
}

const testSetLabels  = fs.readFileSync('../data/mnist/t10k-labels-idx1-ubyte');
_ = testSetLabels.readUInt32BE(0)
const m_test_labels = testSetLabels.readUInt32BE(4)
const test_labels = [];
let indexl2 = 8;
while (indexl2 < testSetLabels.byteLength) {
  const array = new Int32Array(1);
  for (let i = 0; i < 1; i++) {
    array[i] = testSetLabels.readUInt8(indexl2++);
  }
  test_labels.push(array);
}

// define the data class
class MnistData {
    
    constructor(batchSize) {
        this.batchSize = batchSize
        this.currentBatch = 0
    }

    reset() {
        this.currentBatch = 0
    }

    //
    nextBatch() {
        let trainX = train_images.slice(this.currentBatch * this.batchSize, (this.currentBatch+1) * this.batchSize)
        let trainY = train_labels.slice(this.currentBatch * this.batchSize, (this.currentBatch+1) * this.batchSize)

        this.currentBatch += 1;

        if (trainX && trainX.length>0) {
            return {
                train: {
                    // xs: tf.tensor2d(trainX, [this.batchSize, IMAGE_SIZE], 'float32').reshape([this.batchSize, train_width, train_height, 1]),
                    xs: tf.tensor2d(trainX).reshape([this.batchSize, train_width, train_height, 1]),
                    labels: tf.oneHot(tf.tensor1d(trainY, 'int32'), 10).toFloat(),
                },
            }
        } else {
            return false
        }
    }
}

module.exports.MnistData = MnistData
module.exports.TestSet = {
    xs: tf.tensor2d(test_images).reshape([m_test_images, train_width, train_height, 1]),
    labels: tf.oneHot(tf.tensor1d(test_labels, 'int32'), 10).toFloat(),
    size: m_test_images
}