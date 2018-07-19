const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node'); 
const model = require("./mnistModel");
const MnistData = require("./mnistData").MnistData;
const TestSet = require("./mnistData").TestSet;
const BATCH_SIZE = 1000;
const EPOCH = 5;

const train = async () => {
    while(true) {
        const batchData = data.nextBatch()
        if (!batchData) {
            data.reset();
            break;
        } else {
            const history = await model.fit(
                batchData.train.xs,
                batchData.train.labels,
                {
                    batchSize: BATCH_SIZE,
                    shuffle: true
                }
            );
            const loss = history.history.loss[0];
            const accuracy = history.history.acc[0];
            // ... plotting code ...
            console.log(`loss is ${loss} and accuracy is ${accuracy}`)
        }
    }
}

const test = () => {
    output = model.predict(TestSet.xs)
    const predictions = output.argMax(1).dataSync();
    const labels = TestSet.labels.argMax(1).dataSync();
    let correct = 0
    for (let i=0; i<TestSet.size; i++) {
        if (predictions[i] === labels[i]) {
            correct++
        }
    }
    const accuracy = ((correct / TestSet.size) * 100).toFixed(2);
    console.log(`testset accuracy is ${accuracy}, correct is ${correct} with len is ${predictions.length} and size is ${TestSet.size} with len is ${labels.length}`)
}

let data = new MnistData(BATCH_SIZE);
let epochTimes = [];

const run = async () => {
    for(let i=0; i < EPOCH; i++) {
        const t1 = new Date();
        await train();
        const t2 = new Date();
        let t = (t2-t1)/1000;
        epochTimes.push(t);
        console.log(`time of this train is ${t} seconds`);
        test();
    }
    avgEpochTime = epochTimes.reduce((a=0, v) => a+v) / epochTimes.length
    console.log(`average time for training all 60000 images (1 epoch) is ${avgEpochTime} seconds`)
}

run()