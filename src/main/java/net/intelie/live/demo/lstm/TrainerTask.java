package net.intelie.live.demo.lstm;

import com.google.common.primitives.Doubles;
import net.intelie.live.*;
import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Semaphore;

public class TrainerTask implements Runnable {
    private static final Logger LOGGER = LoggerFactory.getLogger(TrainerTask.class);

    private final Live live;
    private final String qualifier;
    private final String query;
    private final String trainSpan;
    private final String testSpan;
    private final double batchFactor;
    private final int epochs;
    private double completion;

    public TrainerTask(Live live, String qualifier, String query, String trainSpan, String testSpan, double batchFactor, int epochs) {
        this.live = live;
        this.qualifier = qualifier;
        this.query = query;
        this.trainSpan = trainSpan;
        this.testSpan = testSpan;
        this.batchFactor = batchFactor;
        this.epochs = epochs;
    }

    public double getCompletion() {
        return completion;
    }

    public TrainedData train(Data train, Data test, int miniBatchSize, int epochs) throws Exception {
        while (live.engine().getStorageProviders().size() == 0) {
            LOGGER.info("Waiting for some storage provider to become active");
            Thread.sleep(1000);
        }

        LOGGER.info("Training {} {} {}", miniBatchSize, train.size(), test.size());
        // ----- Load the training data -----

        DataSet trainData = train.toDataSet();
        DataSet testData = test.toDataSet();

        LOGGER.info("Created dataset {} {} {}", miniBatchSize, train.size(), test.size());

        LOGGER.info("Train {}", trainData.getFeatureMatrix());
        LOGGER.info("Train labels {}", trainData.getLabels());
        LOGGER.info("Test {}", testData.getFeatureMatrix());
        LOGGER.info("Test Labels {}", testData.getLabels());

        //Normalize data, including labels (fitLabel=true)
        NormalizerMinMaxScaler normalizer = new NormalizerMinMaxScaler(0, 1);
        normalizer.fitLabel(true);
        normalizer.fit(trainData);              //Collect training data statistics
        normalizer.transform(trainData);
        normalizer.transform(testData);

        LOGGER.info("Normalizing {} {} {}", miniBatchSize, train.size(), test.size());
        LOGGER.info("Train {}", trainData.getFeatureMatrix());
        LOGGER.info("Train labels {}", trainData.getLabels());
        LOGGER.info("Test {}", testData.getFeatureMatrix());
        LOGGER.info("Test Labels {}", testData.getLabels());

        // ----- Configure the network -----
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(140)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(1)
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .learningRate(0.000015)
                .list()
                .layer(0, new GravesLSTM.Builder().activation(Activation.TANH).nIn(1).nOut(10)
                        .build())
                .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.IDENTITY).nIn(10).nOut(1).build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        net.setListeners(new ScoreIterationListener(20));

        // ----- Train the network, evaluating the test set performance at each epoch -----
        LOGGER.info("Training start {} {} {} epochs={}", miniBatchSize, train.size(), test.size(), epochs);

        completion = 0;
        for (int i = 0; i < epochs; i++) {
            if (Thread.interrupted())
                throw new InterruptedException("The processing was interrupted");

            completion = i / (double) epochs;

            net.fit(trainData);
            LOGGER.info("Epoch " + i + " complete. Time series evaluation:");

            //Run regression evaluation on our single column input
            RegressionEvaluation evaluation = new RegressionEvaluation(1);
            INDArray features = testData.getFeatureMatrix();

            INDArray lables = testData.getLabels();
            INDArray predicted = net.output(features, false);

            evaluation.evalTimeSeries(lables, predicted);

            //Just do sout here since the logger will shift the shift the columns of the stats
            System.out.println(evaluation.stats());
        }
        completion = 1;

        net.rnnTimeStep(trainData.getFeatureMatrix());
        net.rnnTimeStep(testData.getFeatureMatrix());

        normalizer.revert(trainData);
        normalizer.revert(testData);

        LOGGER.info("Train {}", trainData.getFeatureMatrix());
        LOGGER.info("Train labels {}", trainData.getLabels());
        LOGGER.info("Test {}", testData.getFeatureMatrix());
        LOGGER.info("Test Labels {}", testData.getLabels());

        return new TrainedData(net, normalizer);
    }

    public void run() {
        try {
            Data trainData = getData(live, trainSpan);
            Data testData = getData(live, testSpan);

            TrainedData trained = train(trainData, testData, (int) (trainData.size() * batchFactor), epochs);

            live.pipes().addInstanceModule(new Predictor(trained, qualifier));
        } catch (Throwable e) {
            LOGGER.error("Error", e);
        }
    }

    private Data getData(Live live, String span) throws Exception {
        Semaphore semaphore = new Semaphore(0);

        List<Double> first = new ArrayList<>(), second = new ArrayList<>();
        live.engine().runQueries(new Query(query).span(span).listenWith(
                new QueryListener.Empty() {
                    @Override
                    public void onEvent(QueryEvent event, boolean history) throws Exception {
                        event.stream().forEach(x -> {
                            if (x.get("x") instanceof Double && x.get("y") instanceof Double) {
                                first.add((Double) x.get("x"));
                                second.add((Double) x.get("y"));
                            }
                        });
                    }

                    @Override
                    public void onDestroy(DestroyInfo event) throws Exception {
                        semaphore.release();
                    }
                }
        ));
        semaphore.acquireUninterruptibly();
        return new Data(Doubles.toArray(first), Doubles.toArray(second));
    }
}
