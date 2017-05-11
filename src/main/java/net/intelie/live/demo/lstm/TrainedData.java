package net.intelie.live.demo.lstm;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;

public class TrainedData {
    private final MultiLayerNetwork net;
    private final NormalizerMinMaxScaler normalizer;

    public TrainedData(MultiLayerNetwork net, NormalizerMinMaxScaler normalizer) {
        this.net = net;
        this.normalizer = normalizer;
    }

    public double predict(double x) {
        NDArray input = new NDArray(new double[][]{new double[]{x}});
        normalizer.transform(input);
        INDArray array = net.rnnTimeStep(input);
        normalizer.revertLabels(array);
        return array.getDouble(0);
    }
}
