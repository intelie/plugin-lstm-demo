package net.intelie.live.demo.lstm;

import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

public class Data {
    private final double[] first;
    private final double[] second;

    public Data(double[] first, double[] second) {
        this.first = first;
        this.second = second;
    }

    public int size() {
        return first.length;
    }

    public DataSet toDataSet() {
        return new DataSet(new NDArray(first, new int[]{1, 1, first.length}, Nd4j.order()),
                new NDArray(second, new int[]{1, 1, first.length}, Nd4j.order()));
    }
}
