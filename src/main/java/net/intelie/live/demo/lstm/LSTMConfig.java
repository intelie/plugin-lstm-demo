package net.intelie.live.demo.lstm;

import net.intelie.live.*;

import java.util.Set;
import java.util.concurrent.ExecutorService;

public class LSTMConfig implements ExtensionConfig {
    private String query;
    private String trainSpan;
    private String testSpan;
    private double batchSizeFactor = 0.2;
    private int epochs = 300;
    private int neuronsHiddenLayer = 6;
    private double learningRate = 0.00015;

    @Override
    public String summarize() {
        return query;
    }

    @Override
    public Set<ExtensionRole> roles() {
        return ExtensionRole.start().ok();
    }

    @Override
    public ValidationBuilder validate(ValidationBuilder builder) {
        return builder
                .requiredValue(query, "query")
                .requiredValue(trainSpan, "trainSpan")
                .requiredValue(testSpan, "testSpan")
                .required(batchSizeFactor > 0.01 && batchSizeFactor < 1, "Batch size must be between 1% and 100%.")
                .required(epochs > 0, "Epochs must be greater than 0");
    }

    public ElementHandle create(PrefixedLive live, ExtensionQualifier qualifier) throws Exception {
        ExecutorService executor = live.system().requestExecutor(1, 1, "");
        TrainerTask trainer = new TrainerTask(live, qualifier.qualifier(), query, trainSpan, testSpan, batchSizeFactor, epochs, learningRate, neuronsHiddenLayer);
        executor.submit(trainer);
        return new ElementHandle.Default(live) {
            @Override
            public ElementState status() {
                if (trainer.getCompletion() < 1)
                    return new ElementState(ElementStatus.VALID_BUT, "Training: " + String.format("%.2f", trainer.getCompletion() * 100) + "%");
                return ElementState.OK;
            }
        };
    }

    public ElementHandle test(PrefixedLive live, ExtensionQualifier qualifier) throws Exception {
        return ElementHandle.OK;
    }
}
