package net.intelie.live.demo.lstm;

import com.google.common.base.Strings;
import net.intelie.pipes.CustomizeFunction;
import net.intelie.pipes.Export;
import net.intelie.pipes.Function;
import net.intelie.pipes.HelpData;
import net.intelie.pipes.util.Escapes;

public class Predictor implements CustomizeFunction {
    private final TrainedData trained;
    private final String qualifier;
    private final String functionName;

    public Predictor(TrainedData trained, String qualifier) {
        if (Strings.isNullOrEmpty(qualifier)) qualifier = "default";

        this.trained = trained;
        this.qualifier = qualifier;
        this.functionName = Escapes.safeIdentifier(qualifier);
    }

    @Export("predict")
    public Double predict(Double x) {
        if (x == null) return null;
        return trained.predict(x);
    }

    @Override
    public String name(Function original) {
        return "predict.lstm." + functionName;
    }

    @Override
    public String description(Function original) {
        return "LSTMPredictor";
    }

    @Override
    public HelpData help(Function original) {
        return new HelpData(
                "function",
                name(original),
                name(original) + "(number x)",
                "Predicts next time series value using LSTM model: " + qualifier,
                null,
                null,
                null,
                null
        );
    }
}
