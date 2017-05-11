package net.intelie.live.demo.lstm;

import net.intelie.live.*;

import java.util.Set;

public class LSTMExtensionType implements ExtensionType<LSTMConfig> {
    private final Live live;

    public LSTMExtensionType(Live live) {
        this.live = live;
    }

    @Override
    public String typename() {
        return "lstm-demo";
    }

    @Override
    public ExtensionArea area() {
        return ExtensionArea.PLATFORM;
    }

    @Override
    public Set<ExtensionRole> roles() {
        return ExtensionRole.start().ok();
    }

    @Override
    public ElementHandle register(ExtensionQualifier qualifier, LSTMConfig config) throws Exception {
        return SafeElement.create(live, qualifier, config::create);
    }

    @Override
    public ElementState test(ExtensionQualifier qualifier, LSTMConfig config) throws Exception {
        SafeElement.create(live, qualifier, config::test).close();
        return ElementState.OK;
    }

    @Override
    public LSTMConfig parseConfig(String config) throws Exception {
        return LiveJson.fromJson(config, LSTMConfig.class);
    }
}
