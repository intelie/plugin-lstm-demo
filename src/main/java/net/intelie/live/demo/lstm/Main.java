package net.intelie.live.demo.lstm;

import net.intelie.live.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.ExecutorService;

public class Main implements LivePlugin {
    private static final Logger LOGGER = LoggerFactory.getLogger(Main.class);

    public void start(Live live) throws Exception {
        live.engine().addExtensionType(new LSTMExtensionType(live));
        live.web().addContent("ui.js", getClass().getResource("/ui.js"));
        live.web().addContent("icon.png", getClass().getResource("/icon.png"));
        live.web().addTag(HtmlTag.Position.BEGIN, new HtmlTag.JsFile(live.web().resolveContent("ui.js")));
    }


}
