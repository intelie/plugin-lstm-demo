(function() {
    var ExtensionService = require('live/services/extension');

    const service = {
        name: 'LSTM Predictor (demo)',
        type: 'lstm-demo',
        origin: 'Plugin LSTM (demo)',
        roles: [],
        icon: '/content/plugin-lstm-demo/icon.png',
        ui: { form: null, view: null }
    };

    ExtensionService.register(service);
})();