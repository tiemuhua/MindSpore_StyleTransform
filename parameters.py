class Parameter:
    class LossParams:
        style_weight = 1.0
        content_weight = 1.0
        smooth_weight = 1.0
        content_layers_weight = {"conv2": 1}
        style_layers_weight = {"conv0": 0.5e-3, "conv1": 0.5e-3, "conv2": 0.5e-3, "conv3": 0.5e-3}

        class VggParams:
            pad_mode = "same"
            padding = 0
            has_bias = False
            weight_init = 'normal'

    class PredictionParams:
        bottle_neck_depth = 100

        class InceptionParams:
            num_classes = 10

    class TransformParams:
        pass

    class DataParams:
        height = 326
        width = 256
        batch_size = 5

    class NormalizeParams:
        eps = 1e-5
