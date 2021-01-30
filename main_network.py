from mindspore import nn

from style_prediction_network import StylePredictionNetwork
from style_transform_network import StyleTransformNetwork


class MainNetwork(nn.Cell):
    def __init__(self):
        super(MainNetwork, self).__init__()
        self._style_prediction_network = StylePredictionNetwork()
        self._style_transform_network = StyleTransformNetwork()

    def construct(self, content_img, style_img):
        # TODO beta and gamma should be merged into one tensor in nn.Cell.__call__ function
        beta, gamma = self._style_prediction_network(style_img)
        generated_img = self._style_transform_network(content_img, beta, gamma)
        return generated_img
