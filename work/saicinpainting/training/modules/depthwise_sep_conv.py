import paddle
import paddle.nn as nn

class DepthWiseSeperableConv(nn.Layer):
    def __init__(self, in_dim, out_dim, *args, **kwargs):
        super().__init__()
        if 'groups' in kwargs:
            # ignoring groups for Depthwise Sep Conv
            del kwargs['groups']
        
        self.depthwise = nn.Conv2D(in_dim, in_dim, *args, groups=in_dim, **kwargs)
        self.pointwise = nn.Conv2D(in_dim, out_dim, kernel_size=1)
        
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out