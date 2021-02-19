import sys, os
sys.path.append('./vgg')
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.ops.operations as P
import file_operations as f
from mindspore import Tensor
from mindspore.train.callback import LossMonitor,ModelCheckpoint,CheckpointConfig
from mindspore import context
import argparse
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import numpy as np
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MindSpore LeNet Example')
    parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend', 'GPU', 'CPU'],
                        help='device where the code will be implemented (default: CPU)')
    args = parser.parse_args()
    context.set_context(mode=context.PYNATIVE_MODE, device_target=args.device_target)
    dataset_sink_mode = not args.device_target == "Ascend"
    
    from mindspore import dataset as ds
    import file_operations as f
    import network
    net=network.network()
    param_dict = load_checkpoint("/root/xl/lab/arrange/dtd2-1_1100.ckpt")
    load_param_into_net(net, param_dict)
    def get_data(num):
        for _ in range(num):
            content=f.load_np_image("colva_beach_sq.jpg") 
            style=f.load_np_image('profile.jpg')
            data=np.append([content],[style],axis=0)
            yield data,data
            
            
    def get_data2(content_path,style_path,num):
        import os
        import random
        content_dirs = os.listdir(content_path)
        style_dirs = os.listdir(style_path)
        
        for i in range(num):
            index1=random.randint(0,len(content_dirs)-1)
            index2=random.randint(0,len(style_dirs)-1)
            content_pa = content_path+content_dirs[index1]
            style_pa=style_path+style_dirs[index2]
            content=f.crop_and_resize(content_pa) 
            style=f.crop_and_resize(style_pa)
            data=np.append([content],[style],axis=0)
            yield data,data


    def create_dataset(num_data, batch_size=16, repeat_size=1):
        input_data = ds.GeneratorDataset(list(get_data(num_data)), column_names=['data','label'])
        return input_data
    
    def create_dataset2(num_data, batch_size=16, repeat_size=1):
        input_data = ds.GeneratorDataset(list(get_data2("/data/imagenet/train/n01614925/","/data/dtd/dtd/images/cobwebbed/",num_data)), column_names=['data','label'])
        return input_data
     
    import loss
    loss = loss.total_loss()
    from mindspore import Model
    print(net.trainable_params())
    opt = nn.Adam(net.trainable_params(), learning_rate=2e-6)
    model = Model(net, loss, opt)
    data_number = 1200
    batch_number = 1
    repeat_number = 1
    ds_train = create_dataset2(data_number, batch_size=batch_number, repeat_size=repeat_number)
    config_ck = CheckpointConfig(save_checkpoint_steps=100, keep_checkpoint_max=3)
    ckpoint_cb = ModelCheckpoint(prefix="dtd_slow", config=config_ck)
    model.train(1, ds_train, callbacks=[ckpoint_cb,LossMonitor()],dataset_sink_mode=dataset_sink_mode)
