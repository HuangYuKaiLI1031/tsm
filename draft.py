# import torch 
# import torch.nn as nn 
# import torchvision

# model = torchvision.models.resnet50(pretrained=False)


# modules = list(model.modules())
# # print(modules)
# first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))
# print(first_conv_idx)
# # # num = next(first_conv_idx)
# conv_layer = modules[first_conv_idx[0]]
# print("conv_layer:",conv_layer)
# container = modules[first_conv_idx[0] - 1]
# # state_dict作为python的字典对象将每一层的参数映射成tensor张量
# # 获取卷积层的名称
# layer_name = list(container.state_dict().keys())[0][:-7]
# print(conv_layer.in_channels)
# # params = [x.clone() for x in conv_layer.parameters()]
# # new_kernel_size = params[0].size()
# # print(params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous())
# # print("contain:",contain)
# # print(modules[first_conv_idx-1])
# # params = [x.clone() for x in conv_layer.parameters()]
# # kernel_size = params[0].size()
# # print(type(kernel_size))
# # # kernel_size[:1] + (3 * 5,) + kernel_size[2:]
# # print((kernel_size[:1] + (3 * 5,) + kernel_size[2:]))
# # print(params[0].size())
# # print(modules[next(first_conv_idx)])
# # print(type(modules))

# # a = [1,2,3,4,5]
# # print((a[:1]+a[2:]+[1,]))
import numpy as np
import torch

x = [
[[1,2,3,4],[5,6,7,8],[9,10,11,12]],
[[13,14,15,16],[17,18,19,20],[21,22,23,24]]
]
x = torch.tensor(x).float()
#
print("shape of x:")  ##[2,3,4]                                                                      
print(x)                                                                                   
#
print("shape of x.mean(axis=0,keepdim=True):")          #[1, 3, 4]
print(x.mean(axis=0,keepdim=True))                       
#
print("shape of x.mean(axis=0,keepdim=False):")         #[3, 4]
print(x.mean(axis=0,keepdim=False).shape)                     
#
print("shape of x.mean(axis=1,keepdim=True):")          #[2, 1, 4]
print(x.mean(axis=1,keepdim=True))                      
#
print("shape of x.mean(axis=1,keepdim=False):")         #[2, 4]
print(x.mean(axis=1,keepdim=False).shape)                    
#
print("shape of x.mean(axis=2,keepdim=True):")          #[2, 3, 1]
print(x.mean(axis=2,keepdim=True).shape)                     
#
print("shape of x.mean(axis=2,keepdim=False):")         #[2, 3]
print(x.mean(axis=2,keepdim=False).shape)  
