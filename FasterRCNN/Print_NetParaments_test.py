import torch
from torchsummary import summary

from nets.frcnn import FasterRCNN
from nets.resnet50 import resnet50
from nets.vgg16 import decom_vgg16
from nets.mobilenet_v3 import mobilenet_v3


if __name__ == "__main__":
    
    # 需要使用device来指定网络在GPU还是CPU运行
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')       
    backbone = "resnet50" 
    # features, classifier = decom_vgg16()
    features = mobilenet_v3()
    model = features.to(device) 
    summary(features, input_size=(3, 600, 600), batch_size=1, device="cuda")  

    # mobilenetv1-yolov4 40,952,893  
    # mobilenetv2-yolov4 39,062,013  
    # mobilenetv3-yolov4 39,989,933  

    # 修改了panet的mobilenetv1-yolov4 12,692,029
    # 修改了panet的mobilenetv2-yolov4 10,801,149
    # 修改了panet的mobilenetv3-yolov4 11,729,069
