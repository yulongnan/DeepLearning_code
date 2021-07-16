# DeepLearning_code

## 权重文件路径排序与索引中断前一次的权重文件
import os
files = os.listdir('./logs')
print(files)
file_weight_pth = []
for file in files:
    print(  os.path.splitext(file) )
    if os.path.splitext(file)[1]=='.pth':
        file_weight_pth.append(file)
print('+++++')
print(file_weight_pth)

file_weight_pth.sort(key=lambda ele:ele[5], reverse=True)
print(file_weight_pth[0])  

## 函数 权重文件路径排序与索引中断前一次的权重文件
def Get_weightFile():
    import os
    files = os.listdir('./logs')
    file_weight_pth = []
    for file in files:
        if os.path.splitext(file)[1]=='.pth':
            file_weight_pth.append(file)
    file_weight_pth.sort(key=lambda ele:ele[5], reverse=True)
    return file_weight_pth[0]

print(Get_weightFile())
