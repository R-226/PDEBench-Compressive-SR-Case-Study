import pickle

# 以二进制读模式打开Pickle文件
with open('upsample_expert.pickle', 'rb') as file:
    # 使用pickle模块的load函数加载数据
    data = pickle.load(file)

# 打印加载的数据
print(data)