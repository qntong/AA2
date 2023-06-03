import pickle
import json

# 读取pickle文件
with open('/home/utter/zhy/AA/IEMOCAP_features_bert.pkl', 'rb') as pkl_file:
    data = pickle.load(pkl_file)
print(data)

# 将数据写入json文件
with open('/home/utter/zhy/AA/IEMOCAP_features_bert.json', 'w') as json_file:
    json.dump(data, json_file)