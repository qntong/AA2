import json
import torch
from transformers import BertTokenizer
import pickle
# 初始化tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 加载数据
with open('./data/DailyDialog/test_data1.json', 'r') as f:
    data = json.load(f)

# 定义一个函数用来处理每一个句子：
def process_text(text, max_length=96):
    inputs = tokenizer.encode_plus(text, max_length=max_length, padding='max_length', truncation=True, return_tensors="pt")
    return inputs['input_ids'][0]

# 遍历数据
for conversation in data:
    for utterance in conversation:
        text = utterance['text']
        features = process_text(text)
        utterance['features'] = features.tolist()  # 在保存的时候，我们还是需要把张量转化为列表，否则无法保存为json

# 保存处理后的数据

with open('./data/DailyDialog/processed_test_data1.pkl', 'wb') as f:
    pickle.dump(data, f)