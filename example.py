# 1.加载数据
from torch.utils.data import Dataset
class SSTDataSet(Dataset):
    def __init__(self, path, test=False):
        self.data=[]
        with open(path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                if test:
                    raw_words = line.strip()
                else:
                    raw_words, target = line.strip().split('\t')
                    
                # simple preprocesiing
                if raw_words.endswith("\""):
                    raw_words = raw_words[:-1]
                if raw_words.startswith('"'):
                    raw_words = raw_words[1:]
                raw_words = raw_words.replace('""','"')
                raw_words = raw_words.split(' ')
                if test:
                    self.data.append({
                        'raw_words':raw_words,
                        'target':0,
                    })
                else:
                    self.data.append({
                        'raw_words':raw_words,
                        'target':1 if target=='positive' else 0,
                    })
        print("# samples: {}".format(len(self.data)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]
    
    def convert_word_to_ids(self, vocab):
        for i in range(len(self.data)):
            ins = self.data[i]
            word_ids = [vocab[x] if x in vocab else vocab['<unk>'] for x in ins['raw_words']]
            self.data[i]['input_ids'] = word_ids


# 读取数据
train_set = SSTDataSet('./train.data')
dev_set = SSTDataSet('./valid.data')

print(train_set[0])



# 2.建立词表，统计训练集和开发集中出现的token加入词表，此外还需要<pad>和<unk>两个特殊的token
# 建立词表的目的是把词映射成数字
vocab = {'<pad>': 0, '<unk>': 1}
for ins in train_set:
    for word in ins['raw_words']:
        if word not in vocab:
            vocab[word] = len(vocab)

for ins in dev_set:
    for word in ins['raw_words']:
        if word not in vocab:
            vocab[word] = len(vocab)

train_set.convert_word_to_ids(vocab)
dev_set.convert_word_to_ids(vocab)

print(train_set[0])

# 若使用预训练的word embedding，如word2vec和glove，一般采用他们提供的token构建词表，同时在下面的模型定义中也使用他们提供的向量完成word embedding的初始化
# 3.建立模型,需要写一个定义模型结构及前向传播的类来继承torch.nn.Module，至少需要实现__init__和forward两个方法
import torch
import torch.nn as nn
import numpy as np

# xavier init
def init_embedding(input_embedding, seed=1337):
    """initiate weights in embedding layer
    """
    torch.manual_seed(seed)
    scope = np.sqrt(3.0 / input_embedding.size(1))
    nn.init.uniform_(input_embedding, -scope, scope)

# xavier init
def init_linear(input_linear, seed=1337):
    """initiate weights in linear layer
    """
    torch.manual_seed(seed)
    scope = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
    nn.init.uniform(input_linear.weight, -scope, scope)
    if input_linear.bias is not None:
        input_linear.bias.data.zero_()

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_labels=2, dropout=0.5, num_layers=1):
        super(LSTMModel, self).__init__()

        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                            bidirectional=True)
        self.out = nn.Linear(hidden_size * 2, num_labels)

        init_linear(self.out)
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, x, y):
        batch_size, seq_len = x.size()
        x = self.dropout(self.embed(x))
        rnn_out, _ = self.lstm(x)
        rnn_out = rnn_out.view(batch_size, seq_len, 2, self.hidden_size)
        rnn_out = torch.cat([rnn_out[:, -1, 0, :], rnn_out[:, 0, 1, :]], dim=-1)
        rnn_out = self.dropout(rnn_out)
        logits = self.out(rnn_out)
        loss = self.loss_fct(logits.view(-1, self.num_labels), y.view(-1))
        pred = torch.argmax(logits, dim=-1)

        return loss, pred

model = LSTMModel(vocab_size=len(vocab), embed_dim=100, hidden_size=200, num_labels=2)
print(model)
        

# 4.训练和评估
# 把数据集划分成一个个mini-batch喂入模型进行训练。这里我们大概需要这么几个模块：
# 1.Optimizer，这里使用torch.optim.Adam（参数更新）
# 2.Sampler,为了高效训练，现在一般将整个训练集划分为多个mini—batch，那么就需要一个sampler来从整个训练集中去采样出batch_size个
# 样本。一般在训练时使用torch.utils.data.RandomSampler以加强随机性，在测试中使用torch.utils.data.SequentialSampler来按顺序
# 输出预测（测试集按顺序输出）
# 3.DataLoader，把Dataset进一步抽象为可以按照batch进行迭代的类，可以看作是组合了Dataset和Sampler，实际上，pytorch会用sampler来得到
# batch_size个采样出的样本的index，再用这些index去根据前面实现的Dateset.__getitem__取出Dataset中对应的样本来组成一个batch。需要要注意的是，
# padding也是在这一步完成的，通常通过向DataLoader的collate_fn参数传入一个函数

# （collate_fn的作用是把每一个batch打包成tensor）
def custom_collate(batch):
    '''padding'''

    DEFAULTT_PADDING_LABEL = 0
    input_ids, targets = [], []
    for x in batch:
        input_ids.append(x['input_ids'])
        targets.append(x['target'])
    max_len = max(len(x) for x in input_ids)
    batch_input_ids = [x + [DEFAULTT_PADDING_LABEL]\
        * (max_len - len(x)) for x in input_ids]
    # 转化为tensor
    batch_input_ids = torch.LongTensor(batch_input_ids)
    batch_targets = torch.LongTensor(targets)
    return batch_input_ids, batch_targets

# 接下来看看自定义的collate_fn用于DataLoader的效果

from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
sampler = RandomSampler(data_source=train_set)
train_data_iter = DataLoader(train_set, batch_size=32, sampler=sampler, num_workers=4, collate_fn=custom_collate)
print('# samples: ',len(train_set))
print('# batches: ',len(train_data_iter))   
for batch in train_data_iter:
    print(batch[0].shape)
    print(batch[1].shape)
    print(batch[0])
    print(batch[1])
    break

# 把一个mini-batch的数据喂入模型，并执行forward求loss和backward求梯度，最后使用优化器更新参数。下面看一下一次迭代的具体流程
from torch.optim import Adam

optimizer = Adam(model.parameters(), lr = 1e-3)
iter = train_data_iter._get_iterator()
batch = iter.next()
x, y = batch

device = 'cuda:6'
model.to(device)
model.train()
loss, pred = model(x.to(device), y.to(device))
print(loss)
print(pred)

print(model.out.weight.grad)
loss.backward()
print(model.out.weight.grad[0][:10])

print(model.out.weight.shape)
print(model.out.weight[0][:10])
optimizer.step()
print(model.out.weight[0][:10])

# 注意每次训练要zero_grad，否则每次的梯度会积累
model.zero_grad()
print(model.out.weight.grad[0][:10])

# 可以使用zero_grad()完成梯度累积

# The first batch
batch_1 = iter.next()
x, y = batch_1
loss, pred = model(x.to(device), y.to(device))
loss.backward()
print("Gradients of the first batch of data")
print(model.out.weight.grad[0][:10])

# The second batch
batch_2 = iter.next()
x, y = batch_2
loss, pred = model(x.to(device), y.to(device))
loss.backward()
print("Gradients of the second batch of data")
print(model.out.weight.grad[0][:10])

# Update with the cumulated gradient
optimizer.step()#这里的step相当于把上面两次计算出的梯度叠加做的step
model.zero_grad()
print(model.out.weight.grad[0][:10])



# 下面是一次完整的训练流程
    # import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from torch.optim import Adam
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
from tqdm import tqdm, trange
from sklearn.metrics import accuracy_score

device = 'cuda:6' if torch.cuda.is_available() else 'cpu'
model.to(device)
optimizer = Adam(model.parameters(), lr = 1e-3)
sampler = RandomSampler(data_source=train_set)
train_data_iter = DataLoader(train_set, batch_size=32, sampler=sampler, num_workers=4, collate_fn=custom_collate)

num_epochs = 10
tr_loss = 0.0
global_step = 0
eval_every = 100

for i_epoch in trange(num_epochs, desc="Epoch"):
    epoch_iterator = tqdm(train_data_iter, desc="Iteration")
    acc = 0
    for step, batch in enumerate(epoch_iterator):
        model.train()
        x, y = batch
        loss, pred = model(x.to(device), y.to(device))
        loss.backward()
        tr_loss += loss.item()
        optimizer.step()
        model.zero_grad()
        global_step += 1
        if global_step % eval_every == 0:
            # evaluate
            total_pred, total_target = [],[]
            model.eval()
            sampler = SequentialSampler(data_source=dev_set)
            dev_data_iter = DataLoader(dev_set, batch_size=32, sampler=sampler, num_workers=4, collate_fn=custom_collate)
            for dev_batch in dev_data_iter:
                x, y = dev_batch
                with torch.no_grad():
                    loss, pred = model(x.to(device), y.to(device))
                    pred = pred.detach().cpu().numpy().tolist()
                    target = y.to('cpu').numpy().tolist()
                    total_pred.extend(pred)
                    total_target.extend(target)
            acc = accuracy_score(total_target, total_pred)
            print("step: {} | loss: {} | acc: {}".format(global_step, tr_loss/eval_every, acc))
            tr_loss = 0
            if acc> 0.8:
                break
    if acc>0.8:
        break


# 用训练好的模型来预测测试集
test_set = SSTDataSet('./test.data', test=True)
test_set.convert_word_to_ids(vocab)
test_sampler = SequentialSampler(data_source=test_set)
test_data_iter = DataLoader(test_set, batch_size=32, sampler=test_sampler, num_workers=4, collate_fn=custom_collate)
preds = []

for step, test_batch in tqdm(enumerate(test_data_iter)):
    model.eval()
    x, y = test_batch
    with torch.no_grad():
        loss, pred = model(x.to(device), y.to(device))
        pred = pred.detach().cpu().numpy().tolist()
        preds.extend(pred)

print(preds)

# 将预测结果写入文件
with open('testResult.txt', encoding='utf-8', mode='w') as f:
    for pred in preds:
        f.write('positive' if pred == 1 else 'negative')
        f.write('\n')

# 将验证集的预测结果写入文件
preds = preds = []
dev_data_iter = DataLoader(dev_set, batch_size=32, shuffle=False, num_workers=4, collate_fn=custom_collate)

for step, valid_batch in tqdm(enumerate(dev_data_iter)):
    model.eval()
    x, y = valid_batch
    with torch.no_grad():
        loss, pred = model(x.to(device), y.to(device))
        pred = pred.detach().cpu().numpy().tolist()
        preds.extend(pred)

print(preds)


with open('validResult.txt', encoding='utf-8', mode='w') as f:
    for pred in preds:
        f.write('positive' if pred == 1 else 'negative')
        f.write('\n')

