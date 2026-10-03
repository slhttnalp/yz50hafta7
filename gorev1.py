import torch
import torch.nn as nn
from torch.nn import functional as F

with open('input.txt', 'r', encoding='utf-8') as f:   #datayi okuma
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars) 

stoi = { ch:i for i,ch in enumerate(chars) }  #klasik stoi islemi
itos = { i:ch for i,ch in enumerate(chars) }

encode = lambda s: [stoi[c] for c in s] 
decode = lambda l: ''.join([itos[i] for i in l]) 

data = torch.tensor(encode(text), dtype=torch.long)  #tum metni sayilara cevirip tek boyutlu bir tensor yapiyoruz
n = int(0.9*len(data)) 
train_data = data[:n]
val_data = data[n:]

torch.manual_seed(1337)
batch_size = 4   # 4 farkli batch aliyoruz
block_size = 8   #her batch 8 eleman

def get_batch(split):  # rastgele mini-batch almak icin fonksiyon yaziyoruz. 
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))   #rastgele 4 tane baslangic indexi veriyor
    
    x = torch.stack([data[i:i+block_size] for i in ix])    #rastgele 4 noktadan baslayan 8 elemanlik inputlar
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])  #x'in bir adim ileri kaydirilmis hali
    return x, y

xb, yb = get_batch('train')

torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        
    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)
        
        if targets is None:
            loss=None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)  #pytorch cross entropy'nin formatina uymasi icin yapiyoruz
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):   #sample alma
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:,-1,:] # gecmiste ne oldugu ile ilgilenmiyor sadece son harfe bakiyor
            probs = F.softmax(logits,dim=-1)
            idx_next = torch.multinomial(probs,num_samples=1)
            idx = torch.cat((idx,idx_next),dim=1) # uretilen harfi dizinin sonuna ekle ve devam et
        return idx

m = BigramLanguageModel(vocab_size)
logits, loss = m(xb,yb)

optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)  #burada ilk defa farkli bi training yontemi kullandik.
batch_size = 32
for steps in range(20000):
    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())
print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=500)[0].tolist()))