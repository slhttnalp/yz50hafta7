import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import urllib.request

batch_size = 32
block_size = 8
max_iters = 6000
eval_interval = 500
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32

torch.manual_seed(1337)
#eger input.txt yoksa internetten bulup indirir
if not os.path.exists('input.txt'):
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    urllib.request.urlretrieve(url, 'input.txt')

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# self-attention head modülü
class Head(nn.Module):
    """ Tek bir self-attention head """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # tril bir ağırlık değildir, bu yüzden register_buffer ile kaydedilir.
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        
        # attention skorlarını hesapla
        wei = q @ k.transpose(-2, -1) * (C ** -0.5) # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # Geleceği maskele
        wei = F.softmax(wei, dim=-1)
        
        v = self.value(x) # (B, T, head_size)
        out = wei @ v     # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out

# yeni bigram modeli
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Artık doğrudan vocab_size yerine bir gömme (embedding) boyutuna iniyoruz
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        # modüle single head'i ekliyoruz (head_size = n_embd kullanıyoruz)
        self.sa_head = Head(n_embd)
        
        # attention çıktısını alıp tekrar kelime dağarcığı boyutuna çeviren lineer katman
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx ve targets hem (B,T) boyutunda tensorler
        tok_emb = self.token_embedding_table(idx) # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, n_embd)
        x = tok_emb + pos_emb # (B, T, n_embd)
        
        x = self.sa_head(x) # Single head self-attention uygula
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # idx'in uzunluğu block_size'ı geçemez, aksi takdirde position embedding hata verir.
            idx_cond = idx[:, -block_size:] 
            
            # idx_cond modelin içine girer
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] # Sadece son zaman adımını al
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# egitim
model = BigramLanguageModel()
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("Eğitim başlıyor... (Cihaz:", device, ")")
for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Adım {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

final_losses = estimate_loss()
print(f"Eğitim tamamlandı: train loss {final_losses['train']:.4f}, val loss {final_losses['val']:.4f}")


"""
1. Val loss ne kadar düştü, neden düştü?
---------------------------------------
Standart Bigram modelinde (Görev 1) val loss yaklaşık 2.41 civarındaydı. 
Head'i (Self-Attention) işin içine kattığımızda val loss yaklaşık 2.39 seviyelerine düştu.
Neden düştü? Çünkü saf Bigram modelinde bir harf üretilirken SADECE bir önceki harfe bakılıyordu. 
Tokens birbirleriyle hiç konuşmuyordu. Self-attention eklediğimizde ise, üretilen harf kendinden 
önceki 8 harfin (block_size) tamamına bakıp, ihtiyacı olan bağlamı Query ve Key çarpımıyla ağırlıklı 
olarak seçmeye başladı. Geçmiş harflerle kurulan bu "iletişim" tahmin gücünü artırdı ve hatayı (loss) düşürdü.

2. Metin üretirken girdiyi neden son block_size harfe kırpmak (crop) zorundayız?
---------------------------------------
Üretim döngüsü (generate) çalışırken metin gitgide uzar. Örneğin 5 harfle başladık, 
döngü döndükçe uzunluk 10, 20, 100 harfe çıkar.
Fakat bizim 'position_embedding_table' matrisimiz sadece 'block_size' (bizim kodda 8) kadar satıra sahip. 
Eğer modele 9 harflik bir tensor verirsek, model 9. pozisyonun embedding'ini arayacak ama 
tabloda sadece 0'dan 7'ye kadar indeksler olduğu için "Index out of bounds" (sınır aşımı) hatası patlayacaktır.
Bu yüzden modelin kapasitesi ne kadarsa (block_size), sadece o kadarlık bir pencereyi 
(son 8 harfi) kesip (idx_cond = idx[:, -block_size:]) modele vermeliyiz.
"""

print("sample")
context = torch.zeros((1, 1), dtype=torch.long, device=device) # İlk karakter 'newline' (0 indeksi)
print(decode(model.generate(context, max_new_tokens=200)[0].tolist()))