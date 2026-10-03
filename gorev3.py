import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

B, T, C = 4, 8, 32
head_size = 16
x = torch.randn(B, T, C)

# position embedding
# attention mekanizması sırayı bilmez. hangi token'ın hangi sırada 
# olduğunu modele bildirmek için pozisyon vektörleri eklemeliyiz.
pos_emb_table = nn.Embedding(T, C)
positions = torch.arange(T) # 0'dan 7'ye kadar pozisyonlar
pos_emb = pos_emb_table(positions) # (T, C) boyutunda matris
x = x + pos_emb # Broadcast ile (B, T, C) boyutuna yayılır ve birleştirilir


# self attention head
# query: ben ne ariyorum? key: ben ne içeriyorum?
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)
k = key(x)
q = query(x) 

# Query ve Key'in dot producti ile "affinities" hesaplanıyor
wei = q @ k.transpose(-2, -1) # matris carpimi icin transpose aliyoruz


# scaling
print("scaling deneyi")
k_test = torch.randn(B, T, head_size)
q_test = torch.randn(B, T, head_size)
wei_unscaled = q_test @ k_test.transpose(-2, -1)

# Normalde q ve k'nin varyansı 1 iken, çarpımlarının varyansı head_size'a (16) çıkar.
print(f"Bölmeden önceki varyans (yaklaşık head_size yani {head_size} olmalı): {wei_unscaled.var():.4f}")

wei_scaled = wei_unscaled * (head_size ** -0.5)
print(f"Böldükten sonraki varyans (yaklaşık 1 olmalı): {wei_scaled.var():.4f}\n")

# softmaxin çok sivrileşmemesi için head sizeın kareköküne bölüyoruz. bu sayede varyans 1 oluyor.
print("Softmax'in sivrileşmesi (Bölmeden - yüksek varyansla):")
# Temsili yüksek değerli bir array: Biri çok yüksekse Softmax diğerlerini tamamen öldürür.
print(torch.softmax(torch.tensor([0.1, -0.2, 0.3, -0.2, 8.0]), dim=-1)) 

print("Softmax'in yumuşaması (Bölünerek - düşük varyansla):")
# Değerler normalize edildiğinde ihtimaller daha sağlıklı dağılır.
print(torch.softmax(torch.tensor([0.1, -0.2, 0.3, -0.2, 8.0]) * 0.1, dim=-1)) 

# Kendi wei matrisimizi kurala uygun olarak scale ediyoruz (makaledeki gibi)
wei = wei * (head_size ** -0.5)


# maskeleme (Lower Triangular Mask) ve Value Çarpım
# geleceği gizlemek zorundayız, aksi takdirde model cevabı direkt "kopya çeker".
tril = torch.tril(torch.ones(T, T))
wei = wei.masked_fill(tril == 0, float('-inf')) 
wei = F.softmax(wei, dim=-1) # -inf olan kısımlar exp(-inf) = 0 olur.

print("bir örnekle wei matrisi")
print(wei[0]) # Sadece 1. batch'in ağırlıklarını yazdırıyoruz

# YORUM / OKUMA (Bir Satırın İncelenmesi):
# Yukarıdaki wei[0] çıktısında örneğin 4. satıra (index 3) bakarsak (Tensor'ün 4. satırı):
# [x.xx, y.yy, z.zz, w.ww, 0.0000, 0.0000, 0.0000, 0.0000] şeklinde bir dağılım görürüz.
# Bu satır bize şunu söyler: 4. token tahmin yaparken;
# 1. token'a x.xx oranında, 2. token'a y.yy oranında, 3. token'a z.zz oranında 
# ve bizzat kendisine w.ww oranında dikkat ediyor. (Bu 4 değerin toplamı 1'dir).
# Sağ taraftaki 0'lar, 5, 6, 7 ve 8. token'lardır. Maskeleme sayesinde 4. token
# kendisinden sonra gelen gelecekteki harflere hiçbir şekilde bakamaz (ağırlık 0).

# Son olarak Value matrisi ile ağırlıkları çarpıyoruz
v = value(x) # (B, T, head_size) -> Bir token seçilirse ne taşıyacak
out = wei @ v # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)

print('out shape', out.shape) # Beklenen: torch.Size([4, 8, 16])