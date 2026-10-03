import torch
import torch.nn.functional as F

torch.manual_seed(1337)
B, T, C = 4, 8, 2 
x = torch.randn(B, T, C)

#1: İç içe For Döngüsü 
# Her harf için, o anki harf ve ondan öncekilerin ortalamasını alıyoruz
xbow = torch.zeros((B, T, C))
for b in range(B):
    for t in range(T):
        xprev = x[b, :t+1] # t. zamana kadar olan geçmiş (şekil: t+1, C)
        xbow[b, t] = torch.mean(xprev, dim=0)

#2: Matris Çarpımı (torch.tril ile)
# torch.tril (triangular lower): Alt üçgensel matris oluşturur 
wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(1, keepdim=True)
xbow2 = wei @ x 

#3: Softmax
tril = torch.tril(torch.ones(T, T))
wei = torch.zeros((T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))
# Softmax, -inf olan yerleri 0'a, diğer yerleri ise eşit ağırlıklara böler
wei = F.softmax(wei, dim=-1)
xbow3 = wei @ x

#3 yöntemin de matematiksel olarak tamamen aynı sonucu verdiğini kanıtlama
print("xbow == xbow2", torch.allclose(xbow, xbow2))
print("xbow == xbow3", torch.allclose(xbow, xbow3))

print("\nÖrnek wei matrisi:\n", wei)
