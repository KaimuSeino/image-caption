"""
pack_padded_sequenceとは

pack_padded_sequenceとは何をしていて、なぜ必要なのかを考える
RNN（ LSTM または GRU ）をトレーニングする場合、可変長配列をバッチ処理することは困難である。
例えばサイズ8のバッチの配列の長さが[4, 3, 4, 7, 8, 6, 8, 5]の場合、全ての配列をパディングすれば結果として
8つの配列の長さ8になる。最終的に64回の計算（8x8）を行うことになるが、必要な計算は45回だけである。
つまりこの問題を解決してくれるのがpack_padded_sequenceである。
"""

import torch

a = torch.full((3, 3), fill_value=1)
b = torch.full((2, 3), fill_value=2)
c = torch.full((1, 3), fill_value=3)

print(a)
print(b)
print(c)

"""
1. pad_sequence
関数pad_sequenceによって、すべての長さが配列の最大長さに揃えられて、余分な要素には0が入っている。
"""
from torch.nn.utils.rnn import pad_sequence
d = pad_sequence([a,b,c], batch_first=True)
print("pad_sequence", d)

"""
2. pack_sequence
複数の配列[a,b,c]を一つにまとめてpackedSequenceオブジェクトに格納するための関数。
"""
from torch.nn.utils.rnn import pack_sequence
e = pack_sequence([a,b,c])
print("pack_sequence", e)

"""
3. pack_padded_sequence
関数 pack_padded_sequenceは、「pad_sequence」された配列に、「pack_sequence」の処理をしてくれる。
"""
from torch.nn.utils.rnn import pack_padded_sequence
embeddings = pad_sequence([a,b,c], batch_first=True)
lengths = torch.tensor([3,2,1])
f = pack_padded_sequence(embeddings, lengths, batch_first=True)
print("pack_padded_sequence", f)