"""
RNN（ Recurrent Neural Network ）
再帰型ニューラルネットワークという。通常のニューラルネットワークでは全ての入力が独立して処理されるが、
RNNは「過去の情報を保持」しつつ、計算を行うことができる。そのため、「文脈」や「時間的な依存関係」を考慮できる。

RNNの課題
1.勾配消失・勾配爆発問題：
    長いシーケンスを学習する際、勾配が非常に小さくなる（勾配消失）か、非常に大きくなる（勾配爆発）問題がある。
2.長期依存関係の学習が難しい：
    過去の学習が遠く離れている場合、それを適切に保持して利用するのが難しい。
これらの問題を解決するために、LSTM（ Long Short-Term Memory ）やGRU（ Gated Recurrent Unit ）などの派生モデルが開発されている。
"""
import torch
import torch.nn as nn
import numpy as np

data = "hello world"
chars = list(set(data))
# chars = ['d', 'l', 'o', 'h', 'w', 'e', ' ', 'r']
char_to_idx = {ch: idx for idx, ch in enumerate(chars)} # 文字 -> インデックス
# char_to_idx {' ': 0, 'h': 1, 'w': 2, 'e': 3, 'd': 4, 'o': 5, 'r': 6, 'l': 7}
idx_to_char = {idx: ch for idx, ch in enumerate(chars)} # インテックス -> 文字
# index_to_char {0: ' ', 1: 'h', 2: 'w', 3: 'e', 4: 'd', 5: 'o', 6: 'r', 7: 'l'}
print('chars:', chars)
print('char_to_idx: ', char_to_idx)
print('index_to_char: ', idx_to_char)

# パラメータ
input_size = len(chars)
hidden_size = 10
output_size = len(chars)
sequence_length = len(data) - 1 # 入力と出力のペア数

print('input_size:', input_size)
print('hidden_size:', hidden_size)
print('output_size:', output_size)
print('sequence_length:', sequence_length)

# データの準備
x_data = [char_to_idx[ch] for ch in data[:-1]] # 入力: "hello worl"
y_data = [char_to_idx[ch] for ch in data[1:]] # 出力: "ello world"
"""
ワンホットエンコーディング

データを機械学習が扱いやすい数値データに変換するための方法である。
カテゴリごとにバイナリベクトルを作成して、各ベクトルの中で対応するカテゴリのみを1、それ以外をすべて0にする。
"""
x_one_hot = [np.eye(input_size)[x] for x in x_data] # ワンホットエンコーディング

print('x_data:', x_data)
print('y_data:', y_data)
print('x_one_hot:', x_one_hot)

# テンソルに変換
inputs = torch.tensor(x_one_hot, dtype=torch.float32)
labels = torch.tensor(y_data, dtype=torch.long)

print('inputs:', inputs)
print('labels:', labels)

# RNNモデルの定義
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden) # RNNの計算
        out = self.fc(out) # 全結合層で出力
        return out, hidden

# モデルの初期化
model = SimpleRNN(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("########モデルの初期化###########")
print("model:", model)
print("criterion:", criterion)
print("optimizer:", optimizer)

# トレーニング
num_epochs = 100
for epoch in range(num_epochs):
    hidden = torch.zeros(1, 1, hidden_size) # 隠れ状態の初期化
    optimizer.zero_grad()
    outputs, hidden = model(inputs.unsqueeze(0), hidden)
    loss = criterion(outputs.view(-1, output_size), labels)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1} / {num_epochs}], Loss: {loss.item():.4f}")

# モデルのテスト
print("-------モデルのテスト--------")
hidden = torch.zeros(1, 1, hidden_size)
test_input = inputs[0].unsqueeze(0).unsqueeze(0) # 最初の文字("h")

print("hidden:", hidden)
print("test_input:", test_input)

result = []
for _ in range(sequence_length):
    output, hidden = model(test_input, hidden)
    _, predicted = torch.max(output, 2) # 最大値のインデックスを取得
    result.append(idx_to_char[predicted.item()])
    test_input = torch.eye(input_size)[predicted.item()].unsqueeze(0).unsqueeze(0)

print("Generated sequence:", "".join(result))