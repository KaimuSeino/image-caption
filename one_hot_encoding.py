"""
ワンホットエンコーディングとは？

カテゴリデータを機械学習モデルが扱いやすい数値データに変換するための方法の一つである。
この方法では、カテゴリごとにバイナリベクトルを作成し、各ベクトルの中で対応するカテゴリのみを1、それ以外をすべて0にする.

特徴
1.カテゴリを数値ベクトルに変換する方法
各カテゴリにユニークな整数IDを割り当てた後、それをバイナリ形式のベクトルに変換する。
2.カテゴリ同士の優劣や大小関係を作らない
例えば「赤=0」「青=1」「緑=2」のようなラベルエンコーディングでは、数値の大小関係が誤解される可能性がある。
一方、ワンホットエンコーディングはその問題を回避する。
"""

# ["赤", "青", "緑"]
categories = ["赤", "青", "緑"]

# カテゴリをインデックスにマッピング
catecory_to_index = {}
for idx, catecory in enumerate(categories):
    catecory_to_index[catecory] = idx


# 手動でワンホットエンコーディングを実装してみる
def one_hot_encode(data, num_categories):
    one_hot = [0] * num_categories
    one_hot[data] = 1
    return one_hot

encoded = []
for idx, catecory in enumerate(categories):
    celData = one_hot_encode(catecory_to_index[catecory], len(categories))
    encoded.append(celData)

print(encoded)