"""
キャプションデータの前処理

データを学習できる形に整える必要がある。
"""

import nltk
import pickle
from collections import Counter
from pycocotools.coco import COCO
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

class Vocabulary(object):
    def __init__(self): # データの初期化。コンストラクタと呼ばれる初期化のための関数である。
        self.word2idx = {} # self.word2idx辞書を初期化する
        self.idx2word = {} # self.idx2word辞書を初期化する
        self.idx = 0 # self.idx変数を初期化する

    def add_word(self, word):
        if not word in self.word2idx: # 引数wordがself.word2idxのkeyにない場合に実行する
            self.word2idx[word] = self.idx # 辞書self.word2idxにkey=word、value=self.idxの要素を加えている
            self.idx2word[self.idx] = word # 辞書self.idx2wordにkey=self.idx、value=wordの要素を加えている
            self.idx += 1 # 変数self.idxに1を加えている

    def __call__(self, word): # 「__call__(self)」メソッドは、メソッドを指定しないで実行できるメソッドである
        if not word in self.word2idx: # 引数wordがself.word2idxのkeyにない場合は、「<unk>」を返却する
            return self.word2idx['<unk>']
        return self.word2idx[word] # self.word2idxにkey=word、value=self.idxを返却している

    def __len__(self): # 「__len__(self)」メソッドはlen()に対して返却するメソッド
        return len(self.word2idx) # len(self.word2idx)を返却している

def make_vocab(json, threshold):
    coco = COCO(json) # COCO APIの初期化をする
    counter = Counter() # collections.Counterの初期化をしている
    ids = coco.anns.keys() # coco.annsはjsonデータのキャプション部分であり、coco.anns.keys()はそのkey値のことである
    for i, id in enumerate(ids): # coco.annsのkey値を順次取り出すループ
        caption = str(coco.anns[id]['caption']) # key値からキャプションを取り出す
        tokens = nltk.tokenize.word_tokenize(caption.lower()) # キャプションをトークン化する
        counter.update(tokens) # Counterインスタンスは引数無しで作成してupdate()メソッドで値を追加する

        if (i+1) % 1000 == 0: # ループ1000回ごとに進捗状況を出力している。
            print("[{}/{}] Tokenized captions.".format(i+1, len(ids)))

    words = [word for word, cnt in counter.items() if cnt >= threshold] # 出現回数がthreshold以上のトークンのみ取り出してwordsにいれる

    vocab = Vocabulary() # Vocablaryクラスをインスタンス化したvocabに4つの言葉を追加している
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    for i, word in enumerate(words): # vocabにwordsを追加している
        vocab.add_word(word)
    return vocab

vocab = make_vocab(json='./data/annotations/captions_val2014.json', threshold=4) # make_vocabメソッドに「json='./dataannotations/captions_val2014.json'」と「threshold=4」を与えて、その戻り値をvocabに与える
vocab_path = './data/vocab.pkl' # vocabを保存するvocab_pathを定義している
with open(vocab_path, 'wb') as f: # vocabをpickleファイル形式で保存している
    pickle.dump(vocab, f)
print("Total vocabulary size: {}".format(len(vocab)))
print("Saved vocabulary wrapper to '{}'".format(vocab_path))
