import nltk
import pickle
import os
import numpy as np
from PIL import Image
from collections import Counter
from pycocotools.coco import COCO
import glob
import matplotlib.pyplot as plt
import nltk
nltk.download('punkt')
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
import torch.utils.data as data
from vocabulary import Vocabulary


"""
データセットの作成

torchvisionには主要なDatasetがすでに用意されている。
しかし、ここでは画像のデータとキャプションのデータの両方を揃えるために新規で作成する。
"""
class CocoDataset(data.Dataset):
    def __init__(self, root, json, vocab, transform=None) -> None:
        self.root = root # 画像のディレクトリを渡す
        self.coco = COCO(json) # cocoキャプションファイル名（json）を渡す。
        self.ids = list(self.coco.anns.keys()) # ボキャブラリークラスのキー値のリストを渡す。
        self.vocab = vocab # ボキャブラリークラスのインスタンスを渡す。
        self.transform = transform # 画像変換するfransformを渡す。
    
    def __getitem__(self, index): # idexと同じ番号のdataを返す関数。
        coco = self.coco # self.cocoをcocoに渡している
        vocab = self.vocab # self.vocabをvocabに渡している
        ann_id = self.ids[index] # self.ids[index]をann_idに渡している
        caption = coco.anns[ann_id]['caption'] # coco.anns[ann_id]['caption']をcaptionに与えている
        img_id = coco.anns[ann_id]['image_id'] # coco.anns[ann_id]['image_id']をimg_idに与えている
        path = coco.loadImgs(img_id)[0]['file_name'] # coco.loadImgs(img_id)[0]['file_name']をpathに与えている

        image = Image.open(os.path.join(self.root, path)).convert('RGB') # 画像ファイルを読み込んで「RGB」モードに変換している。画像モードには、「RGBA」等もある。
        if self.transform is not None: # self.transformに何も設定されていない場合を除いては、画像にself.framsformを実行して変形する。リサイズ、回転、線形変換等ができる。
            image = self.transform(image)
        
        """
        日本語を文節単位で分かち書きする

        原文        | 今日はよく寝ました
        単語分かち書き| 今日/は/よく/寝/まし/た
        """
        tokens = nltk.tokenize.word_tokenize(str(caption).lower()) # nltk.tokenize.wordを用いて文章captionを各単語に分書している
        caption = [] # リストcaptionを初期化する
        caption.append(vocab('<start>')) # captionの先頭に「<start>」のID番号を与えている
        caption.extend([vocab(token) for token in tokens]) # captionに文章captionの分かち書きした各単語のIDを与えている
        caption.append(vocab('<end>')) # captionの最後に「<end>」のID番号を与えている
        target = torch.Tensor(caption) # リストcaptionをtorch.Tensorで「Tensor」型に変換している。それをtargetに与えている
        return image, target
    
    def __len__(self): # 「len(dataset)」とすると実行されて、全データの個数を返す
        return len(self.ids) # 全キャプション数を呼び出し元に返却している

"""
データローダーの作成

データローダーとは、datasetsからバッチ毎にデータを取り出すことを目的に使われる。
基本的に、torch.utils.data.DataLoaderを使用する。
つまり、datasetsはデータ全てのリストであり、Dataloaderはそのdatasetsの中身をバッチごとに固めた集合のこと。
"""
def collate_fn(data): # 引数dataには、画像データであるimageとID番号「Tensor」で構成されたキャプションが含まれる
    data.sort(key=lambda x: len(x[1]), reverse=True) # dataをキャプションの長さでソートしている。デフォルトは昇順なので、引数reverseをTrueにして降順でソートする
    """
    転置とは

    1 2 -> 1 3
    3 4 -> 2 4
    のように行と列を入れ替えることを転置という。
    この場合、「*data」を使うことで、dataの最初の要素がimagesのリスト、2つ目の要素がcaptionsのリストして分離される。
    """
    images, captions = zip(*data) # Pythonの組み込み関数zip()を使って、dataからimageとcaptionsを同時に取り出している。ここでは、「zip(*data)」と記述してあるとおり、dataに「*」がついている。これは配列を転置するという意味

    images = torch.stack(images, 0) # 画像データを一つの配列に埋め込んでいる。画像データは3じげん（縦、横、RGB）なので、配列は4次元になる

    lengths = [len(cap) for cap in captions] # キャプションの長さの配列lengthsを作成している
    targets = torch.zeros(len(captions), max(lengths)).long() # キャプション数✖️キャプション最大長さの2次元配列targetsを初期化している
    for i, cap in enumerate(captions): # キャプションを順次取り出す
        end = lengths[i] # キャプションの長さを取り出している
        targets[i, :end] = cap[:end] # キャプションを一つの配列に詰め込んでいる
    return images, targets, lengths # 画像を埋め込んだ配列images、キャプションを詰め込んだtargets、キャプションの長さを詰め込んだ配列lengthsを呼び出し元に返却している

def get_loader(root, json, vocab, transform, batch_size, shuffle, num_workers):
    coco = CocoDataset( # CocoDatasetクラスをインスタンス化している
        root=root,
        json=json,
        vocab=vocab,
        transform=transform
    )

    data_loader = torch.utils.data.DataLoader( # torch.utils.data.DataLoaderの戻り値をdata_loaderに割り当てている
        dataset=coco,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    return data_loader

"""
ニューラルネットワークモデルの作成

Seq2Seq（sequence to sequence）モデルを応用する。Seq2Seqは、EncoderとDecoderを備えたEncoder-Decoderモデルを使って、
系列データを別の系列データに変換するモデルのことを指す。Seq2Seqで翻訳したり、対話モデルを作ったりすることが可能になる。

Encoderとは、InputData（画像、テキスト、音声、動画等）を何かしらの（固定長）特徴ベクトルに変換する機構のことをいう。
Decoderとは、Encoderでエンコードされた特徴ベクトルをデコードして何か新しいデータを生む機構のことをいう。
OutputDataはInputDataと同じデータ形式である必要はなく、画像、テキスト、音声などいろいろ利用可能。

Encoder-DecoderモデルとはEncoderとDecoderを繋げることによって作られる。
これはいわゆる生成系のモデルであり、画像をテキストにしたり、音声からテキストを生成したり、日本語から英語に変換したりといろいろなことに利用できる。
"""

# Encoderモデル
class Encoder(nn.Module):
    def __init__(self, embed_size): # 「__init__(self, embed_size)」メソッドの定義。インスタンスの初期化を行う
        super(Encoder, self).__init__() # 親クラスの「Encoder」の「__init__()」を継承する
        resnet = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1) # resnet152の学習済みモデルをresnetに設定している
        modules = list(resnet.children())[:-1] # resnetの全ての層をmodulesに与えている
        self.resnet = nn.Sequential(*modules) # modulesの全ての層をnn.Sequentialで結合してself.resnetに与えている
        self.linear = nn.Linear(resnet.fc.in_features, embed_size) # 入力サイズ「resnet.fc.in_features」、出力サイズ「embed_size」のLinear層をself.linearに定義している
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01) # 1次元のバッチ正規化を設定している
    
    def forward(self, images): # forwardメソッドを定義
        with torch.no_grad(): # 「torch.no_grad()」で勾配初期化した状態で、self.resnet(images)の出力をfeaturesに割り当てている
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1) # featuresを二次元配列にreshapeしている
        features = self.bn(self.linear(features)) # featuresにself.linearとself.bnを定義
        return features

# Decoderモデル
class Decoder(nn.Module): # Decoderクラスを定義
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20): # インスタンスの初期化を行う
        super(Decoder, self).__init__() # 親クラスの「Decoder」の「__init__()」を継承することを述べている

        """
        「nn.Embedding」層の設定をしている。これはいわゆる埋め込み層というもので、tokenIDを埋め込みベクトルに変換している。
        入力サイズはvocab_sizeで、出力サイズはembed_sizeである。

        埋め込み層（Embedding Layer）とは？
        ニューラルネットワークにおいて単語インデックスをベクトルにマッピングする層である。
        深層学習の長短期記憶（LSTM）ネットワークやTransformerのトークン埋め込み層などに使用される。
        """
        self.embed = nn.Embedding(vocab_size, embed_size)

        """
        「nn.LSTM」層を設定している。ここでは、入力サイズがembed_size、隠れ層のサイズがhidden_size、隠れ層の総数がnum_layers、
        「batch_first=True」でバッチの次元を0次元に設定している。
        """
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size) # nn.Linear層の設定をしている。入力サイズがhidden_sizeで、出力サイズがvocab_sizeである
        self.max_seg_length = max_seq_length # self.max_seg_lengthを設定している
    
    def forward(self, features, captions, lengths): # forwardメソッドを設定している
        embeddings = self.embed(captions) # self.embed層にcaptionsを入力データとして与えている

        """
        torch.cat((features.unsqueeze(1), embeddings), 1)で、まず画像の特徴量featuresの1次元目を生成して埋め込みベクトルembeddingsと
        次元数を合わせた上で、直列に結合する。結合する際は1次元目方向に結合していく。
        """
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)

        """
        pack_padded_sequence（embeddings, lengths, batch_first=True）で、embeddingsの計算が最小になるように配列を作成する。
        ここで、「batch_first=True」としているので、0次元がバッチサイズになる。
        """
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed) # self.lstm層にpackedを入力する。出力がhiddensに入る。
        outputs = self.linear(hiddens[0]) # pack_padded_sequenceを使用しているので、hiddensの[0]を取って、hiddens[0]をself.linear層に入れて、その出力がoutputsに入る
        return outputs
    
    def predict(self, features, states=None): # predictメソッドを定義している。入力は画像の特徴量featuresと初回はNoneのstatesである
        predicted_ids = [] # predicted_idsを初期化している
        inputs = features.unsqueeze(1) # 画像の特徴量featuresの1次元目を生成して初回の入力値をinputsに入れる
        for _ in range(self.max_seg_length): # self.max_seg_lengthの数のループを作成
            hiddens, states = self.lstm(inputs, states) # self.lstm層にinputsとstatesを入力する。出力がhiddensとstatesに入る。ここで、hiddensがLSTMの出力で、statesは隠れ層の記憶状態である
            outputs = self.linear(hiddens.squeeze(1)) # LSTMの出力hiddensの1次元目を省略してself.linear層に入れて、その出力がoutputsに入る
            _, predicted = outputs.max(1) # 「outputs.max(1)」は最大値とその要素位置の2つを返すので、その最大値を_で、その要素の位置をpredictedで受け取る
            predicted_ids.append(predicted) # predictedの値をpredicted_idsに追加している
            inputs = self.embed(predicted) # predictedの埋め込みベクトルをinputsに入れている
            inputs = inputs.unsqueeze(1) # inputsの一次元目を追加する
        predicted_ids = torch.stack(predicted_ids, 1) # predicted_idsを1次元方向にスタック結合する
        return predicted_ids


if __name__ == "__main__":
    # 学習
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_path = 'models/' # モデルを保存するためのパスを定義
    crop_size = 224 # 画像ファイルから画像データを切り抜くサイズを指定
    vocab_path = 'data/vocab.pkl' # vocab.pklファイルパス
    image_dir = 'data/resized2014' # リサイズ後の画像ファイルのパスを定義
    caption_path = 'data/annotations/captions_train2014.json' # json形式のキャプションファイルのパスを定義
    log_step = 10 # ログを出力する間隔（バッチ回数）を定義
    save_step = 1000 # 学習中および学習後にモデルを保存する間隔（バッチ回数）を定義
    embed_size = 10 # 埋め込みベクトルの次元数を定義
    hidden_size = 512 # LSTMの隠れ層の次元数をhidden_sizeに定義
    num_layers = 1 # LSTMの隠れ層の層数をnum_layersに定義
    num_epochs = 10 # echo単位の学習回数を定義
    batch_size = 128 # 1バッチあたりのデータ数を定義
    num_workers = 2 # 同時並行処理する数を定義
    learning_rate = 0.001 # 学習率を定義

    if not os.path.exists(model_path): # model_pathにディレクトリがなければ作成する
        os.makedirs(model_path)

    transform = transforms.Compose([ # torchvision.transforms.Composeでデータの前処理の設定を行う
        transforms.RandomCrop(crop_size), # ランダムにcrop_size（縦・横）に切り抜く
        transforms.RandomHorizontalFlip(), # ランダムに左右反転させる
        transforms.ToTensor(), # numpyやPIL Image(W,H,C)をTensor(C,W,H)に変換し、データ型もuint8からfloat32に変換し、データの範囲[0, 255]を[0, 1.0]に変換する
        # ↓はデータの正規化を行なっている。ここでの正規化は画像の各チャンネルの平均及び標準偏差を与える
        # 例として、torchvisionで提供されている学習済みのモデルを使う場合は、ImageNetデータセットで学習したモデルなので、その平均及び標準偏差を与える
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.299, 0.224, 0.225])
    ])

    with open(vocab_path, 'rb') as f: # vocab_pathに設定したvocab.pklファイルを読み込む
        vocab = pickle.load(f)

    data_loader = get_loader(image_dir, caption_path, vocab, transform, batch_size, shuffle=True, num_workers=num_workers)

    encoder = Encoder(embed_size).to(device) # to.(device)はtoメソッドでテンソルを「device」に転送するという意味
    decoder = Decoder(embed_size, hidden_size, len(vocab), num_layers).to(device)

    criterion = nn.CrossEntropyLoss() # 損失関数の定義
    # decoderとencoderの最適化パラメータを集めてparamsに設定している
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate) # 各種パラメータをオプティマイザーに設定

    # データの数をカウントしてtotal_stepに設定する
    total_step = len(data_loader)

    for epoch in range(num_epochs):
        for i, (images, captions, lengths) in enumerate(data_loader):

            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()

            if i % log_step == 0:
                print(f"Epoch [{epoch}/{num_epochs}], Step [{i}/{total_step}], Loss: {loss.item():.4f}, Perplexity: {np.exp(loss.item()):5.4f}")
            
            if (i+1) % save_step == 0:
                torch.save(decoder.state_dict(), os.path.join(
                    model_path, f"decoder-{epoch+1}-{i+1}.ckpt"
                ))
                torch.save(encoder.state_dict(), os.path.join(
                    model_path, f"encoder-{epoch+1}-{i+1}.ckpt"
                ))