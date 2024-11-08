"""
画像のリサイズ

学習に使用する画像のデータはtrain2014ディレクトリにあるが、
サイズがバラバラであるため実際に使用するときの画像データのサイズに揃える
"""

import os
from PIL import Image

def resize_image(image, size): # 画像をリサイズするメソッド
    # Pillowのresizeメソッドを用いてリサイズした画像を返している。引数には、リサンプリングする際に使われるフィルターを指定する。省略した場合はデフォルトでNEARESTが使われる。
    # ここでは、ANTIALIASが選択されており、アンチエイリアス処理しますということで、アンチエイリアスを掛けると画像変換時のギザギザがなくなる。
    return image.resize((size,size), Image.ANTIALIAS)

def resize_images(image_dir, output_dir, size): # resize_imageメソッドを使って画像をリサイズするメソッド「resize_images」を定義している
    if not os.path.exists(output_dir): # 出力ディレクトリがない場合は出力ディレクトリを作成する
        os.makedirs(output_dir)

    images = os.listdir(image_dir) # os.listdirでファイル・ディレクトリの一覧を取得してimagesに入れている
    num_images = len(images) # ファイル数を取得してnum_imagesにいれている
    for i, image in enumerate(images): # 画像ファイルを一枚ずつ「resize_image」メソッドを使ってリサイズして出力ディレクトリに保存する
        with open(os.path.join(image_dir, image), 'r+b') as f: # 
            with Image.open(f) as img:
                img = resize_image(img, size)
                img.save(os.path.join(output_dir, image), img.format)
        if (i+1) % 100 == 0: # リサイズ処理の進捗状況を出力している
            print ("[{}/{}] Resized the images and saved into '{}'."
                   .format(i+1, num_images, output_dir))

image_dir = './data/train2014/' # 画像のディレクトリ名をimage_dirに与えている
output_dir = './data/resized2014/' # リサイズした画像の保存先をoutput_dirに与えている
image_size = 256 # リサイズ後の画像サイズをimage_sizeに与えている
resize_images(image_dir, output_dir, image_size)
