# Keras入門

## Anaconda3 へのパスを通す

conda init を実行しましょう。 zshの人はzsh。bashの人はbash

## NotWritableError が出る人へ

インストールディレクトリの権限が不足している恐れがあります。 https://qiita.com/richardf/items/5106e2000c8fe71d10 で解決を図ってください

## 仮想環境を作る

```
$ conda create --name kerastutorial python=3.6
$ conda activate kerastutorial
```

## Anaconda Navigator のセットアップ

Anaconda Navigator を起動し、 Application on を kerastutorial に変更する。

![](images/application_on.png)

## JupyterLab のインストール

Anaconda Navigator を使って JupyterLab を起動する

![](images/jupyterlab.png)

http://localhost:8888/lab へ接続する

![](images/notebook.png)

## Keras の実行確認

https://keras.io/#getting-started-30-seconds-to-keras のコードを実行する。

```
from keras.models import Sequential

model = Sequential()
```

Shift+Enter を押してエラーが出なかったらOK。もし、モジュールがないと言われた場合は、 Anaconda Navigator の Application on が kerastutorial になっているか確認する。
