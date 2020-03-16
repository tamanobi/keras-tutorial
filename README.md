# Anaconda3 へのパスを通す

conda init を実行しましょう。 zshの人はzsh。bashの人はbash

# NotWritableError が出る人へ

インストールディレクトリの権限が不足している恐れがあります。 https://qiita.com/richardf/items/5106e2000c8fe71d10 で解決を図ってください

# 仮想環境を作る

```
$ conda create --name kerastutorial python=3.6
$ conda activate kerastutorial
```
