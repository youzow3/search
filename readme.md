# search

perplexity.aiのような検索エンジンを目指したスクリプトです。

## 使い方

依存関係のインストールなど

```bash
python3 bootstrap.py
```

パラメータは--helpを使って確認してください。

### search.pyの実行について

このプロジェクトではllama.cppのサーバを使用します。bootstrap.py を用いた場合には基本的にllama-server.sh、llama-server.batが生成されるはずですのでそれを最初に起動してください。

```bash
./llama-server.sh
```

```cmd
./llama-server.bat
```

このあとにsearch.pyを実行するようにしてください。そうでないと検索文を入力したときにサーバに接続できない旨が表示されて強制終了します。
サーバは適宜バックグラウンドや、べつのシェルで実行するようにしてください。

```bash
python3 search.py
```

