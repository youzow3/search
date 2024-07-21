# search.py

perplexity.aiのような検索エンジンをローカル環境、自前の言語モデルを用いて使うことを目指したPythonスクリプトです。

## 使い方

llama.cppのサーバを開始した後：

```bash
python3 search.py
```

また、*bootstrap.py*を使うことで依存関係の自動インストール、ビルドが可能です。必要なツール（コンパイラなど）がインストールされているか確認してから実行することをおすすめします。また、サーバの実行スクリプトも生成されますがOS、ビルドシステムによって生成されない場合があります。

## 設計

### 概要

1. ユーザからの入力の受け取り

2. 検索計画の作成

3. 必要な情報の取得

4. 検索結果の要約

### 計画 [ Application.plan() ]

ユーザの入力を読み込み、AIにYes/Noの質問を投げることでプランを組むように実装しています。質問と行動は以下の通りです。

| 質問                                         | Yesの場合            |
| :------------------------------------------- | :------------------: |
| 知らない用語、言葉はあるか                   | 検索 &rarr; 動的計画 |
| より正確な検索のためにユーザとの対話は必要か | 動的計画             |
| 正確な情報を提供することが現時点でできるか   | 要約                 |
| 検索中、検索後にユーザとの対話は必要か       | 検索 &rarr; 動的計画 |

### 検索 [ Application.search() ]

検索語の生成からウェブサイトから要点を引き抜く作業まで行います。実際の動きは以下のとおりです。

1. 検索語の生成

2. 集めるべき情報のリストを作成

3. googlesearch.search()より検索結果の取得

4. analyze\_is\_useful() を呼び出し実際にウェブサイトを読むかを決定

5. analyze() により要点のリストを取得

6. 検索の途中切り上げの判断

#### analyze\_is\_useful() について

この関数ではAIにウェブサイトが使いやすそうかどうかをYes/Noで聞いて判断をしています。AI自身のバイアスなどにより全く関係ない場合でもYesとしたりその逆のことをする可能性があります。

#### analyze() について

ウェブサイトのデータをMarkdown化してAIが有用と判断した内容をリストにして返します。analyze\_is\_useful() と同様に、不必要な情報を含めたりする可能性があります。また、ウェブサイト自体が誤った情報を書いている場合にそれをそのまま使ってしまうことも考えられます。

#### 検索の途中切り上げについて

ラップトップなどでの運用を考える際、明らかに十分な情報が集まっているにも関わらず情報収集を行うことを避けるため、以下の２つの条件分岐を用意して途中で検索を切り上げるようにしています。

* 検索結果を与えるのに十分な情報が集まっているか

* 現在の検索語について、十分な情報が集まっているか

### 動的計画 [ Application.plan\_dynamic() ]

ユーザの質問があまりにも抽象的すぎる・分かりづらいなどの場合にAIが的外れな検索・検索結果を与えることを避けるためにAIからユーザに質問を与えて、計画を再度行うというものです。実際の動きは以下の通りです。

1. 質問の発行

2. 計画 (Application.plan() の呼び出し)

検索を行ったあとの動的計画の場合、質問は検索によって得た要点を参考にしながら生成されるようになっています。

### 要約 [ Application.summarize() ]

集めた情報から最終結果として要約します。最初の計画を行う時点で「正確な情報を提供することができるか」にYesを返した場合、インターネットは一切使わずにAIの知識のみで書かれることになります。

### AIの生成アルゴリズムに関して

このスクリプトではAIの応答に依存している箇所が多々存在します。それらの部分でエラーを起こさないため基本的にはXML形式で応答をさせるようにプロンプトを書いています。また、生成されたXMLが不正でないことを保証できるようなコードになっています。不正かどうかのチェックは*verify_xml()*関数によって返される関数によって行われます。詳細な実装は*generate_xml()*と、*xml_instruction_and_verify()*を参考にしてください。

# 参考

[llama.cpp](https://github.com/ggerganov/llama.cpp)
