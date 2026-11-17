# LLM の被験体

:gb: [English](../en/llm-subjects.md)

言語モデルを動物の代わりに実験箱に入れる方法は 2 つあります。`claude` コマンドを通した Claude のモデルと、出力のスコアから行動の確率を読む手元のモデルです。パッケージは [README](../../README.ja.md) のとおりに入れてください。下の例はこのリポジトリを clone したときの `examples/` にあります。このページが extra (`viz`) の名前を挙げているところでは、README でそれに挙げているライブラリを入れてください。

## API キーなしで本物の LLM を被験体にする

`ClaudeCLIClient` は `claude` コマンド (Claude Code) を 1 プロセス常駐させ、各ステップを 1 つのユーザーターンとして送ります。プロセスが持つ会話がそのまま被験体の文脈内の履歴になります。ログイン済みの Claude サブスクリプションの枠で動き、API キーも extra も要りません (`PATH` に `claude` があり、一度 `claude` を起動してログインしてあれば足ります)。`StreamingLLMSubject` は最新のステップだけを送り (`Result: Reward` / `Stimulus: Light ON`)、返事から行動名を読み取ります。行動名が 1 つに定まらない返事は `Nothing` として出し、`invalid_count` に数えます (元の返事は `history` に残ります)。

```python
from ai_operant_box.agent.api_client import ClaudeCLIClient
from ai_operant_box.agent.llm_agent import SYSTEM_PROMPT, StreamingLLMSubject
from ai_operant_box.chamber.environment import OperantBox
from ai_operant_box.chamber.schedule import ContinuousReinforcement

with ClaudeCLIClient(model="haiku", system_prompt=SYSTEM_PROMPT) as client:
    subject = StreamingLLMSubject(client)
    box = OperantBox(subject=subject, schedule=ContinuousReinforcement())
    for _ in range(20):
        box.step(1.0)
print(len(box.records), subject.invalid_count, client.latencies[-1])
```

`examples/claude_cli_crf.py` は同じセッションをコマンドラインから走らせ、累積記録 (PNG。`viz` extra が要ります) と、ステップごとの往復時間を含む JSON の要約を書き出します。

system prompt は Claude Code の既定の prompt を置き換えます (`--system-prompt`)。`--append-system-prompt` ではコーディング支援用の prompt が残り、小さいモデルは役割を断ります。ツール・MCP サーバ・設定ファイルとその hook・セッションの保存はすべて切り、プロセスは空の一時ディレクトリで動くので、プロジェクトの `CLAUDE.md` は被験体に届きません。ただし CLI は system prompt の前に 1 文の自己紹介を、また作業ディレクトリ・モデル名・日付の短い注記を加えます。これはどのセッションでも同じです。

CRF でのステップあたりの往復時間の実測 (1 台のマシン、2026 年 10 月):

| モデルの別名 | ステップ数 | 平均 秒 / ステップ | 最初のステップ 秒 |
|---|---|---|---|
| `haiku` | 20 | 1.1〜1.4 | 約 4 |
| `haiku` | 150 | 0.8〜1.1 | 約 3〜4 |
| `sonnet` | 20 | 1.9 | 約 2.5 |
| `opus` | 20 | 2.6 | 約 3.9 |

`haiku` では 150 ステップまでステップあたりの時間は伸びませんでした。見積もりには `haiku` 1 ステップ 1.4 秒、`sonnet` 2.5 秒、`opus` 3.5 秒を使います。`haiku` の 600 ステップのセッションは約 14 分かかります。測定時点で別名 `haiku` は `claude-haiku-4-5-20251001` に解決されました。別名が指すモデルは CLI が決めます。

**利用枠への影響。** 呼び出しは API の請求ではなく、サブスクリプションの利用枠を消費します。`client.last_result["total_cost_usd"]` はその会話の API 換算額で、枠の消費量の目安になります。`haiku` 150 ステップで約 0.24 USD 相当でした。報告される 7 日間の利用率 (`client.rate_limit_info`) は、`haiku` 数百ステップの前後で小数 2 桁の精度では動きませんでした。`sonnet` と `opus` はステップあたりの消費が大きくなります。`ClaudeCLIClient.generate()` は `LLMSubject` の状態を持たない `generate(model, system, messages)` 互換の口を残していますが、毎ステップ新しいプロセスを起動して全履歴を送り直すので遅く、既定にはしていません。

## 手元のモデルで反応確率を直接読む

ホストされたチャット API は文字列しか返さないので、モデルがレバーを押す確率は、返事を何度も標本抽出して推定するしかありません。llama.cpp で動かす手元の GGUF モデルは次のトークンの logits を出すので、`ai_operant_box.agent.local_logprob` は各ステップで各行動の確率を標本抽出なしに直接計算します (`llama-cpp-python` が要ります)。

- `LocalLogprobClient` は ChatML の会話 (Qwen をはじめ多くの指示調整済み GGUF モデルのチャット形式) を保持します。`score(text)` はユーザーのターンを足し、各行動について、それをアシスタントの返事としたときのトークン列全体の対数確率 (各トークンの確率の積)、先頭トークンだけの対数確率、行動の集合の中で正規化し直した確率を返します。`temperature` は softmax の前に各トークンの logits を割ります。
- `LlamaCppBackend` は各 prompt のうち前のステップから新しく増えた部分だけを評価するので、600 ステップのセッションでも速く動きます。
- `LocalLogprobSubject` は他の被験体と同じく `OperantBox` で走ります。実験箱がスケジュールを当てるには行動が 1 つ発せられる必要があり、それは計算した分布から種付きで引いた行動 (`policy="sample"`) か、最も確率の高い行動 (`policy="greedy"`) です。`history` の各要素は、そのステップで読んだ確率 `p` を持ちます。

```bash
pip install llama-cpp-python matplotlib
python examples/local_logprob_fi.py --model path/to/model.gguf --interval 30 --steps 600 --out-dir runs/fi30
```

このスクリプトは P(Lever Press) をセッション時間に沿って、また前の強化子からの経過時間ごとの平均として描き、全ステップの確率を JSON に書き出します。Qwen2.5-1.5B-Instruct (Q4_K_M) を Apple silicon の 1 台で動かすと、FI 30 の 600 ステップに約 7 分かかりました (1 ステップ平均 0.7 秒。prompt は約 13,000 トークンまで伸びました)。このトークナイザーでは `Lever Press` と `Grooming` の先頭トークンが 1 文字 (`L`、`G`) なので、正規化し直すのは先頭トークンの確率ではなくトークン列全体の確率です。

再利用した prefix の結果を最初から評価し直した結果と突き合わせるテストは、環境変数 `AOB_GGUF` が GGUF ファイルを指すときだけ走ります。他のテストは偽のバックエンドを使い、モデルは要りません。
