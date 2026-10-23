# ai-operant-box

:gb: [English README](README.md)

LLM-as-subject シミュレータ。大規模言語モデルがオペラント箱の中の
仮想個体として振る舞い、各試行で弁別刺激 (SD) を知覚し、反応 (R) を
発し、結果 (SR) を受け取る — オペラント条件づけの三項随伴性。

## 概要

本パッケージは以下を提供する:

- **Subjects (`ai_operant_box.agent`)** — 抽象基底の `Subject`、決定論
  的数値エージェント (`InductionAgent`, `InductionSubject`,
  `ScallopingAgent`)、そして行動方針が Claude を通した文脈内履歴で
  与えられる LLM バックエンドの `LLMSubject`。
- **Chambers (`ai_operant_box.chamber`)** — 単一被験体の `OperantBox`
  と複数被験体の `LeaderFollowerBox`。それぞれ離散時間の SD-R-SR
  ループを駆動する。
- **Schedules (`ai_operant_box.chamber.schedule`,
  `ai_operant_box.schedules`)** — CRF / FI に加え、Fleshler-Hoffman
  変動 / ランダムの比率 / 間隔スケジュール、および相互依存二者
  スケジュール。
- **Analytics (`ai_operant_box.analytics`)** — 累積記録器、複数
  セッション比較、スキャロップ・モデル・フィッタ（matplotlib と
  SciPy は analytics 関数内で遅延インポート; インポート時には不要）。
- **CLI (`ai_operant_box.cli`)** — smoke test で使われる最小限の
  エントリポイント。

## インストール

clone した場合、本体に必要なのは NumPy だけ:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e .
```

数値エージェント・チャンバー・スケジュール・`ai-operant-box` コマンドは
これで動く。開発用 extra はテスト道具と、任意の OperantKit ライブラリ
（session-analyzer、experiment-core、session-recorder）を加える。
それらがまだパッケージインデックスから入手できない場合は、wheel を
1 つのディレクトリにビルドまたはダウンロードし、pip にそこを指定する:

```bash
.venv/bin/python -m pip install --find-links <wheel-dir> -e ".[dev]"
```

`claude` コマンド経由で本物の LLM を被験体にする場合は extra は要らない（「API キーなしで本物の LLM を被験体にする」を参照）。代わりに Anthropic API を使うには:

```bash
.venv/bin/python -m pip install -e ".[dev,llm]"
export ANTHROPIC_API_KEY="..."
```

## 30 秒で動く例

直前の強化子からの経過時間の 2 乗に比例して反応確率が上がる数値被験体（`P(R) = min(1, k * t**2)`）を、FI 30 秒のもとで 1 秒刻み 600 ステップ走らせる。LLM も API キーも要らない:

```python
import random

from ai_operant_box.agent.scallop_agent import ScallopingAgent
from ai_operant_box.chamber.environment import OperantBox
from ai_operant_box.chamber.schedule import FixedInterval

random.seed(42)  # ScallopingAgent はモジュール共通の乱数を使う
agent = ScallopingAgent(k=0.005)
box = OperantBox(subject=agent, schedule=FixedInterval(30.0))

for _ in range(600):  # FI 30 秒を 1 秒刻みで 600 ステップ
    agent.update_time(1.0)
    box.step(1.0)

responses = len(box.records)
reinforcers = sum(r.is_reinforced for r in box.records)
print(f"responses={responses} reinforcers={reinforcers}")
# responses=422 reinforcers=20

# 30 秒の間隔を 10 等分した各区間の反応数（セッション全体で合算）
bins = [0] * 10
last_sr = 0.0
for r in box.records:
    bins[min(9, int((r.timestamp - last_sr) / 3.0))] += 1
    if r.is_reinforced:
        last_sr = r.timestamp
print(bins)
# [1, 5, 12, 30, 54, 60, 60, 60, 60, 80]
```

各間隔の前半では反応がまれで、終わりに向かって増える（FI スキャロップ）。最後の区間には各間隔の強化された反応も入る。時計は模擬なので 1 秒もかからずに終わり、毎回同じ数値を出す。

API キーなしで同じチャンバーに言語モデルを入れるには次の節を参照。API キーを使う場合は `llm` extra を入れ、`ai_operant_box.agent.llm_agent.LLMSubject` を `AnthropicClient` と組み合わせる。事象を OKL v1 のログとして残すには `canonical` extra を入れ、`ai_operant_box.async_simulator.canonical_sink.mirror_events_to_sink` で [session-recorder](https://github.com/OperantKit/session-recorder) の sink に渡す。

## API キーなしで本物の LLM を被験体にする

`ClaudeCLIClient` は `claude` コマンド（Claude Code）を 1 プロセス常駐させ、各ステップを 1 つのユーザーターンとして送る。プロセスが持つ会話がそのまま被験体の文脈内の履歴になる。ログイン済みの Claude サブスクリプションの枠で動き、API キーも extra も要らない（`PATH` に `claude` があり、一度 `claude` を起動してログインしてあればよい）。`StreamingLLMSubject` は最新のステップだけを送り（`Result: Reward` / `Stimulus: Light ON`）、応答から行動名を読み取る。行動名が 1 つに定まらない応答は `Nothing` として出し、`invalid_count` に数える（元の応答は `history` に残る）。

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

`examples/claude_cli_crf.py` は同じセッションをコマンドラインから走らせ、累積記録（PNG。`viz` extra が要る）と、ステップごとの往復時間を含む JSON の要約を書き出す。

system prompt は Claude Code の既定の prompt を置き換える（`--system-prompt`）。`--append-system-prompt` ではコーディング支援用の prompt が残り、小さいモデルは役割を断る。ツール・MCP サーバ・設定ファイルとその hook・セッションの保存はすべて切り、プロセスは空の一時ディレクトリで動くので、プロジェクトの `CLAUDE.md` は被験体に届かない。ただし CLI は system prompt の前に 1 文の自己紹介を、また作業ディレクトリ・モデル名・日付の短い注記を加える。これはどのセッションでも同じである。

CRF でのステップあたりの往復時間の実測（1 台のマシン、2026 年 10 月）:

| モデルの別名 | ステップ数 | 平均 秒 / ステップ | 最初のステップ 秒 |
|---|---|---|---|
| `haiku` | 20 | 1.1〜1.4 | 約 4 |
| `haiku` | 150 | 0.8〜1.1 | 約 3〜4 |
| `sonnet` | 20 | 1.9 | 約 2.5 |
| `opus` | 20 | 2.6 | 約 3.9 |

`haiku` では 150 ステップまでステップあたりの時間は伸びなかった。見積もりには `haiku` 1 ステップ 1.4 秒、`sonnet` 2.5 秒、`opus` 3.5 秒を使う。`haiku` の 600 ステップのセッションは約 14 分かかる。測定時点で別名 `haiku` は `claude-haiku-4-5-20251001` に解決された。別名が指すモデルは CLI が決める。

**利用枠への影響。** 呼び出しは API の請求ではなく、サブスクリプションの利用枠を消費する。`client.last_result["total_cost_usd"]` はその会話の API 換算額で、枠の消費量の目安になる。`haiku` 150 ステップで約 0.24 USD 相当だった。報告される 7 日間の利用率（`client.rate_limit_info`）は、`haiku` 数百ステップの前後で小数 2 桁の精度では動かなかった。`sonnet` と `opus` はステップあたりの消費が大きい。`ClaudeCLIClient.generate()` は `LLMSubject` の状態を持たない `generate(model, system, messages)` 互換の口を残しているが、毎ステップ新しいプロセスを起動して全履歴を送り直すので遅く、既定にはしない。

## 局所モデルで反応確率を直接読む

ホストされたチャット API は文字列しか返さないので、モデルがレバーを押す確率は、応答を何度も標本抽出して推定するしかない。llama.cpp で動かす局所の GGUF モデルは次のトークンの logits を出すので、`ai_operant_box.agent.local_logprob` は各ステップで各行動の確率を標本抽出なしに直接計算する（`local` extra: `llama-cpp-python`）。

- `LocalLogprobClient` は ChatML の会話（Qwen をはじめ多くの指示調整済み GGUF モデルのチャット形式）を保持する。`score(text)` はユーザーのターンを足し、各行動について、それをアシスタントの返答としたときのトークン列全体の対数確率（各トークンの確率の積）、先頭トークンだけの対数確率、行動の集合の中で正規化し直した確率を返す。`temperature` は softmax の前に各トークンの logits を割る。
- `LlamaCppBackend` は各 prompt のうち前のステップから新しく増えた部分だけを評価するので、600 ステップのセッションでも速い。
- `LocalLogprobSubject` は他の被験体と同じく `OperantBox` で走る。チャンバーがスケジュールを当てるには行動が 1 つ発せられる必要があり、それは計算した分布から種付きで引いた行動（`policy="sample"`）か、最も確率の高い行動（`policy="greedy"`）である。`history` の各要素は、そのステップで読んだ確率 `p` を持つ。

```bash
.venv/bin/python -m pip install -e ".[local,viz]"
.venv/bin/python examples/local_logprob_fi.py --model path/to/model.gguf --interval 30 --steps 600 --out-dir runs/fi30
```

このスクリプトは P(Lever Press) をセッション時間に沿って、また前の強化子からの経過時間ごとの平均として描き、全ステップの確率を JSON に書き出す。Qwen2.5-1.5B-Instruct（Q4_K_M）を Apple silicon の 1 台で動かすと、FI 30 の 600 ステップに約 7 分かかった（1 ステップ平均 0.7 秒。prompt は約 13,000 トークンまで伸びた）。このトークナイザーでは `Lever Press` と `Grooming` の先頭トークンが 1 文字（`L`、`G`）なので、正規化し直すのは先頭トークンの確率ではなくトークン列全体の確率である。

再利用した prefix の結果を最初から評価し直した結果と突き合わせるテストは、環境変数 `AOB_GGUF` が GGUF ファイルを指すときだけ走る。他のテストは偽のバックエンドを使い、モデルは要らない。

## 2 本のレバー: 並立 VI VI とマッチング法則

`ai_operant_box.chamber.concurrent.ConcurrentBox` は、`OperantBox` と同じ SD-R-SR のステップを 2 つの操作体で走らせる（`concurrent` extra: `contingency-py`）。切替後遅延（COD; Catania, 1966）を含む強化の判定は `contingency-py` の `Concurrent` スケジュールが行う。`concurrent_vi_vi(mean_left, mean_right, cod=2.0, seed=...)` は種付きの成分で `Conc VI VI` を作る。

- **被験体の I/F。** どの `Subject` でも使える。`act()` は行動名を返し、`operanda` がレバー押しとみなす行動名を操作体のキーに写す。既定は `"Left Lever"` → `"left"`、`"Right Lever"` → `"right"`（LLM 被験体向けの語彙は `CONCURRENT_ACTIONS = ("Left Lever", "Right Lever", "Nothing")`。例: `StreamingLLMSubject(client, actions=CONCURRENT_ACTIONS)`）。それ以外の行動名は反応ではなく、スケジュールには時間の刻みだけが渡る。`learn()` は直前に押したレバーについて `"Food Pellet"` か `None` を受け取る。
- **記録。** 反応 1 回が `ChoiceRecord(timestamp, operandum, is_reinforced, cod_suppressed)` 1 件になる。
- **マッチングの当てはめ。** `ai_operant_box.analytics.matching.choice_totals(records, start=...)` は助走の後のレバーごとの反応数と強化数を数え、`fit_generalized_matching(b1, b2, r1, r2)` は条件をまたいで `log(B1/B2) = a * log(R1/R2) + log b`（常用対数; Baum, 1974）を最小二乗で当てはめ、`a`・`log_b`・R² を返す。`correction=0.5` は全カウントに 0.5 を足し、空のセルがある短いセッションでも当てはめられるようにする（推定値と一緒に報告すること）。`changeovers(records, start=...)` はレバー間の切替回数を数える。
- **LLM 被験体。** `CONCURRENT_SYSTEM_PROMPT` は単一レバー用 prompt の 2 レバー版で、行動名とフィードバックだけを伝え、レバーの強化のされ方には触れない。`examples/claude_cli_matching.py --model haiku` は `claude` CLI 経由で Claude のモデルを下の 4 水準で走らせ（条件ごとに新しいセッション、1 秒のステップを 100 回、全体で 3 秒に 1 強化、既定は COD なし）、カウント・切替回数・2 種の当てはめ（セッション全体と後半）を JSON に書く。`--plot <dir>` で当てはめを描く。
- **参照用の被験体。** `MatchingAgent(sensitivity=s, mean_visit=v)` は毎ステップどちらかのレバーを押し、得た強化子から確率 `R_L^s / (R_L^s + R_R^s)` で左を選び、1 ステップごとに確率 `1/v` でレバーを選び直す。チャンバーと当てはめを確かめるためのもので、生体のモデルではない。

```bash
.venv/bin/python -m pip install -e ".[concurrent]"
.venv/bin/python examples/concurrent_matching.py --cod 0 --sensitivity 1
```

強化子の比 4 水準（1:1、1:3、3:1、1:9。全体で 10 秒に 1 強化。1 秒のステップを 4,000 回、最初の 1,000 回は除く）で、参照用の被験体は COD なしで `a = 1.006`、`log b = 0.015`、R² = 0.998、`sensitivity=0.5` で `a = 0.450` だった。COD 2 秒のときの結果は、入っている `contingency-py` で変わる。0.1.1 は COD の間に準備された強化子を捨てるので、乏しい側のレバーがその強化子を失い、参照用の被験体は片側だけを選ぶ状態に流れる（強化数 0 のレバーがあると当てはめは定義できない）。COD 後の最初の反応まで強化子を保持する `contingency-py`（Herrnstein, 1961）では `a = 0.966`、`log b = 0.015`、R² = 0.997 だった。COD ありのテストは、入っている `contingency-py` が強化子を捨てる場合はスキップされる。

## 自作の 50 行のチャンバーでは足りない理由

モデルに「押すか押さないか」を尋ねて押した回数を数える短いループには、結果を動物のデータと比べられるようにする部分が欠けている:

- **三項随伴性をそのまま I/F にしている。** 被験体はすべて `sense(SD)`・`act() -> R`・`learn(SR)` を実装する（`ai_operant_box.agent.base.Subject`）。数値エージェントと LLM が同じチャンバー・同じスケジュールで走る。
- **模擬時計。** `VirtualClock` はステップ幅だけ進むので、間隔スケジュールは API の応答待ちの実時間ではなくセッション時間で計られる。
- **Fleshler-Hoffman (1962) の数列による変動スケジュール。** VI と VR の値は Fleshler-Hoffman の指数近似から作る。RI と RR は別に用意している。
- **二者の相互依存スケジュール。** `LeaderFollowerBox` は、先導者の反応の後の窓の中で追従者が反応したとき両者に強化子を渡し、反応率・強化率・平均 IRT・同期率を返す。
- **FI スキャロップのモデル。** `ai_operant_box.analytics.model_fit.ModelFitAnalyzer` は区間ごとの反応確率に `P(R | t) = min(1, k * t**2)` を当てはめる。
- **LLM 被験体の文脈内の学習履歴。** `LLMSubject` はそれまでの刺激・反応・結果を会話のターンとしてモデルに送り返す。

## テスト実行

```bash
.venv/bin/python -m pytest -q
```

Anthropic SDK を行使するテストは、`anthropic` パッケージがインストール
されていない場合、自動的にスキップされる。

## 三項随伴性

コード構造は基本的オペラント単位を反映する:

1. **SD (弁別刺激)** — `Chamber` が `Subject` に各ステップの開始で
   提示する。
2. **R (反応)** — `Subject` が発する（数値方針または LLM 呼び出し）。
3. **SR (結果)** — `Schedule` が `Subject` に供給し、`Subject` の
   `learn(reinforcer)` が内部状態を更新する。

## 任意の extra と OperantKit ライブラリ

本体の依存は NumPy だけ。他の OperantKit ライブラリは任意で、遅延
インポートされる:

| Extra | 入るもの | 使う箇所 |
|---|---|---|
| `viz` | `session-analyzer`、matplotlib、SciPy | `ai_operant_box.analytics`（累積記録、スキャロップの当てはめ） |
| `canonical` | `experiment-core`、`session-recorder` | `ai_operant_box.async_simulator.canonical_sink`（独自の事象列を OKL v1 に写す） |
| `llm` | `anthropic` | `ai_operant_box.agent.api_client` |
| `local` | `llama-cpp-python` | `ai_operant_box.agent.local_logprob` |
| `concurrent` | `contingency-py` | `ai_operant_box.chamber.concurrent` |

その他:

- 小さな `chamber.environment.VirtualClock`（`ManualClock` スタイルの
  単調カウンタ）がローカルに提供され、`experiment-io` の HAL なしで
  chamber を走らせられる。
- 単一レバーのスケジュール・プリミティブ (`chamber.schedule` と
  `schedules/`) は本パッケージ内の実装。`contingency-py` を import するのは
  `chamber.concurrent` だけ（`concurrent` extra）。

## Anthropic SDK（オプトイン）

`anthropic` パッケージは `[llm]` extra 配下で宣言された **オプション**
依存関係。`ai_operant_box.agent.api_client` の内部で遅延インポート
されるため、パッケージのインポートや決定論的テストの実行に SDK は
不要。API キーは `ANTHROPIC_API_KEY` 環境変数から読まれ、ハード
コードされることはない。

## 参考文献

- Baum, W. M., & Grace, R. C. (2020). Matching theory and induction explain
  operant performance. *Journal of the Experimental Analysis of Behavior*,
  *113*(2), 390-418. https://doi.org/10.1002/jeab.583
- Baum, W. M. (1974). On two types of deviation from the matching law:
  Bias and undermatching. *Journal of the Experimental Analysis of
  Behavior*, *22*(1), 231-242. https://doi.org/10.1901/jeab.1974.22-231
- Benvenuti, M. F., et al. (2024). The role of schedules of reinforcement in
  the experimental analysis of social behavior. *Perspectives on Behavior
  Science*. https://doi.org/10.1007/s40614-024-00421-5
- Catania, A. C. (1966). Concurrent performances: Reinforcement
  interaction and response independence. *Journal of the Experimental
  Analysis of Behavior*, *9*(3), 253-263. https://doi.org/10.1901/jeab.1966.9-253
- Fleshler, M., & Hoffman, H. S. (1962). A progression for generating
  variable-interval schedules. *Journal of the Experimental Analysis of
  Behavior*, *5*(4), 529-530. https://doi.org/10.1901/jeab.1962.5-529
- Herrnstein, R. J. (1961). Relative and absolute strength of response as
  a function of frequency of reinforcement. *Journal of the Experimental
  Analysis of Behavior*, *4*(3), 267-272. https://doi.org/10.1901/jeab.1961.4-267
- Skinner, B. F. (1938). *The behavior of organisms: An experimental
  analysis*. Appleton-Century.
