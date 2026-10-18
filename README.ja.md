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

LLM バックエンドの被験体を有効化するには:

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

同じチャンバーに言語モデルを入れるには `llm` extra を入れて `ai_operant_box.agent.llm_agent.LLMSubject` を使う。事象を OKL v1 のログとして残すには `canonical` extra を入れ、`ai_operant_box.async_simulator.canonical_sink.mirror_events_to_sink` で [session-recorder](https://github.com/OperantKit/session-recorder) の sink に渡す。

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

その他:

- 小さな `chamber.environment.VirtualClock`（`ManualClock` スタイルの
  単調カウンタ）がローカルに提供され、`experiment-io` の HAL なしで
  chamber を走らせられる。
- スケジュール・プリミティブ (`chamber.schedule` と `schedules/`) は
  本パッケージ内の実装。`contingency-py` は import しない。

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
- Benvenuti, M. F., et al. (2024). The role of schedules of reinforcement in
  the experimental analysis of social behavior. *Perspectives on Behavior
  Science*. https://doi.org/10.1007/s40614-024-00421-5
- Fleshler, M., & Hoffman, H. S. (1962). A progression for generating
  variable-interval schedules. *Journal of the Experimental Analysis of
  Behavior*, *5*(4), 529-530. https://doi.org/10.1901/jeab.1962.5-529
- Skinner, B. F. (1938). *The behavior of organisms: An experimental
  analysis*. Appleton-Century.
