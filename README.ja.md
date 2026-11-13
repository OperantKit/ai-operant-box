# ai-operant-box

:gb: [English README](README.md)

ai-operant-box は、大規模言語モデル（LLM）や単純な数値エージェントを、動物の代わりに模擬のオペラント実験箱に入れます。被験体はステップごとに弁別刺激（SD）を示され、反応（R）し、強化スケジュールが定める結果（SR）を受け取ります。これが三項随伴性です（Skinner, 1938）。モデルの行動がスケジュールの制御下に入るかを、動物と同じ手続きと指標で問いたい研究者や教員に向けたものです。[OperantKit](https://github.com/OperantKit) の中では、LLM を被験体にするシミュレータを担い、イベントを OKL ログ（OperantKit の平文のイベントログ）として書き出して他のパッケージに渡せます。

- **同じ実験箱にどの被験体も入る。** 数値エージェントと LLM が同じスケジュールで走るので、モデルを参照用の被験体と比べられます。
- **API の時間ではなくセッションの時間。** 時計は模擬なので、モデルが返事に要した時間は間隔に数えられません。
- **動物の文献の指標:** FI スキャロップの当てはめ、FI のパタンの指標、一般化マッチング法則。

## インストール


Python 3.11 以上。ai-operant-box はまだ PyPI に公開されていないので、GitHub から入れます。本体が依存するのは NumPy だけで、数値エージェント、単一レバーの実験箱、下の例が動きます。

```bash
pip install git+https://github.com/OperantKit/ai-operant-box
```

累積記録、スキャロップの当てはめ、OKL ログ、2 本レバーの実験箱は他の OperantKit パッケージを使います。これらも PyPI に無いので、一緒に GitHub から入れます。Matplotlib と SciPy は session-analyzer と一緒に入ります。

```bash
# PyPI に公開されるまでは、OperantKit の依存と一緒に GitHub から入れる。
pip install git+https://github.com/OperantKit/contingency-dsl-py \
            git+https://github.com/OperantKit/contingency-py \
            git+https://github.com/OperantKit/experiment-core \
            git+https://github.com/OperantKit/session-recorder \
            git+https://github.com/OperantKit/session-analyzer
```

Claude のモデルを被験体にするには、`claude` コマンド（ログイン済みの Claude Code）か、`pip install anthropic` と `ANTHROPIC_API_KEY` のどちらかが要ります。

## 30 秒で動く例

前の強化子からの経過時間の 2 乗で反応確率が増える数値の被験体（`P(R) = min(1, k * t**2)`）を、固定間隔 30 秒のスケジュールで 1 秒のステップ 600 回走らせます。LLM も API キーも要りません。

```python
import random

from ai_operant_box.agent.scallop_agent import ScallopingAgent
from ai_operant_box.chamber.environment import OperantBox
from ai_operant_box.chamber.schedule import FixedInterval

random.seed(42)  # ScallopingAgent はモジュールの乱数生成器から引く
agent = ScallopingAgent(k=0.005)
box = OperantBox(subject=agent, schedule=FixedInterval(30.0))

for _ in range(600):  # FI 30 秒で 1 秒のステップを 600 回
    agent.update_time(1.0)
    box.step(1.0)

responses = len(box.records)
reinforcers = sum(r.is_reinforced for r in box.records)
print(f"responses={responses} reinforcers={reinforcers}")
# responses=422 reinforcers=20

# 30 秒の間隔を 10 等分した区間ごとの反応数（セッション全体で合算）
bins = [0] * 10
last_sr = 0.0
for r in box.records:
    bins[min(9, int((r.timestamp - last_sr) / 3.0))] += 1
    if r.is_reinforced:
        last_sr = r.timestamp
print(bins)
# [1, 5, 12, 30, 54, 60, 60, 60, 60, 80]
```

反応は各間隔の初めには少なく、終わりに向かって増えます（FI スキャロップ）。最後の区間には各間隔の強化された反応も入ります。時計は模擬なので、1 秒かからずに終わり、毎回同じ数を出力します。

## 何が入っているか

| 部品 | 中身 |
|---|---|
| 被験体（`ai_operant_box.agent`） | `Subject` の I/F（`sense(SD)`、`act() -> R`、`learn(SR)`）。数値エージェント（`ScallopingAgent`、`BernoulliAgent`、Baum & Grace, 2020 に基づく `InductionAgent`、`MatchingAgent`）。Anthropic API・`claude` コマンド・手元の GGUF モデルを通した LLM の被験体 |
| 実験箱（`ai_operant_box.chamber`） | `OperantBox`（レバー 1 本。contingency-py のスケジュールも受け、`Overlay` などの outcome の罰を読み分ける）、`ConcurrentBox`（レバー 2 本。contingency-py を使った切替遅延付きの並立 VI VI）、`LeaderFollowerBox`（相互依存スケジュールの下の 2 個体。Benvenuti et al., 2024）、`MatchingToSampleBox`（会話の 1 往復が 1 試行。contingency-py の `MatchingToSample` を使う） |
| スケジュール | CRF、EXT、FI、DRL、VI、VR、RI、RR、2 個体の相互依存スケジュール |
| 解析（`ai_operant_box.analytics`） | 累積記録、FI スキャロップの当てはめ `P(R \| t) = min(1, k * t**2)`、FI のパタンの指標（quarter life、曲率指数）、一般化マッチング法則の当てはめ |
| OKL への出力 | `async_simulator.canonical_sink.mirror_events_to_sink` がイベントを [session-recorder](https://github.com/OperantKit/session-recorder) の書き出し先に渡す |
| 例（`examples/`） | Claude の被験体で CRF、CRF の後の消去、FI、DRL、罰の有無を切り替える VI、並立 VI VI、ヨークト強化、2 個体のスケジュール、刺激等価性の試験（Sidman & Tailby, 1982）。数値の被験体で並立 VI VI。手元のモデルで FI |

LLM を被験体として走らせる方法と、ステップあたりの実測時間: [docs/ja/llm-subjects.md](docs/ja/llm-subjects.md)。2 本レバーの実験箱とマッチング法則の当てはめ: [docs/ja/concurrent-matching.md](docs/ja/concurrent-matching.md)。

## 制限

- **時間は離散的なステップで進みます。** ふつうは 1 秒で、1 ステップの中の反応は区別しません。
- **変動スケジュールは Fleshler–Hoffman ではありません。** 単一レバーの実験箱の VI・VR・RI・RR の値は、指数分布から独立に引いて整数に丸めたものです。contingency-py のスケジュールを使うのは 2 本レバーの実験箱だけです。
- **被験体はモデルで、生体ではありません。** 数値エージェントは実験箱と指標を確かめるためのものです。`LLMSubject` はモデルの返事全体を行動とみなすので、返事が行動名そのものであるときだけ数えられます。`StreamingLLMSubject` は返事の中から行動名を探し、行動名が 1 つに定まらない返事を「Nothing」として記録します。
- **ホストされた LLM の被験体は Claude のモデルだけ**です（Anthropic API か `claude` コマンド）。他のモデルには自作のクライアントが要ります。
- **`ai-operant-box` コマンドは短い動作確認（`ai-operant-box smoke`。同じ引数なら毎回同じ行を出し、`--seed` で別の乱数列にする）しかしません。** 実験は Python か `examples/` のスクリプトから走らせます。

## AI のコーディングエージェント向け

AI のコーディングエージェント向けの案内は [AGENTS.md](AGENTS.md) にあります。

## 開発

```bash
python -m venv .venv
G=git+https://github.com/OperantKit
.venv/bin/python -m pip install --no-deps -e . $G/contingency-dsl-py $G/contingency-py \
  $G/experiment-core $G/session-recorder $G/session-analyzer
.venv/bin/python -m pip install numpy matplotlib scipy pytest pytest-cov pytest-asyncio
.venv/bin/python -m pytest
```

テストに anthropic SDK と `claude` コマンドは要りません。手元のモデルを読み込むテスト 1 つだけは、`llama-cpp-python` が入っていて `AOB_GGUF` が GGUF ファイルを指すとき以外はスキップされます。`tests/test_readme_examples.py` は両方の README の「30 秒で動く例」を走らせ、出力された行を照合します。

## 参考文献

- Baum, W. M., & Grace, R. C. (2020). Matching theory and induction explain operant performance. *Journal of the Experimental Analysis of Behavior, 113*(2), 390–418. https://doi.org/10.1002/jeab.583
- Benvenuti, M. F., Bernardy, J. L., de Toledo, T. F. N., Santiago, J., Siqueira, J. de O., & Silveira, P. S. P. (2024). The role of schedules of reinforcement in the experimental analysis of social behavior. *Perspectives on Behavior Science, 47*(4), 717–738. https://doi.org/10.1007/s40614-024-00421-5
- Sidman, M., & Tailby, W. (1982). Conditional discrimination vs. matching to sample: An expansion of the testing paradigm. *Journal of the Experimental Analysis of Behavior, 37*(1), 5-22. https://doi.org/10.1901/jeab.1982.37-5
- Skinner, B. F. (1938). *The behavior of organisms: An experimental analysis*. Appleton-Century.

## ライセンス

MIT。[LICENSE](LICENSE) を参照してください。
