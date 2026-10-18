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
