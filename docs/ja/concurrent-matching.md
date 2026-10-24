# 2 本のレバー: 並立 VI VI とマッチング法則

:gb: [English](../en/concurrent-matching.md)

`ai_operant_box.chamber.concurrent.ConcurrentBox` は、`OperantBox` と同じ SD-R-SR のステップを 2 つのオペランダムで走らせます（contingency-py が要ります。[README](../../README.ja.md) のとおりに入れてください）。切替遅延（COD; Catania, 1966）を含む強化の判定は contingency-py の `Concurrent` スケジュールが行います。`concurrent_vi_vi(mean_left, mean_right, cod=2.0, seed=...)` は種付きの成分で `Conc VI VI` を作ります。

- **被験体の I/F。** どの `Subject` でも使えます。`act()` は行動名を返し、`operanda` がレバー押しとみなす行動名をオペランダムのキーに写します。既定は `"Left Lever"` → `"left"`、`"Right Lever"` → `"right"` です（LLM の被験体向けの語彙は `CONCURRENT_ACTIONS = ("Left Lever", "Right Lever", "Nothing")`。例: `StreamingLLMSubject(client, actions=CONCURRENT_ACTIONS)`）。それ以外の行動名は反応ではなく、スケジュールには時間の刻みだけが渡ります。`learn()` は直前に押したレバーについて `"Food Pellet"` か `None` を受け取ります。
- **記録。** 反応 1 回が `ChoiceRecord(timestamp, operandum, is_reinforced, cod_suppressed)` 1 件になります。
- **マッチングの当てはめ。** `ai_operant_box.analytics.matching.choice_totals(records, start=...)` は助走の後のレバーごとの反応数と強化数を数え、`fit_generalized_matching(b1, b2, r1, r2)` は条件をまたいで `log(B1/B2) = a * log(R1/R2) + log b`（常用対数; Baum, 1974）を最小二乗で当てはめ、`a`・`log_b`・R² を返します。`correction=0.5` は全カウントに 0.5 を足し、空のセルがある短いセッションでも当てはめられるようにします（推定値と一緒に報告してください）。`changeovers(records, start=...)` はレバー間の切替の回数を数えます。
- **LLM の被験体。** `CONCURRENT_SYSTEM_PROMPT` は単一レバー用の prompt の 2 レバー版で、行動名とフィードバックだけを伝え、レバーの強化のされ方には触れません。`examples/claude_cli_matching.py --model haiku` は `claude` CLI 経由で Claude のモデルを下の 4 水準で走らせ（条件ごとに新しいセッション、1 秒のステップを 100 回、全体で 3 秒に 1 強化、既定は COD なし）、カウント・切替の回数・2 種の当てはめ（セッション全体と後半）を JSON に書きます。`--plot <dir>` で当てはめを描きます。
- **参照用の被験体。** `MatchingAgent(sensitivity=s, mean_visit=v)` は毎ステップどちらかのレバーを押し、得た強化子から確率 `R_L^s / (R_L^s + R_R^s)` で左を選び、1 ステップごとに確率 `1/v` でレバーを選び直します。実験箱と当てはめを確かめるためのもので、生体のモデルではありません。

```bash
python examples/concurrent_matching.py --cod 0 --sensitivity 1
```

強化子の比 4 水準（1:1、1:3、3:1、1:9。全体で 10 秒に 1 強化。1 秒のステップを 4,000 回、最初の 1,000 回は除く）で、参照用の被験体は COD なしで `a = 1.006`、`log b = 0.015`、R² = 0.998、`sensitivity=0.5` で `a = 0.450` でした。COD 2 秒のときの結果は、入っている contingency-py で変わります。0.1.1 は COD の間に設定された強化子を捨てるので、強化の少ない側のレバーがその強化子を失い、参照用の被験体は片側だけを選ぶ状態に流れます（強化数 0 のレバーがあると当てはめは定義できません）。COD 後の最初の反応まで強化子を保持する contingency-py（Herrnstein, 1961）では `a = 0.966`、`log b = 0.015`、R² = 0.997 でした。COD ありのテストは、入っている contingency-py が強化子を捨てる場合はスキップされます。

## 参考文献

- Baum, W. M. (1974). On two types of deviation from the matching law: Bias and undermatching. *Journal of the Experimental Analysis of Behavior, 22*(1), 231–242. https://doi.org/10.1901/jeab.1974.22-231
- Catania, A. C. (1966). Concurrent operants. In W. K. Honig (Ed.), *Operant behavior: Areas of research and application* (pp. 213–270). Appleton-Century-Crofts.
- Herrnstein, R. J. (1961). Relative and absolute strength of response as a function of frequency of reinforcement. *Journal of the Experimental Analysis of Behavior, 4*(3), 267–272. https://doi.org/10.1901/jeab.1961.4-267
