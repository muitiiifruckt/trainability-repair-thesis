# Автономная исследовательская программа: от вмешательств к выбору repair

Зафиксировано для программы `minatar-repair-20261007`. Машиночитаемые настройки —
[`configs/research_program.json`](../configs/research_program.json); исполнение —
[`experiments/rl_runner.py`](../experiments/rl_runner.py). Этот документ описывает
правила исследования и границы выводов. Наличие кода не означает, что все этапы
уже завершены; состояние выполнения хранится в `runs/<program_id>/manifest.json`
и `progress.json`.

## Вопрос и основной estimand

В каких состояниях обучавшегося агента разные вмешательства дают воспроизводимо
разный эффект, и добавляют ли измеренные **до вмешательства** diagnostics пользу
при выборе repair сверх возраста, недавнего return и одной общей стратегии?

Для фиксированного checkpoint и режима продолжения основной результат — return
после заранее заданного бюджета. Основное сравнение действия `a` с `continue` —
парная разность `j_final(a) - j_final(continue)`. Дополнительно сохраняются кривая
обучения, её AUC, изменение качества сразу после вмешательства и доля вредящих
continuation runs. Восстановление старого head может сначала ухудшить policy;
рост final return не отменяет эту цену. Это измерение ответа на вмешательство,
а не самостоятельное доказательство существования «испорченных весов».

Единица независимого split и внешнего bootstrap — **полная исходная история
обучения**, `trajectory_id`. Несколько возрастов одного агента не становятся
независимыми seeds. Внутри checkpoint действия сравниваются по одинаковым
continuation seeds; после различающихся действий траектории среды могут
разойтись. Общие seeds уменьшают часть шума, но не обеспечивают одинаковую
occupancy в live RL.

## Конечная последовательность этапов

1. **Инварианты и smoke.** Проверка клонирования checkpoint, привязки optimizer к
   действительным параметрам, сохранения replay и clocks, конечности обновлений,
   независимости evaluation RNG. Короткий запуск используется для проверки
   исполнения и скорости, не как научный результат.
2. **Synthetic sanity и LR controls.** Teacher tasks позволяют проверить
   чувствительность к специально созданным состояниям и нескольким learning
   rates. Continuation LR: `0.00025`, `0.001`, `0.004`, 500 updates; source LR
   `0.001`. Эти задачи не являются RL и не доказывают потерю plasticity в RL.
3. **Первый online screen.** MinAtar `breakout`, `asterix`; source seeds `0,1,2`;
   nominal ages `50 000`, `200 000` переходов; четыре действия; два независимых
   повтора продолжения. Итого 96 основных branches. Horizon заранее установлен
   в `50 000` переходов; evaluation в `0,5 000,20 000,50 000`, по 20 episodes.
   Checkpoint сдвигается минимум на 500 шагов после target update; в анализе
   сохраняются и nominal age, и фактический `environment_steps`.
4. **Fixed-TD mechanisms.** Отдельные панели recent/old replay и старых/обновлённых
   targets, контролируемые updates и LR sweep. Source seeds `0,1`; 2 000 updates
   основной панели, 500 updates коротких панелей; LR grid
   `0.0000625,0.000125,0.00025,0.0005,0.001`. Heldout transitions отделяются от
   fitting transitions. Это supervised TD-fitting assay; его loss и learning
   progress нельзя выдавать за live-policy return или selection на unseen game.
5. **Повторы и разрешение неопределённости.** Повторы `2,3` для всей основной
   панели, а не только для её победителей. При неопределённости добавляются
   source seeds `3,4,5` и повторы `4,5`. Новые данные уточняют development signal;
   многократное рассмотрение результатов не превращается в подтверждающий test.
6. **Проверка baseline и альтернативных условий.** Если поздний baseline не
   научился, сначала source age `500 000`, затем при необходимости `1 000 000`;
   RMSprop control отделяется от Adam. При отсутствии natural-mode heterogeneity
   доступны заранее объявленные stressors: replay ratio 4 и reward scale 0.1,
   horizon `20 000`, два повтора. Stress modes анализируются отдельно и не
   заменяют natural-mode доказательство. LR, подобранный на fixed-TD данных
   других исходных историй, проверяется отдельно в live continuation.
7. **Prospective diagnostics и selection.** Только после пригодного
   natural-mode signal: development extension `space_invaders`; grouped CV и
   leave-one-game-out. Сначала проверяется добавочная польза diagnostic Ridge
   против SBS и age/return Ridge. Tree depth 2 — заранее объявленный вторичный
   класс; выбор класса по его development score требует отдельного подтверждения.
8. **Frozen transfer, условный этап.** `freeway`, `seaquest` остаются reserved до
   заморозки preprocessing, моделей, меню, бюджета, сравнений и cost accounting.
   Source seeds `0,1,2`, continuation repeats `100,101` — ограниченная точность.
   Это перенос внутри MinAtar, а не между произвольными семействами RL задач.

Бюджет одного пакета ограничен `12 000 000` optimizer updates и 86 400 секундами;
возобновление сохраняется каждые 5 000 шагов. Текущий runner при достижении лимита
закрывает пакет с отчётом и начинает следующий: это граница учёта ресурсов,
а не общий лимит всей программы или отрицательный научный результат.

## Learner и меню вмешательств

Общий DQN: 10 input channels, convolution 16, hidden size 128, 6 actions, float32,
Adam LR `0.00025`, `eps=1e-8`, betas `(0.9,0.999)`, batch 32, replay capacity
100 000, warmup 5 000, gamma 0.99, target period 1 000 updates, grad clip 10,
replay ratio 1. Epsilon снижается до 0.1 за 100 000 шагов после warmup.
MinAtar: sticky-action probability 0.1, difficulty ramping включён, episode cap
10 000; truncation не приравнивается к настоящему terminal для bootstrap target.

Основной factorial:

- `continue`: сохранённое состояние;
- `optimizer_reset`: очищается optimizer memory; online weights сохраняются;
- `head_reset`: **существующие параметры** online output head получают значения
  исходной инициализации этой истории; optimizer memory сохраняется;
- `head_and_optimizer_reset`: эти два вмешательства одновременно.

В основном меню не сбрасываются target network, replay, среда, exploration clock,
глобальный счётчик updates и scheduler. Adam clock входит в optimizer memory;
его отдельный reset и эквивалентный LR/epsilon schedule — controls механизма.
Номинально равные LR не гарантируют равных фактических Adam updates.

При параметрном сигнале отдельно доступны `fresh_head_reset`,
`fresh_head_and_optimizer_reset`, shrink/perturb `(0.99,0.01)` и head-only
injection. Это конкретные локальные варианты, не полное воспроизведение каждого
авторского RL рецепта. Fresh sample отличается от возвращения старой
инициализации. Injection меняет параметризацию и optimizer groups, поэтому её
нельзя считать единственным изменением прежнего weight state.

Основной [`rl_analysis.py`](../experiments/rl_analysis.py) агрегирует только
основные четыре действия. Сырые auxiliary branches сохраняются, но для их
научного сравнения необходим отдельный отчёт с правильными controls. Их отсутствие
в основном графике не означает отсутствия запуска или эффекта.

## Анализ, gates и что они означают

Основные strata — `(source_mode,budget_env_steps)`; режимы и разные horizons не
смешиваются. Фактический диапазон gradient budgets сохраняется: warmup и
незначительно разное число updates не должны незаметно изменять сравнение.
Конечный return незавершённой ветки не считается результатом полного бюджета.
Failed/missing branches блокируют progression; описательные success-pair estimates
помечаются и не заменяют анализ частоты сбоев.

Внутри каждой game — равный вес checkpoint внутри source history и равный вес
историй. Returns представлены в исходных единицах каждой game. Общая SBS
выбирается на training histories по средним парным эффектам, равным весам game;
raw scales могут влиять на этот выбор. Нормализация по heldout returns запрещена.

Наивный максимум результатов всех действий на тех же повторах показывается
отдельно как оптимистичная описательная величина. **Это не population oracle** и
не строгая верхняя граница истинной value. Cross-fitted estimated winner выбирает
действие на одной половине повторов и оценивается на другой, затем половины
меняются местами. SBS обучается вне **всей** evaluated source history, включая
остальные ages. Bootstrap 1 000 раз пересэмплирует source histories и парные
векторы continuation outcomes; winner и SBS переоцениваются в каждом draw.
Selection/evaluation halves остаются раздельными. Это интервалы фиксированной
исследовательской процедуры, без поправки на множественные сравнения и
последовательное расширение development данных.

Категориальный `primary_gate`:

- `incomplete`: отсутствуют необходимые строки/файлы, есть failed/nonfinite или
  budget-mismatched branches, либо невозможно независимое SBS сравнение;
- `baseline_unlearned`: все известные source checkpoints не превысили declared
  random-policy threshold; сначала проверяется learner;
- `uncertain`: неизвестна baseline qualification, меньше четырёх полных повторов,
  меньше пяти source histories на game или interval не разделяет альтернативы;
- `heterogeneous`: минимум два выбираемых действия и положительная нижняя CI
  estimated-winner advantage над SBS хотя бы на одной достаточной game;
- `single_repair`: один и тот же winner и SBS, а CI advantage не демонстрирует
  прирост; вывод ограничен этим меню и horizon, не доказывает отсутствие
  неоднородности вообще.

`gate_is_confirmatory=false` и `provisional=true` обязательны. Gate пригоден для
маршрутизации development работы, а не для заявления статистически подтверждённой
общей гипотезы. Положительная одна game не доказывает перенос на остальные;
отрицательные эффекты и harm fraction остальных game сохраняются отдельно.
Порог heterogeneity в текущем анализе — declared `heterogeneity_margin`, иначе 0
в raw-return units. Adaptive practical threshold 0.05 в конфигурации не следует
молча трактовать как нормализованный effect size.

Самостоятельный `parameter_repair_signal` использует контрасты
`head_reset-continue` и `head_and_optimizer_reset-optimizer_reset`. При >=5
histories/game нижняя CI выше declared `parameter_effect_margin` (иначе 0)
означает `positive`; недостаточная точность — `uncertain`, сбои — `incomplete`.
Сигнал относится к эффекту определённого head intervention условно на optimizer,
не идентифицирует единственную биологическую/оптимизационную «причину поломки».

## Selector и воспроизводимый контракт

Ridge предсказывает полный вектор repair effects, alpha из
`[0.1,1,10,100]`; decision tree depth 2. Inner CV grouped по source history,
outer CV также grouped; дополнительный split оставляет целую game. Median
imputer, список features, scaler и hyperparameters fit только на training fold.
Age/return baseline получает nominal age, actual environment steps, recent
return и recent return slope. Diagnostics не содержат final outcomes, repair
responses, oracle labels или future observations; ID game/seed не используется
как shortcut. Blacklist названий features — дополнительная защита, но не замена
проверке времени сбора diagnostics.

OOF selector intervals отражают evaluation uncertainty условно на fitted
outer-fold policies; полное обучение selector не повторяется в bootstrap. Поэтому
`promising_grouped_development_signal` — development signal, а не подтверждённая
transfer value. Baselines: continue, training-only SBS, age/return Ridge;
в frozen transfer добавляется short-probe chooser, которому начисляется стоимость
**всех** trial updates, включая отвергнутые действия. Probe weights не переносятся
в итоговую ветку. Общий update budget и реальные forward/wall-time расходы
учитываются отдельно.

API анализа:

- `analyze(run_dir, bootstrap_repeats=1000) -> dict` — descriptive statistics,
  categorical gates, standalone parameter signal и worker progress;
- `fit_selectors(run_dir, bootstrap_repeats=1000, strata=None) -> dict` — явно
  запрошенный exploratory nested CV;
- `build_report(run_dir, output_dir, bootstrap_repeats=1000) -> list[Path]` —
  `repair_analysis.json`, `selector_analysis.json`, headless PNG. Автоматически
  fit selector только при natural `heterogeneous` gate; иначе
  `deferred_before_confirmed_signal` (имя статуса, а не confirmatory claim);
- `train_selector(run_dir, stratum, family='ridge', feature_set='diagnostic')`
  возвращает замораживаемый fitted object; `choose_action(model, features)`
  применяет его preprocessing и возвращает ID основного действия.

Строки `checkpoints.jsonl`, `diagnostics.jsonl`, `outcomes.jsonl` связаны через
`checkpoint_id`; pairing — `(checkpoint_id,continuation_seed,repeat)`.
Development whitelist: `screen.games` плюс `development_extension_games`, за
вычетом reserved games независимо от содержимого screen списка. Reserved
checkpoint без основного factorial outcome не делает development report
незавершённым: его evaluation находится в отдельном `transfer_outcomes.jsonl`.
Frozen protocol и SHA fitted artifacts должны предшествовать reserved outcomes;
после открытия reserved игр нельзя менять family, features, horizon или primary
comparison по их победителям.

Интересный результат: воспроизводимые противоположные repair responses,
инкремент diagnostic над age/return и SBS, сохранённый на независимых историях и
на заранее закрытых game. Конфаунд: один reset выигрывает только при одном LR,
необученном source baseline, невключённой цене chooser, совместном reset target
или optimizer, либо optimism возникает только у same-sample maximum. Нулевой
результат при ограниченной точности — `uncertain`, а не доказательство того, что
plasticity не существует или selector принципиально невозможен.
