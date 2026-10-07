# Формат экспериментальных данных — версия 1

Исполняемый договор находится в `experiments/rl_runner.py`: компактные записи — JSON/JSONL, полные network/optimizer/replay/environment/RNG snapshots — локальные `.pt`. Smoke и native API проверки уже выполнены. JSON использует null с явным failure status вместо NaN/Infinity.

## Конфигурация и manifest

`config.json` фиксирует learner, main_menu, источники, бюджеты, LR grid и development/reserved games. `manifest.json`: canonical config hash, SHA исходного кода, версии, wheel-validation hash, jobs, пакеты, решения и report commits. `dependency_manifest.json` — эффективные версии runtime. Изменение config hash требует нового output root.

`progress.json` — текущие stage/job/clocks/return/time. `journal.jsonl`: что проверено → зачем → результат → объяснение → следующий тест. Статус выполнения не является научным outcome.

## `checkpoints.jsonl`

Одна запись на checkpoint: `checkpoint_id`, `trajectory_id`, `game`/`environment_id`, `environment_family`, `training_seed`, `source_mode`, `nominal_age`, фактические `environment_steps`, `gradient_updates`, `target_update_age`, `artifact_path`, `sha256`, `config_hash`, `j_pre`, `pre_evaluation`, `random_return`, предварительный `baseline_learned`. Learner/architecture заданы в immutable config; optimizer/replay/RNG входят в artifact.

Вся `trajectory_id` — одна группа независимости. Два возраста одной истории не являются двумя source seeds. `baseline_learned` — screening эвристика относительно отдельного random anchor, не статистическое доказательство обучения.

## `diagnostics.jsonl`

Фактические поля: `checkpoint_id`, `diagnostic_version`, `features` (словарь чисел/null), `details`, `measured_before_intervention`, `source_mode`. `rl-pre-v1`: age/clocks, recent return/slope, relative weight norm, feature rank/dormancy, gradient/moment и TD-target/error statistics. Стоимость вычисления хранится в details. Imputation/scaling fit только внутри training folds.

Feature values можно хранить как словарь чисел, но order/units/нормировку явно версионировать. Не включать post-repair return или oracle label в features.

## `outcomes.jsonl`

Одна запись на branch: `job_id`, `stage`, `checkpoint_id`, `trajectory_id`, `game`, `source_mode`, `training_seed`, `nominal_age`, `repair_id`, `repair_seed`, `repeat`, `continuation_seed`, `evaluation_seeds`, `budget_env_steps`, `completed_env_steps`, фактический `budget_gradient_updates`, `probe_gradient_updates`, `wall_time_sec`, `j_pre`, `j_immediate`, `j_final`, `adaptation_auc`, `evaluation_curve`, `failure_reason`, `checkpoint_sha256`.

`evaluation_curve`: actual age, raw return каждого episode и общие для actions seeds. AUC нормирована на horizon. Numerical failure сохраняется с null final; инфраструктурная ошибка повторяется с прежним seed после исправления. Успешные пары помечаются отдельно при failures.

## Splits и frozen transfer

Grouped fold audits содержат training/evaluation trajectory IDs, feature names и preprocessing provenance. Все checkpoints одной trajectory принадлежат одной части. `transfer_protocol.json` фиксирует development/reserved games, fitted model hashes, comparisons, seeds и probe costs до reserved evaluations; canonical envelope SHA проверяется при чтении.

`transfer_outcomes.jsonl` дополнительно хранит `policy_id`, `selected_action`, `phase`, `protocol_hash`, `evaluation_started_at`, declared `total_budget_gradient_updates`, `continuation_gradient_updates`, `continuation_env_steps`. Chooser пробует четыре ремонта по 500 updates, выбрасывает probe weights и независимо продолжает за оставшиеся 48k updates из общего бюджета 50k.

`transfer_summary.json` содержит raw per-game effects, harm/failure/missing rates и history-grouped uncertainty; population oracle не оценивается.

## `repairs.json`

Для каждого `repair_id`: точная спецификация изменённых weights, optimizer tensors/timestep, target state, replay, normalization и schedules; источник дополнительной информации; immediate prediction preservation; стоимость сбора данных; доступность при реальном применении.

Main four repairs сохраняют target/replay/exploration в момент вмешательства. Полный optimizer reset очищает m/v/t, head reset возвращает только online head к собственной initialization и сохраняет Parameter identities. Injection — отдельная head-only residual конструкция с frozen old/reference heads; это смешанное вмешательство, не pure weights repair.

## Fixed-TD и synthetic

`mechanisms/<checkpoint>_r<repeat>.json`: две frozen teacher functions, episode-disjoint split/tapes hashes, main7 curves, main4×5LR, data×target panel, errors под обеими teachers, actual updates и forward/training examples. Недостаток данных даёт explicit skip без скрытого collector. Refreshed bootstrap не считается правильной разметкой.

`synthetic_lr_sweep.json` помечен **not_RL**: source LR .001, continuation LR .00025/.001/.004, общие tape prefixes для 100/500 updates, failures и actual compute. Большие checkpoints и raw replay остаются локально в `runs/`; compact outcomes/configs/hashes/plots экспортируются в Git.
