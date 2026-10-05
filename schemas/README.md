# Формат будущих экспериментальных данных

Данных ещё нет. Ниже договорённость о формате до первых запусков.

## `checkpoints.jsonl`

Одна запись на checkpoint: `checkpoint_id`, `trajectory_id`, `environment_id`, `environment_family`, `algorithm`, `architecture`, `training_seed`, `environment_steps`, `gradient_updates`, `artifact_path`, `sha256`, `config_hash`, `normalization_state_id`, `optimizer_age`, `target_update_age`, `replay_id`, `rng_snapshot_id`.

## `diagnostics.jsonl`

Одна запись на diagnostic calculation: `checkpoint_id`, `diagnostic_version`, `reference_dataset_id`, `feature_names`, `feature_values`, `layer_names`, `diagnostic_env_steps`, `diagnostic_gradient_updates`, `diagnostic_wall_time_sec`, `measured_before_intervention`.

Feature values можно хранить как словарь чисел, но order/units/нормировку явно версионировать. Не включать post-repair return или oracle label в features.

## `outcomes.jsonl`

Одна запись на branch: `checkpoint_id`, `repair_id`, `repair_config_hash`, `continuation_seed`, `evaluation_seeds`, `budget_env_steps`, `budget_gradient_updates`, `wall_time_sec`, `j_pre`, `j_immediate`, `j_final`, `adaptation_auc`, `evaluation_curve`, `failure_reason`, `branch_state_hash`.

`evaluation_curve`: пары `(env_steps, mean_return)` плюс числа episodes и uncertainty. Не заменять failed branches на нулевые rewards без явного правила. Сохранять исходы всех actions.

## `splits.json`

Явные `train`, `development`, `test` списки environment families и trajectory IDs, `split_version`, `created_before_test_results`, `normalization_fit_ids`. Все checkpoints одной trajectory принадлежат одной части.

## `repairs.json`

Для каждого `repair_id`: точная спецификация изменённых weights, optimizer tensors/timestep, target state, replay, normalization и schedules; источник дополнительной информации; immediate prediction preservation; стоимость сбора данных; доступность при реальном применении.

Это схема записей, не implementation API. Реализацию runner выбирать после пилота и проверки авторского кода.
