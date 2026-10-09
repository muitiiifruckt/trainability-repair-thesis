> Актуальный срез 9 октября около 14:41 MSK: 72 из 96 screening-веток завершены, расчёты возобновлены. Текущие результаты — `CURRENT_STATUS.md`, исходные данные — `monitoring/20261009T1140Z.json`, график — `monitoring/history_effects_20261009T1140Z.png`. Разделы с «95% interval» ниже являются устаревшим срезом 16 строк; при малом числе histories эти интервалы не используются для научных выводов. Дополнительные оговорки — `NOTES_manual.md`.

# Результаты автономной программы

Это живой отчёт: завершённые запуски отделены от запланированных. Минимум для вывода — независимые истории обучения и continuation repeats.

Обновление: 2026-10-08 23:48:40; серия: package_0.
Checkpoints: 12. Завершённые ветки: {'screen': 16, 'confirm': 0, 'stress': 0, 'parameters': 0, 'lr_control': 0}.
Текущий пакет: 1; optimizer updates: 0.

## natural, бюджет 50000 transitions

Статус: **серия ещё не завершена или содержит пропуски/сбои**.
Историй: 1; полных checkpoint×repeat панелей: 4; минимум repeats: 2.
Numerical failures: {}; missing/nonfinite finals: 0.

- breakout, continue: paired effect 0.000; 95% interval [0.0, 0.0]; histories 1.
- breakout, optimizer_reset: paired effect -0.512; 95% interval [-0.6499999999999999, -0.375]; histories 1.
- breakout, head_reset: paired effect -0.412; 95% interval [-0.8499999999999996, 0.025000000000000355]; histories 1.
- breakout, head_and_optimizer_reset: paired effect -1.487; 95% interval [-1.7499999999999996, -1.225]; histories 1.

Интервалы группируются по source history. Шумный максимум не считается oracle; незавершённые или failed пары не объявляются успехом.

## Обновление 2026-10-09 (после возобновления)

Кампания простояла около 22 часов (координатора и workers не было, `progress.json` не менялся) и возобновлена `scripts/research_supervisor.py` с теми же seeds и savepoints; с тех пор новых infrastructure errors нет. Импортировано 24 основных outcomes: полное меню из четырёх действий на трёх ячейках (Breakout seed 0 в возрастах 50k и 200k; seed 1 в возрасте 50k), по два repeat. Разброс шума: SE парной разности final return на ветку 0.6–1.9 при 20 evaluation episodes (воспроизводится `scripts/debate_noise_floor.py`, вывод в консоль; на момент записи — по ранним 16 строкам).

Описательно, парные эффекты относительно continue по 6 ячейкам (checkpoint, repeat): совместный reset отрицателен во всех шести (от −0.35 до −2.2); optimizer reset и head reset имеют смешанный знак. Независимых историй две, поэтому интервалы не вычисляются (порог пять историй), а направление остаётся **неопределённым** как научный вывод. Gates, конфигурация и очередь не менялись. Анализ: `scripts/interaction_age_analysis.py`, `scripts/power_from_partial.py` (оценка мощности возможна от трёх историй), результаты в `research/results/parallel/`.

Дополнительно: параллельный дебат двух исследователей (`research/debate/`) обнаружил утечку будущих меток в одной из синтетических диагностик (AUC 0.972) — результат снят; патч whitelist признаков для `rl_analysis.py` проверен на копии и отложен до конца кампании, чтобы не менять код с зафиксированным SHA в `dependency_manifest.json`.

## Интерпретация и следующие проверки

Screen проверяет полезность четырёх вмешательств в online RL. Fixed-TD probes проверяют fitting при фиксированных данных и targets; это отдельный outcome. LR controls проверяют объяснение через шаг оптимизации. Stress режимы исследуют границы, отдельно от обычного обучения.

Seaquest и Freeway разрешены для проверок API; обучение и policy evaluation открываются только после freeze протокола и подтверждения development signal.

Конфигурации, сырые JSONL, версии и SHA сохранены рядом. Большие полные checkpoints и replay находятся в локальном runs/; они исключены из Git.
