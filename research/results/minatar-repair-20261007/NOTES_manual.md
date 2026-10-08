# Ручные заметки кампании (вне зоны перезаписи генератора)

Генератор `experiments/rl_runner.py` пересоздаёт `REPORT.md` с нуля при ротации пакета (`write_text`, около 2026-10-09 23:49) и может сам закоммитить
папку results. Этот файл генератор не трогает. Ниже — полный снимок ручного состояния `REPORT.md` на 2026-10-09.

## Оговорки к автогенерируемым частям отчёта (аудит B2, `research/debate/log.md`)
- «95% interval» при 1–2 историях в автогенерируемом разделе — это размах двух повторов внутри checkpoint (`rl_analysis.py:153-185`: checkpoint не пересэмплируются),
  а не доверительный интервал. Реальное покрытие номинальных 95% при 1/2/3/5/6/12 историях: 0.44/0.71/0.84/0.89/0.89/0.94 (`scripts/debate_b2_ci_coverage.py`).
  Интервалы считать только при >=5 историй, лучше t-интервал/sign-flip по историям.
- `paired_effects_*.png` и `repair_analysis.json` (`ci95`, `small_history_count: true`) при малом числе историй — не результат. `continue.immediate_change.ci95` исключает 0
  при нулевом истинном изменении, потому что j_pre и j_immediate считаются на разных eval seeds (`rl_runner.py:289` против `:362`).
- Снимки `monitoring/*.md` неизменяемы; их поправки — в `monitoring/ERRATA.md`.

---

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
