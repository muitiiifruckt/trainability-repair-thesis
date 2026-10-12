# Результаты автономной программы

**Редакционная проверка:** intervals при <5 source histories скрыты согласно AGENTS.md. Screen и confirmation сравниваются отдельно в [CURRENT_STATUS.md](CURRENT_STATUS.md); повторы одной source history не являются независимыми histories.

Это живой отчёт: завершённые запуски отделены от запланированных. Минимум для вывода — независимые истории обучения и continuation repeats.

Обновление: 2026-10-12 12:51:50; серия: package_2.
Checkpoints: 18. Завершённые ветки: {'screen': 96, 'confirm': 96, 'stress': 0, 'parameters': 0, 'lr_control': 0}.
Текущий пакет: 3; optimizer updates: 0.

## natural, бюджет 50000 transitions

Статус: **серия ещё не завершена или содержит пропуски/сбои**.
Историй: 6; полных checkpoint×repeat панелей: 48; минимум repeats: 4.
Numerical failures: {}; missing/nonfinite finals: 0.

- asterix, continue: paired effect 0.000; histories 3. Интервал скрыт (n<5).
- breakout, continue: paired effect 0.000; histories 3. Интервал скрыт (n<5).
- asterix, optimizer_reset: paired effect 0.010; histories 3. Интервал скрыт (n<5).
- breakout, optimizer_reset: paired effect -0.098; histories 3. Интервал скрыт (n<5).
- asterix, head_reset: paired effect 0.054; histories 3. Интервал скрыт (n<5).
- breakout, head_reset: paired effect -0.221; histories 3. Интервал скрыт (n<5).
- asterix, head_and_optimizer_reset: paired effect 0.310; histories 3. Интервал скрыт (n<5).
- breakout, head_and_optimizer_reset: paired effect -0.444; histories 3. Интервал скрыт (n<5).

Интервалы группируются по source history. Шумный максимум не считается oracle; незавершённые или failed пары не объявляются успехом.

## Интерпретация и следующие проверки

Screen проверяет полезность четырёх вмешательств в online RL. Fixed-TD probes проверяют fitting при фиксированных данных и targets; это отдельный outcome. LR controls проверяют объяснение через шаг оптимизации. Stress режимы исследуют границы, отдельно от обычного обучения.

Seaquest и Freeway разрешены для проверок API; обучение и policy evaluation открываются только после freeze протокола и подтверждения development signal.

Конфигурации, сырые JSONL, версии и SHA сохранены рядом. Большие полные checkpoints и replay находятся в локальном runs/; они исключены из Git.
