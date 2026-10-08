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

## Интерпретация и следующие проверки

Screen проверяет полезность четырёх вмешательств в online RL. Fixed-TD probes проверяют fitting при фиксированных данных и targets; это отдельный outcome. LR controls проверяют объяснение через шаг оптимизации. Stress режимы исследуют границы, отдельно от обычного обучения.

Seaquest и Freeway разрешены для проверок API; обучение и policy evaluation открываются только после freeze протокола и подтверждения development signal.

Конфигурации, сырые JSONL, версии и SHA сохранены рядом. Большие полные checkpoints и replay находятся в локальном runs/; они исключены из Git.
