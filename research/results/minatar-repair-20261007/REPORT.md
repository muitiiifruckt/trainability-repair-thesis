# Результаты автономной программы

Это живой отчёт: завершённые запуски отделены от запланированных. Минимум для вывода — независимые истории обучения и continuation repeats.

Обновление: 2026-10-18 22:47:03; серия: package_5.
Checkpoints: 24. Завершённые ветки: {'screen': 192, 'confirm': 268, 'stress': 0, 'parameters': 0, 'lr_control': 0}.
Текущий пакет: 6; optimizer updates: 0.

## natural, бюджет 50000 transitions

Статус: **серия ещё не завершена или содержит пропуски/сбои**.
Историй: 12; полных checkpoint×repeat панелей: 114; минимум repeats: 4.
Numerical failures: {}; missing/nonfinite finals: 0.

- asterix, continue: paired effect 0.000; 95% interval [0.0, 0.0]; histories 6.
- breakout, continue: paired effect 0.000; 95% interval [0.0, 0.0]; histories 6.
- asterix, optimizer_reset: paired effect 0.083; 95% interval [-0.14484375, 0.3104166666666667]; histories 6.
- breakout, optimizer_reset: paired effect 0.067; 95% interval [-0.269045138888889, 0.4048784722222223]; histories 6.
- asterix, head_reset: paired effect 0.091; 95% interval [-0.09067708333333334, 0.3000520833333334]; histories 6.
- breakout, head_reset: paired effect -0.268; 95% interval [-0.6229340277777776, 0.134435763888889]; histories 6.
- asterix, head_and_optimizer_reset: paired effect 0.264; 95% interval [0.011432291666666691, 0.49690104166666677]; histories 6.
- breakout, head_and_optimizer_reset: paired effect -0.608; 95% interval [-0.9670399305555556, -0.24268229166666658]; histories 6.

Интервалы группируются по source history. Шумный максимум не считается oracle; незавершённые или failed пары не объявляются успехом.

## Интерпретация и следующие проверки

Screen проверяет полезность четырёх вмешательств в online RL. Fixed-TD probes проверяют fitting при фиксированных данных и targets; это отдельный outcome. LR controls проверяют объяснение через шаг оптимизации. Stress режимы исследуют границы, отдельно от обычного обучения.

Seaquest и Freeway разрешены для проверок API; обучение и policy evaluation открываются только после freeze протокола и подтверждения development signal.

Конфигурации, сырые JSONL, версии и SHA сохранены рядом. Большие полные checkpoints и replay находятся в локальном runs/; они исключены из Git.
