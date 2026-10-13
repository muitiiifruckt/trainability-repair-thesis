# Результаты автономной программы

Это живой отчёт: завершённые запуски отделены от запланированных. Минимум для вывода — независимые истории обучения и continuation repeats.

Обновление: 2026-10-13 15:50:43; серия: package_3.
Checkpoints: 24. Завершённые ветки: {'screen': 168, 'confirm': 160, 'stress': 0, 'parameters': 0, 'lr_control': 0}.
Текущий пакет: 4; optimizer updates: 0.

## natural, бюджет 50000 transitions

Статус: **серия ещё не завершена или содержит пропуски/сбои**.
Историй: 11; полных checkpoint×repeat панелей: 82; минимум repeats: 2.
Numerical failures: {}; missing/nonfinite finals: 0.

- asterix, continue: paired effect 0.000; 95% interval [0.0, 0.0]; histories 5.
- breakout, continue: paired effect 0.000; 95% interval [0.0, 0.0]; histories 6.
- asterix, optimizer_reset: paired effect -0.031; 95% interval [-0.32999999999999996, 0.2600625]; histories 5.
- breakout, optimizer_reset: paired effect -0.046; 95% interval [-0.36463541666666666, 0.3209114583333332]; histories 6.
- asterix, head_reset: paired effect 0.045; 95% interval [-0.16628125000000005, 0.25253125]; histories 5.
- breakout, head_reset: paired effect -0.293; 95% interval [-0.8072916666666665, 0.24294270833333323]; histories 6.
- asterix, head_and_optimizer_reset: paired effect 0.171; 95% interval [-0.16000000000000006, 0.5214062499999998]; histories 5.
- breakout, head_and_optimizer_reset: paired effect -0.603; 95% interval [-1.0479947916666665, -0.18117187500000007]; histories 6.

Интервалы группируются по source history. Шумный максимум не считается oracle; незавершённые или failed пары не объявляются успехом.

## Интерпретация и следующие проверки

Screen проверяет полезность четырёх вмешательств в online RL. Fixed-TD probes проверяют fitting при фиксированных данных и targets; это отдельный outcome. LR controls проверяют объяснение через шаг оптимизации. Stress режимы исследуют границы, отдельно от обычного обучения.

Seaquest и Freeway разрешены для проверок API; обучение и policy evaluation открываются только после freeze протокола и подтверждения development signal.

Конфигурации, сырые JSONL, версии и SHA сохранены рядом. Большие полные checkpoints и replay находятся в локальном runs/; они исключены из Git.
