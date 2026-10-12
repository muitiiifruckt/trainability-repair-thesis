# Результаты автономной программы

Это живой отчёт: завершённые запуски отделены от запланированных. Минимум для вывода — независимые истории обучения и continuation repeats.

Обновление: 2026-10-12 12:51:28; серия: first_online_screen_96_branches.
Checkpoints: 18. Завершённые ветки: {'screen': 96, 'confirm': 96, 'stress': 0, 'parameters': 0, 'lr_control': 0}.
Текущий пакет: 2; optimizer updates: 2,990,697.

## natural, бюджет 50000 transitions

Статус: **серия ещё не завершена или содержит пропуски/сбои**.
Историй: 6; полных checkpoint×repeat панелей: 48; минимум repeats: 4.
Numerical failures: {}; missing/nonfinite finals: 0.

- asterix, continue: paired effect 0.000; 95% interval [0.0, 0.0]; histories 3.
- breakout, continue: paired effect 0.000; 95% interval [0.0, 0.0]; histories 3.
- asterix, optimizer_reset: paired effect 0.010; 95% interval [-0.25421875000000005, 0.3166666666666667]; histories 3.
- breakout, optimizer_reset: paired effect -0.098; 95% interval [-0.4604166666666667, 0.3042187500000002]; histories 3.
- asterix, head_reset: paired effect 0.054; 95% interval [-0.24796875000000002, 0.3167708333333333]; histories 3.
- breakout, head_reset: paired effect -0.221; 95% interval [-0.9813020833333331, 0.5792187499999999]; histories 3.
- asterix, head_and_optimizer_reset: paired effect 0.310; 95% interval [-0.12744791666666666, 0.6875]; histories 3.
- breakout, head_and_optimizer_reset: paired effect -0.444; 95% interval [-1.0627083333333334, 0.08750000000000002]; histories 3.

Интервалы группируются по source history. Шумный максимум не считается oracle; незавершённые или failed пары не объявляются успехом.

## Интерпретация и следующие проверки

Screen проверяет полезность четырёх вмешательств в online RL. Fixed-TD probes проверяют fitting при фиксированных данных и targets; это отдельный outcome. LR controls проверяют объяснение через шаг оптимизации. Stress режимы исследуют границы, отдельно от обычного обучения.

Seaquest и Freeway разрешены для проверок API; обучение и policy evaluation открываются только после freeze протокола и подтверждения development signal.

Конфигурации, сырые JSONL, версии и SHA сохранены рядом. Большие полные checkpoints и replay находятся в локальном runs/; они исключены из Git.
