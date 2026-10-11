# Результаты автономной программы

Это живой отчёт: завершённые запуски отделены от запланированных. Минимум для вывода — независимые истории обучения и continuation repeats.

Обновление: 2026-10-11 12:50:06; серия: package_1.
Checkpoints: 12. Завершённые ветки: {'screen': 96, 'confirm': 48, 'stress': 0, 'parameters': 0, 'lr_control': 0}.
Текущий пакет: 2; optimizer updates: 0.

## natural, бюджет 50000 transitions

Статус: **неопределённо: разрешения эксперимента недостаточно**.
Историй: 6; полных checkpoint×repeat панелей: 36; минимум repeats: 2.
Numerical failures: {}; missing/nonfinite finals: 0.

- asterix, continue: paired effect 0.000; 95% interval [0.0, 0.0]; histories 3.
- breakout, continue: paired effect 0.000; 95% interval [0.0, 0.0]; histories 3.
- asterix, optimizer_reset: paired effect 0.337; 95% interval [0.024999999999999977, 0.7083333333333334]; histories 3.
- breakout, optimizer_reset: paired effect -0.098; 95% interval [-0.4917187500000001, 0.2896354166666667]; histories 3.
- asterix, head_reset: paired effect -0.004; 95% interval [-0.34593750000000006, 0.2875]; histories 3.
- breakout, head_reset: paired effect -0.221; 95% interval [-0.9296874999999997, 0.6083854166666666]; histories 3.
- asterix, head_and_optimizer_reset: paired effect 0.771; 95% interval [0.39999999999999997, 1.262708333333333]; histories 3.
- breakout, head_and_optimizer_reset: paired effect -0.444; 95% interval [-0.975, 0.08755208333333338]; histories 3.

Интервалы группируются по source history. Шумный максимум не считается oracle; незавершённые или failed пары не объявляются успехом.

## Интерпретация и следующие проверки

Screen проверяет полезность четырёх вмешательств в online RL. Fixed-TD probes проверяют fitting при фиксированных данных и targets; это отдельный outcome. LR controls проверяют объяснение через шаг оптимизации. Stress режимы исследуют границы, отдельно от обычного обучения.

Seaquest и Freeway разрешены для проверок API; обучение и policy evaluation открываются только после freeze протокола и подтверждения development signal.

Конфигурации, сырые JSONL, версии и SHA сохранены рядом. Большие полные checkpoints и replay находятся в локальном runs/; они исключены из Git.
