# Результаты автономной программы

Это живой отчёт: завершённые запуски отделены от запланированных. Минимум для вывода — независимые истории обучения и continuation repeats.

Обновление: 2026-10-22 18:18:59; серия: package_6.
Checkpoints: 24. Завершённые ветки: {'screen': 192, 'confirm': 312, 'stress': 0, 'parameters': 0, 'lr_control': 0}.
Текущий пакет: 7; optimizer updates: 0.

## natural, бюджет 50000 transitions

Статус: **неопределённо: разрешения эксперимента недостаточно**.
Историй: 12; полных checkpoint×repeat панелей: 126; минимум repeats: 4.
Numerical failures: {}; missing/nonfinite finals: 0.

- asterix, continue: paired effect 0.000; 95% interval [0.0, 0.0]; histories 6.
- breakout, continue: paired effect 0.000; 95% interval [0.0, 0.0]; histories 6.
- asterix, optimizer_reset: paired effect 0.084; 95% interval [-0.15434027777777776, 0.31875000000000003]; histories 6.
- breakout, optimizer_reset: paired effect 0.060; 95% interval [-0.27309027777777783, 0.41263888888888867]; histories 6.
- asterix, head_reset: paired effect 0.091; 95% interval [-0.09172743055555557, 0.26981770833333335]; histories 6.
- breakout, head_reset: paired effect -0.182; 95% interval [-0.5645833333333331, 0.1980381944444444]; histories 6.
- asterix, head_and_optimizer_reset: paired effect 0.236; 95% interval [-0.00209201388888885, 0.4524479166666667]; histories 6.
- breakout, head_and_optimizer_reset: paired effect -0.515; 95% interval [-0.8326388888888889, -0.17498263888888893]; histories 6.

Интервалы группируются по source history. Шумный максимум не считается oracle; незавершённые или failed пары не объявляются успехом.

## Интерпретация и следующие проверки

Screen проверяет полезность четырёх вмешательств в online RL. Fixed-TD probes проверяют fitting при фиксированных данных и targets; это отдельный outcome. LR controls проверяют объяснение через шаг оптимизации. Stress режимы исследуют границы, отдельно от обычного обучения.

Seaquest и Freeway разрешены для проверок API; обучение и policy evaluation открываются только после freeze протокола и подтверждения development signal.

Конфигурации, сырые JSONL, версии и SHA сохранены рядом. Большие полные checkpoints и replay находятся в локальном runs/; они исключены из Git.
