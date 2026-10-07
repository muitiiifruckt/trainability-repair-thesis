# Результаты автономной программы

Это живой отчёт: завершённые запуски отделены от запланированных. Минимум для вывода — независимые истории обучения и continuation repeats.

Обновление: 2026-10-07 22:37:22; серия: intermediate.
Checkpoints: 0. Завершённые ветки: {'screen': 0, 'confirm': 0, 'stress': 0, 'parameters': 0, 'lr_control': 0}.
Текущий пакет: 0; optimizer updates: 1,937.

## Интерпретация и следующие проверки

Screen проверяет полезность четырёх вмешательств в online RL. Fixed-TD probes проверяют fitting при фиксированных данных и targets; это отдельный outcome. LR controls проверяют объяснение через шаг оптимизации. Stress режимы исследуют границы, отдельно от обычного обучения.

Seaquest и Freeway разрешены для проверок API; обучение и policy evaluation открываются только после freeze протокола и подтверждения development signal.

Конфигурации, сырые JSONL, версии и SHA сохранены рядом. Большие полные checkpoints и replay находятся в локальном runs/; они исключены из Git.

## Первые исходные checkpoints и выполнение

Промежуточный source sanity: Breakout seeds 1/2 на 50 499 transitions дают mean raw return 5.15/5.2 на 20 evaluation episodes, random anchor 0.6. Asterix seed 0: 1.0 против random 0.75. Target age во всех трёх checkpoints — 500 updates. Это ещё не результаты ремонта; остальные checkpoints и все continuation branches ожидаются.

Один source worker получил MemoryError при сохранении состояния. Последний checksum-verified savepoint сохранён. Healthy workers продолжаются; supervisor повторит незавершённую очередь с прежними seeds и меньшей параллельностью. Numerical failures пока не наблюдались. Полный набор: 65 tests passed; отдельно выполнены process-spawn и crash/resume проверки.

Фоновый supervisor отслеживает единственный coordinator и возобновляет очередь. Проверка в этом чате настроена каждые полчаса; сообщения только по содержательным изменениям. Текущие данные: локальный `runs/minatar-repair-20261007/progress.json`, source workers и manifest. Эта секция будет заменена очередным автоматически обновлённым отчётом после завершённой серии.
