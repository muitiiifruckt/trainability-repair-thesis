# Результаты автономной программы

Это живой отчёт: завершённые запуски отделены от запланированных. Минимум для вывода — независимые истории обучения и continuation repeats.

Проверка состояния: 2026-10-08, около 00:36 MSK; выполняется первая online screen серия.
Все 12 исходных checkpoints завершены и импортированы coordinator. В worker JSONL доступны четыре завершённые ветки (continue и optimizer reset, два repeats); их текущие jobs ещё выполняют остальные действия меню.
Текущий пакет: 0; учтены 1,179,937 optimizer updates. Работа текущей волны ещё не полностью включена в этот счётчик.

## Интерпретация и следующие проверки

Screen проверяет полезность четырёх вмешательств в online RL. Fixed-TD probes проверяют fitting при фиксированных данных и targets; это отдельный outcome. LR controls проверяют объяснение через шаг оптимизации. Stress режимы исследуют границы, отдельно от обычного обучения.

Seaquest и Freeway разрешены для проверок API; обучение и policy evaluation открываются только после freeze протокола и подтверждения development signal.

Конфигурации, сырые JSONL, версии и SHA сохранены рядом. Большие полные checkpoints и replay находятся в локальном runs/; они исключены из Git.

## Первые исходные checkpoints и выполнение

Source sanity: Breakout seeds 0/1/2 на 50 499 transitions дают mean raw return 3.9/5.15/5.2 на 20 evaluation episodes, random anchor 0.6. На 200 499 transitions те же seeds дают 7.75/7.8/6.6. Asterix seeds 0/1/2 дают 1.0/1.1/0.65 на младшем и 2.25/1.5/1.1 на старшем возрасте, random anchor 0.75. Target age во всех двенадцати checkpoints — 500 updates. Это результаты обучения исходных агентов; они сами по себе не измеряют изменение trainability.

Первое частичное сравнение на natural Breakout seed 0, младший checkpoint: final return continue — 5.85/6.95, optimizer reset — 5.10/5.95. Парные эффекты reset: −0.75/−1.00. Это одна training history и два repeats; head reset и joint reset ещё считаются. Направление пока **неопределённо**, общий вред или пользу optimizer reset не установили.

Recovery curve показывает, почему горизонт важен: на 5k transitions optimizer reset имеет больший средний return, на 50k — меньший. Primary outcome остаётся заранее выбранным final return; ранняя скорость восстановления и AUC рассматриваются отдельно. Raw данные и воспроизводимый график сохранены в `monitoring/20261007T2134Z.json`, `monitoring/20261007T2134Z.md` и `monitoring/partial_repairs_breakout_seed0_age50000.png`.

Один source worker получил MemoryError при сохранении состояния. Supervisor дождался здоровых workers и возобновил очередь с двумя workers. Проблемный Breakout seed 0 продолжился с checksum-verified savepoint, сохранил первый checkpoint и обучается дальше. Текущей ошибки coordinator нет. Numerical failures пока не наблюдались. Полный набор: 65 tests passed; отдельно выполнены process-spawn и crash/resume проверки.

Исправлена повреждённая кодировка инструкции фонового контроля; сохранённый текст проверен повторным чтением. Снимок данных этой проверки и объяснение следующего шага находятся в `monitoring/20261007T2034Z.json` и `monitoring/20261007T2034Z.md`.

Фоновый supervisor отслеживает единственный coordinator и возобновляет очередь. Проверка в этом чате настроена каждые полчаса; сообщения только по содержательным изменениям. Текущие данные: локальный `runs/minatar-repair-20261007/progress.json`, source workers и manifest. Эта секция будет заменена очередным автоматически обновлённым отчётом после завершённой серии.
