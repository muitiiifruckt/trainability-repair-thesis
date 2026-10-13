# Восстановление после аварийного завершения Windows, 13 октября

## Проверка и причина

При возвращении отсутствовали все workspace Python и живые worker leases. Windows зарегистрировала новый boot 13 октября 09:12:57 UTC, Kernel-Power 41 и EventLog 6008. BugcheckCode=0, SleepInProgress=4; конкретная причина выключения неизвестна. Event 6008 сообщает прежнее shutdown 12 октября 20:33:18 UTC, тогда как последний обычный worker log датирован 20:37:31 UTC. Это расхождение не позволяет указать точный момент прерывания. В хвосте нет traceback, MemoryError или FloatingPointError. Независимый агент воспроизвёл OS/liveness checks.

Supervisor JSON (211 bytes), manifest/progress/resume JSON обоих pending workers целиком заполнены NUL. При этом root manifest, frozen config/code, 24 checkpoints/diagnostics и **328 root outcomes** читаются нормально. Все root outcomes конечны, failures=0; SHA256 `51a62a6774f31dcd937a2f2a055822c8462073be14037b35624a4304e6809f8d`. В каждом worker сохранились три raw completed outcomes continue/optimizer/head по 50k. Их шесть результатов дополнительно защищены SHA и сверкой со source checkpoint.

Классификация — **инфраструктурное прерывание и повреждение control metadata**, не отрицательный научный результат. Повреждённый resume JSON не позволяет проверить исходный `.pt` checksum/curve. Непроверенный `.pt` не объявляется verified savepoint: обе незавершённые joint branches повторяются от проверенного source Asterix seed 4, age 50k, с прежними IDs/repeats 2/3 и seeds.

## Исправление и проверки

Внешний helper `scripts/recover_nul_control_files.py`, commit **b29eba8**, не импортирует torch и не меняет learner/config. Он удерживает оба root locks, исключает живых owners, сверяет child config с frozen parent, recomputes stable task digest, проверяет source SHA и восстанавливает completed jobs только из полного raw JSONL. Замена разрешена только для файлов, целиком состоящих из NUL. Исходные повреждённые файлы и непроверенные PT сохранены в игнорируемом `runs/minatar-repair-20261007/recovery/20261013_nul_controls/`; raw outcomes не переписаны. Audit записан до изменений, успешный apply подтверждён `applied.json` до запуска supervisor. Записи helper используют flush/fsync перед replace; для всей scientific инфраструктуры durability gap остаётся открытым.

Прошли **14 stdlib checks за 0.032 s** (пять новых и девять resume checks), без optimizer updates. Проверены отказ от замены valid file, сохранение только трёх completed jobs при stale branch_complete event, запрет partial/nonfinite/другого repeat, согласование source clocks и запрет обратного ledger. Независимый review не нашёл blocking risks; multi-file apply при сбое требует отдельного recovery, supervisor до успешного apply не запускается.

После применения проверены valid reconstructed manifests, прежние raw SHA и отсутствие scratch resume slots. Один hidden supervisor возобновил recorded plan с двумя workers. Scientific code/config и power settings не менялись.

## Accounting и фактический результат

Повреждённые child manifests восстановлены по последнему progress родителя: repeat 2 — 150k completed + 29,051 partial = **179,051**; repeat 3 — 150k + 28,992 = **178,992**. Это ресурсные counters, не learner clocks. Parent ledgers перед resume были нулевыми. Frozen scheduler однократно reconciles **358,043** прежних observed updates и импортирует шесть completed rows; прежние три действия не обучаются повторно.

Оба полных joint reruns завершены с бюджетом 50k и без failures. Final child ledgers **229,051 / 228,992**, суммарно **458,043**: 358,043 наблюдавшейся прежней работы + 100k новых updates. Сюда включены **58,043** discarded partial updates; их не скрываем как бесплатный restart. Работа после последнего наблюдавшегося counter могла остаться ненаблюдаемой. Original raw prefixes шести завершённых rows сохранили SHA и совпадают с импортированными root rows.

Просроченный package 3 закрыт **до новых optimizer updates** с 7,700,254 updates и 97,119.38 s (**26.98 h**, включая простой). Metadata не переписаны задним числом; strict 24h ceiling прежним runner не обеспечен. Package 4 открыт перед reconciliation и свежим обучением. В [проверенном срезе](20261013_nul_control_recovery.json) на 13:23:07 UTC он содержит 458,043 updates и 1,967.48 s; supervisor PID 17528/coordinator 19124 живы, следующие workers обучаются, numerical failures отсутствуют.

Root теперь **336 outcomes**: additional_sources 144/192, ещё 48 ожидают; additional_repeats 0/192. Серия ещё неполна, scientific conclusions и pilot cohort 192 не переоцениваются по несбалансированному срезу. Seaquest/Freeway закрыты. Следующий контроль — завершение balanced серии и independent repeats; новый durability protocol только в следующей кампании с new output_root. Автоматический package report обновлён frozen runner; interval suppressor проверен (0 intervals n<5 на этом срезе). Крупный generated mechanism_summary остаётся известной проблемой export/Git hygiene; push в этом ходе не выполнялся.
