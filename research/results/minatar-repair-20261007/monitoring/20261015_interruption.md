# Восстановление после перезагрузок Windows, 15 октября

## Подтверждённая причина и границы наблюдения

Все прежние research supervisor/coordinator/workers завершены. Последний root progress — 14 октября 13:27:34 UTC, children — 13:27:29 UTC. В Windows Kernel-Boot уже в 13:30:05 UTC зарегистрирован full boot; затем TrustedInstaller User32 1074 в 13:31:34 UTC фиксирует **Operating System: Update (Planned)**, code0x80020003, следующие reboots — 13:31:47 и 13:53:24 UTC, между ними winlogon1074. Planned update/reboots подтверждены; точный момент завершения Python и причина первого boot отдельно не наблюдались. Новых Kernel-Power41/EventLog6008 в проверенном отрезке нет. Последовали power off14 октября17:52UTC, hybrid boot15 октября10:20UTC, ещё sleep/wake10:59–11:00UTC. Поздние WER с апрельскими/сентябрьскими dumps не объявляются новым crash.

Старый child PID13640 теперь принадлежит protected **svchost.exe**, созданному14 октября13:53:57UTC. Он не является research writer; системный процесс не останавливался. Native query к нему недоступен, поэтому утверждение «PID мёртв» не используется как описание OS inventory. Возобновление по существующей scripts-orchestration успешно прошло, worker leases сменились на новые Python. Живых research writers до запуска не было, второй coordinator не создавался.

Классификация — инфраструктурное прекращение после системных перезагрузок, не MemoryError, FloatingPointError или отрицательный repair outcome. JSON сейчас valid, NUL corruption нет; power/update settings не менялись.

## Сохранность и возобновление

Preflight root: **456 outcomes**, failures0, SHA256 `255ec133a0708bc4517d618c5bf66301fc5598c745578f27aa9b144b7ac74470`. Полный accepted additional_sources завершён192/192; additional_repeats72/192 ещё неполон. Verified source **Breakout seed4 age200k**: env200,499/gradient195,500, SHA `0b3a692496e9178412169ccabe26c9863f11d27e6303dc983d802268e9338aca`. В двух pending tasks уже завершены continue/optimizer по50k (четыре local raw rows). Незавершённый head_reset repeats4/5 восстановлен от45k:

- repeat4 SHA `4e2d6be61a7ca182ee79e242f94c3a2e51ef4a7cbb48ec1954f7535b226b74cc`;
- repeat5 SHA `40d0eadf329e8a5499f34c846c704baea1577d9d36b3a2e29a5ff7f31e8a7612`.

Все binary checksums, scientific config/nine code hashes, recorded decision/operations и full completed budgets проверены до запуска. Stable IDs/continuation/evaluation seeds сохранены. Normal hidden resume применяет прежний controller e06f068/wrapper SHA `35ec4448d658cacce2fa02b2efad417a4d2702ccd88b28d7de24338dc7628d37`; scientific code/config не менялись, новые learner jobs вне принятой очереди не добавлялись. Общие regression tests не перезапускались: code unchanged, проверки этого восстановления — integrity/clocks/raw import/accounting.

На срезе **11:25 UTC** actual supervisor **2920**, coordinator **8492**, workers **20148/3008** живы. Head steps45k→46,844/46,849, gradient242,344/242,349=source195,500+steps. Root **460 outcomes** после неизменного импорта четырёх completed rows; первоначальный prefix456 сохранён побайтно. Independent read-only audit подтвердил single tree, clocks, seeds и hashes.

## Учёт ресурсов

Parent package5 до возобновления0 updates; прежние child physical counters **145,120+145,106=290,226** reconciled однократно. Из них200k — completed continue/optimizer,90k — восстановленные head steps, **226** — observed discarded updates после savepoints120+106. Они не списываются; ненаблюдаемая работа после последнего counter возможна. Remaining head10k и joint100k ещё выполняются; ожидаемые final ledgers200,120/200,106 и root **400,226** пока расчёт, не факт этого среза.

На возобновлении wall package5 **~22.38h**, updates меньше12M, wall меньше24h; календарный downtime учитывается. Package4 ранее закрыт frozen runner с6,466,837 updates/**87,154.33s=24.2095h**, то есть754.33s сверх строгого24h ceiling. Это наблюдаемый infrastructure defect, metadata не переписываются. Cooperative strict deadline остаётся обязательным исправлением новой кампании согласно completion audit.

Полный balanced natural cohort384 (seeds0–5/repeats0–3) анализируется отдельным научным срезом; текущие дополнительные repeats4/5 не подмешиваются. Reserved Seaquest/Freeway закрыты. [Compact evidence](20261015_interruption.json).
