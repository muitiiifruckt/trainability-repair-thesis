# Возобновление после прекращения процессов, 14 октября

## Что подтверждено и что неизвестно

До запуска отсутствовали все workspace Python, включая supervisor 16392, coordinator 19148 и workers 14860/13512. Контрольные JSON и raw outcomes читаются нормально; NUL corruption и reused PID в этот раз не обнаружены. Последний root progress — 13 октября 18:15:54 UTC, child progress — 18:16:00/18:16:04 UTC. Worker status=running в старом manifest не является признаком живого процесса.

Windows подтверждает sleep 13 октября 16:23:30 UTC (Button or Lid), wake 18:14:41 UTC (Power Button), затем sleep 19:54:43 UTC (Application API) и hybrid boot/wake 14 октября 07:40:08 UTC, BootType=1. **Причина завершения Python не установлена:** около последнего progress нет события, доказывающего завершение, а следующий подтверждённый sleep произошёл позже. Новых Kernel-Power 41, EventLog 6008 или User32 1074 после 13 октября 16:00 UTC нет. Application WER записи с dump names апреля/сентября не объявляются новым BSOD. В последнем learner log нет Traceback, MemoryError или FloatingPointError. Power settings не изменялись.

Это **инфраструктурное прерывание**, отдельно от численных и научных результатов. Independent read-only audit воспроизвёл OS/liveness checks и отсутствие второго controller.

## Проверка сохранений и очереди

Перед возобновлением root содержит **368 завершённых outcomes**, failures=0, SHA256 `8fcabfca68d65f8aae4d6974ed42b9995a2238edc63e9d500308a776c1ebadc2`. Frozen config и все девять scientific code hashes совпали с dependency manifest; recorded decision/operations проверены. План остаётся additional_sources: **176/192**, затем additional_repeats **0/192**. Новый adaptive gate до завершения принятого increase_resolution не запускался.

Verified source — **Asterix seed5 age200k**, environment_steps=200,499, gradient_updates=195,500, SHA `e60c28fd841126569ef39b4f4330430c66cda97ff12b477c3b6d9acfda7fcca1`. В каждой pending ветке уже завершены continue/optimizer/head по 50k; их **шесть raw rows** сохранены и проверены. Продолжены только joint branches с прежними task IDs, repeats0/1, continuation/evaluation seeds:

- repeat0: savepoint35,000, SHA `aa66c1f66ab09c09e4a4d71e77e89dac97d77c790b9e0d78088e7f413932024b`;
- repeat1: savepoint35,000, SHA `a360f623bbb02384b91282ef95696b5e717544d03d181e8f706a66c7a634fadb`.

Checksums двоичных branch/source файлов проверены до старта. Existing hidden resume использует внешний controller revision **e06f068**, wrapper SHA `35ec4448d658cacce2fa02b2efad417a4d2702ccd88b28d7de24338dc7628d37`. Нового learner/config кода или экспериментального дизайна нет, поэтому повторный общий запуск tests не выполнялся; проверки — integrity, canonical plan, source clocks, full completed budgets и raw import. Последний regression suite controller: 26 stdlib checks 13 октября.

## Фактическое возобновление и accounting

На срезе **07:45 UTC** работают один supervisor **9616**, actual coordinator **17984**, два workers **13648/15980**. Venv launcher parents не являются дополнительными coordinators. Root стал **374 outcomes**: импортированы шесть прежних rows, их содержимое совпало с local raw; исходный root prefix 368 rows сохранён побайтно. Новых failure outcomes нет. Оба joint learner продвинулись после 35k: repeat1 **39,301 steps /234,801 gradient clock**, repeat0 **39,283 /234,783**. Clocks равны source 195,500 + continuation steps; повторные resource counters не подменяют clocks.

Physical counters до остановки: repeat1 **186,755**, repeat0 **186,438**. Parent package 4 однократно reconciles **373,193**: **2,063,644 → 2,436,837**. Из этого 300k — завершённые три действия двух веток, 70k — восстановленные joint steps, **3,193** — observed discarded steps после последних savepoints (1,755+1,438). Эти затраты остаются в budget, а ненаблюдаемая работа после последнего counter возможна. Оставшиеся 30k joint updates ещё выполняются; ожидаемый root после их завершения **2,466,837**, это расчёт, не факт на данном срезе.

Package 4 на срезе **68085.60s = 18.91h**, updates меньше 12M и wall меньше 24h; downtime входит в calendar budget. Исторические strict deadline gaps не переписываются и остаются в completion audit.

Scientific expanded cohort ещё неполон, поэтому новый вывод о пользе ремонта или selector не делается. Verified pilot 192 по-прежнему неопределённый; reserved Seaquest/Freeway закрыты. После полного balanced cohort обновить сырые сводки/графики/анализ; текущая работа продолжает прежнюю авторизованную очередь. Evidence: [compact JSON](20261014_power_interruption.json).

## Завершение восстановленных веток

На 07:48 UTC оба joint продолжения завершены до 50k, learner clock 245,500, numerical failures 0. Final physical ledgers: repeat1 **201,755**, repeat0 **201,438**; root package 4 **2,466,837** =2,063,644+373,193+30k. Ранее рассчитанная величина теперь наблюдается. Root **376 outcomes**, additional_sources **184/192**, следующая волна repeats2/3 того же source выполняется двумя здоровыми workers. Все восемь local outcomes совпадают с root, исходные prefixes сохранены, completed budgets 50k и source hashes проверены. Balanced expanded series ещё не завершена, новых scientific claims нет.
