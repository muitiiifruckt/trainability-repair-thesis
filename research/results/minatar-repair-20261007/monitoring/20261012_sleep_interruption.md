# Прерывание процессов и восстановление, 12 октября

## Что проверено и зачем

Heartbeat обнаружил отсутствие supervisor PID 13144, coordinator PID 17736 и обоих branch writers. Последний worker progress — 14:29:58 UTC, обычное выполнение Breakout seed 4, age 200k, continue repeats 2/3; ошибок в хвосте нет. Проверка liveness всех workspace processes и leases нужна до resume, чтобы не создавать второго writer.

Windows System фиксирует сон в 14:30:17 UTC (Application API) и пробуждение в 14:49:20 UTC. Shutdown/restart в этом окне не зарегистрирован; boot остался 10 октября. WER после пробуждения ссылается на старые dumps апреля/сентября, поэтому не доказывает новый BSOD. Точная причина завершения всей группы Python не установлена. Классификация — **инфраструктурное прерывание**, не numerical failure и не отрицательный repair outcome. Настройки питания не менялись.

## Сохранения, восстановление и accounting

Перед запуском verified SHA обоих `branch_resume.pt`, metadata job IDs и source checkpoint. Сохранено по **35,000 из 50,000 continuation steps**, initial optimizer clock 195,500; прежние worker/task roots, seed 4 и repeats 2/3 совпадают. Все 248 root outcomes конечны, failure_reason отсутствует. Их SHA256 — `7f68d695486570972d92e605959652a49870fe537ac63c227c13e3cf5f9a7103`. Scientific code/config совпадают с frozen manifest. [Preflight](20261012T1455Z_interruption_preflight.json) сделан без torch и optimizer updates.

Прежняя очередь возобновлена через existing hidden supervisor с двумя workers. Новые управляющие PID 8440/14436; normal restart загрузил подготовленный controller из commit `88b50ab`, SHA `04639ffddea8bdc5f836131946031f9eed4d24d0648bf0471b9480d44388f849`. Recorded plan продолжает additional_sources до нового adaptive gate. Learner/config не менялись.

Независимый read-only audit воспроизвёл reconciliation: root package **3,700,254 → 3,770,254 updates**, ровно 70,000; ledger каждого прежнего worker root содержит accounted_updates=35,000. Повторное обучение с нуля не выполнялось. Children восстановили 35k, сохранили 40k и дошли до следующих optimizer-reset branches. При завершении worker меню оплачивается разность cumulative counter и уже учтённых 35k. Reservation не является фактическим списанием. Работу после последнего сохранённого counter до прекращения процесса нельзя точно восстановить по журналу; этот caveat остаётся, сумму wave actual_new_updates нельзя считать полным расходом.

[Срез после восстановления](20261012_sleep_interruption_resumed.json) на 14:59:15 UTC фиксирует два живых children, 248 root outcomes, отсутствие worker errors, прежний package 3 и выполнение recorded plan. Активные `.pt` во время writers не хэшировались повторно: SHA bytes проверяли только в preflight без живых процессов.

## Вывод и следующий контроль

Восстановление инфраструктуры подтверждено; научный вывод не меняется. Expanded natural панель ещё неполна: additional_sources 56/192, additional_repeats ещё ожидает. Проверять рост progress и ledger, не перезапускать здоровых workers. Следующий analysis — по завершении сбалансированной серии, с trajectory grouping и отдельными cohorts. Reserved игры закрыты; текущий пакет после resume в пределах 12M updates/24h. Exact process-termination cause остаётся неизвестной, при повторении собирать новые OS/process evidence без изменения scientific semantics.
