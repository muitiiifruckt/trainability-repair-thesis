# Выключение Windows и переиспользование PID, 13 октября

## Причина остановки

В 14:28:53 UTC Windows User32 1074 зарегистрировал power off через StartMenuExperienceHost.exe; затем Kernel-Power 42 (Application API). Kernel-Boot 27 показывает hybrid boot, BootType=1, в 14:52:45 UTC; Power-Troubleshooter подтверждает wake. На проверенном отрезке нет новых Kernel-Power 41/EventLog 6008. Это отдельное прерывание после утреннего аварийного завершения; control JSON сейчас читаются нормально. Причину по событиям не заменяем предположением о MemoryError или numerical failure.

Прежние supervisor/coordinator и оба learner завершены. Старый PID **2400** Windows присвоила ChatGPT.exe renderer: новый процесс создан в **14:53:13.382534 UTC**, а lease worker был создан в **14:27:05.386828 UTC**. PID-only проверка ошибочно считала его здоровым writer и блокировала resume. Ни renderer, ни приложение не останавливались. Независимый read-only аудит подтвердил OS events, отсутствие workspace Python до запуска и несовпадение владельца lease.

## Исправление и сохранение данных

Внешняя orchestration `scripts/research_supervisor.py` проверяет image и creation time native Windows API. Известный новый non-Python процесс не владеет старой Python lease. Живые Python, недоступные identities и сомнительные timestamps сохраняют консервативный запрет на запуск. Unknown/malformed ownership metadata также блокируют recovery. Под `worker.lock` повторно проверяются owners, а доказанно устаревший lease переносится в ignored quarantine с исходными bytes/SHA и audit. Общий helper вызывается и supervisor, и standalone wrapper под уже имеющимся lock.

Код сохранён до возобновления: **568152b**, затем schema hardening **e06f068**. Прошли **26 лёгких stdlib tests за 0.116 s**, без Torch и optimizer updates; проверены identity/clock-skew cases, отсутствие изменений branch data, malformed metadata и существующий lock. Independent review указал standalone/schema пробелы; оба устранены до запуска. Scientific `experiments/` и `configs/` не изменялись: все девять SHA сверены с dependency manifest.

Все **360 root outcomes** сохранены, failures=0; preflight SHA256 `59519c74618cbb8a303a9fd9f0b23215b22e0bec357cfaa48393f3577c4c629c`. Verified source — Asterix seed 5, age 50k, gradient_updates=45,500; source SHA `9252cd289146110567888d13822606c5886bf2aac755e6bb158780aba8d210ed`. Два branch savepoints continue repeats 2/3 содержат completed_steps=10,000 и initial_updates=45,500:

- repeat 2: SHA `85734299ac14e68d68f4897174f5c6a3b1e2ae49dfb7812355e53c13590dc248`;
- repeat 3: SHA `78264c7882bb457c6e47e7724f865d5210d0e9db68c391fd5c16fd727cbad2c0`.

Все checksums проверены до запуска. Original task IDs, continuation/evaluation seeds и source clocks сохранены. Исходные control files не заменялись утренним NUL helper повторно. Архивирован только lease PID2400; dead lease PID18292 обработан штатным frozen worker. Archive SHA `929f5596687eca332c6fb01a7b33414095bb091ca210be9f8149f0ac822344fd` совпадает с исходными bytes.

## Возобновление и учёт

Один hidden supervisor штатно продолжил `recorded_resume_plan.json`, workers=2. На срезе **15:30 UTC** supervisor **16392**, coordinator **19148**, workers **14860/13512** живы. Venv launcher создаёт дополнительный parent Python, что не является вторым coordinator. Оба learner продвинулись после загрузки 10k savepoints: continuation **15,062 / 15,081**, gradient clocks **60,562 / 60,581**. Worker error отсутствует.

Parent package 4: **1,658,043 → 1,683,644**, то есть однократный reconciliation прежних **12,801 + 12,800 = 25,601** physical updates. Прежние counters не подменяются восстановленным learner clock. Наблюдавшиеся **2,801 + 2,800 = 5,601** updates после savepoints повторяются, оставаясь в ресурсном бюджете; ненаблюдаемая работа после последнего counter возможна. На срезе package wall **9621.26 s**, меньше 24 h, updates меньше 12M. Историческое нарушение строгого wall ceiling в старых packages остаётся в manifest и completion audit.

Raw prefix 360 outcomes побайтно сохранён. Additional_sources **168/192**, ещё 24 branches; additional_repeats **0/192** остаётся следующим записанным этапом. Recorded decision SHA `02ea086d1da576cc964b53285d0bf7e3f483d945054af3e9c926b49a8184b841` и исполняемый wrapper SHA сверены. Новый adaptive gate не запускался раньше accepted expansion. Возобновление является инфраструктурным результатом; научный итог verified pilot 192 остаётся «неопределённо», частичная расширенная панель его не заменяет. Reserved games закрыты.

Компактные evidence, script hashes, native identities и post-resume snapshot: [JSON](20261013_pid_reuse_recovery.json). После полного balanced cohort обновить результаты и графики; контроль продолжается.
