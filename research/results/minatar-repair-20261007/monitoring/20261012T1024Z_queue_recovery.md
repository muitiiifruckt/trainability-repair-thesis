# Исправление возобновления записанной очереди

## Дефект и причина

После выключения машины 11 октября источники Breakout seeds 3/4 и Asterix seed 3 уже имели шесть импортированных checkpoints, но continuation outcomes для них ещё отсутствовали. Незавершённым оставался source Breakout seed 5. Повторный `auto --resume` начинал pilot → mechanisms → confirmation → adaptive заново, не восстанавливая cursor принятого `increase_resolution`.

`rl_analysis.py` включает глобальные checkpoints_without_outcomes в `incomplete` каждой stratum. `adaptive()` запускает расширение только при gate=uncertain, поэтому мог пропустить именно то расширение, которое уже создало эти checkpoints. Запись decision в manifest служила журналом, не очередью. Позднейшие LR/stress outcomes могли скрыть missing flag, оставив natural panel незавершённой. Независимый read-only аудит воспроизвёл trace; frozen runner SHA `9315c0aa5afa66cf0ec6df4eba70a2c28cade0d1cb76dedad6b2f37186d8f2c3`, analysis SHA `76315e1904017f37ff0374b3d7fd48ed4beca3bd6c4ae28e9ee6f8da61ad2851`.

## Внешнее исправление без смены learner/source semantics

`scripts/resume_recorded_research.py` владеет тем же worker.lock и сначала восстанавливает уже принятое решение:

1. Source seeds 3–5, repeats 0–3, ages 50k/200k: 192 требуемых outcomes.
2. Source seeds 0–5, repeats 4–5, те же ages: ещё 192 outcomes.
3. Дополнительные mechanisms/report, затем прежний frozen auto.

Вызываются неизменённые Campaign.screen/mechanisms; stable task identities, source savepoints, continuation seeds и ledger сохраняются. Нового statistical decision или ремонта не добавлено. Config и девять scientific files соответствуют исходным SHA. Durable `recorded_resume_plan.json` живёт в игнорируемом runs/, содержит SHA решения/config/code, stages и coverage; supervisor возобновляет wrapper. Обе операции запускаются по принятому решению независимо от нового incomplete gate. Готовые outcomes не обучаются заново; записанные numerical failures учитываются отдельно от missing work.

## Фактическое переключение и проверки

Существующий coordinator завершил 12 дополнительных mechanism jobs (312k updates), сохранив последний `natural_asterix_seed3_age200000:1`. Одноразовый helper переключил supervision после complete records и отсутствия живых leases. Срез [recorded_queue.json](20261012T1024Z_recorded_queue.json) фиксирует progress, PID, plan/decision SHA, фактически использованные helper hashes и scientific code checks. Ни одного LR-control outcome до переключения не было; исходные 192 natural outcomes сохранили SHA `a895b8b6ae778f3878ada2e8297b9945d0c4cad3931a885e4d2813ad6115cf16`.

Первый запуск wrapper получил PermissionError при nonblocking worker.lock, **до импорта learner и до Campaign**. Supervisor повторил запуск; новый coordinator выполняет source stage принятой очереди. Это transient infrastructure error, не numerical/repair outcome; обучения на незащищённой очереди не было. Breakout seed 5 завершён с прежнего savepoint: env 200499, optimizer clock 195500; Asterix seed 4 продолжает обучение. Cumulative accounting включает известную повторно потраченную работу после старого savepoint, model clocks остаются clocks сохранённой history.

Четыре stdlib tests прошли за 0.001s: decision переживает incomplete gate, partial coverage не повторяет готовую работу, numerical failure не становится бесконечным retry, stale mechanisms_complete не разрешает остановку running job/child writer. Tests не импортировали torch и не запускали optimizer updates. Windows spawn/main guard, same-lock ownership и stable IDs проверены независимым агентом.

## Замечания независимого аудита

Первый handoff helper использовал advisory boundary и затем ожидал остановки supervisor без final reread. Аудит указал гонку: coordinator мог начать следующий job в этом промежутке. В выполненном handoff интервал составил около 0.01s, сохранённый progress указывает последний finished probe, новых LR outcomes нет; нарушения текущей writer boundary не обнаружено. Фактически исполненная первая версия сохранена в commit `686a7a0` для точного provenance.

## Усиление после независимой проверки

Live handoff выведен из эксплуатации: текущий `handoff_recorded_queue.py` только записывает отдельный read-only audit, не останавливает и не запускает процессы и не перезаписывает исторический handoff record. Здоровый coordinator не перезапускался ради этих правок. Действующий supervisor уже использует durable wrapper, поэтому дальнейшее переключение не требуется.

Сохранённый plan сверяется с решением в manifest, canonical games/ages/menu/horizon и seeds/repeats операций. Finished outcome требует полного environment/gradient budget и прежнего trajectory grouping; только явный FloatingPointError учитывается как numerical failure. Standalone wrapper под worker.lock проверяет живые child leases до Campaign. Ошибки проверки после получения lock записываются до импорта learner, чтобы supervisor видел конкретную причину. Короткий retry lock допустим только при отсутствии здорового другого owner.

Прошли **9 stdlib tests за 0.015 s**, без torch и optimizer updates. Stop helper получил workspace path boundary, включая исключение соседних Thesis-copy/Thesis2; прошли syntax check и четыре path examples, сам stop helper не выполнялся. Независимый final review не нашёл блокирующих ошибок в resume/lease guard; замечание о соседнем пути исправлено. Heartbeat обновлён на durable wrapper/plan, cadence и quiet notification intent сохранены.

[Срез до последней path-boundary правки](20261012_controller_hardening.json) фиксирует живые PID 13144/17736, отдельные active/on-disk helper hashes, plan validation, 192 прежних outcomes с неизменным SHA и неизменные scientific files. [Финальный срез](20261012_controller_hardening_final.json) сохраняет последнюю stop-helper revision и проверки. Active controller продолжает версию из `686a7a0`; усиления wrapper применятся при следующем штатном запуске. Это operational provenance, не смена learner/source semantics. Правки `experiments/`/`configs/` и остальные gaps программы допускаются только новым output_root. [Completion audit](../../../../docs/research_program_completion_audit.md) остаётся открытым, reserved games закрыты до корректного signal/freeze.
