# NEXT — читать первым в каждой сессии

Обновлено: 2026-10-27. Практики: [docs/agent_research_practices.md](../docs/agent_research_practices.md).

## 27 октября — T7 закрыт, начат спринт 03

- **T7, дефицит адаптации PPO.** Этап A — дефицита нет; этап B — «дефицит есть» на `actions`, «практически нет» на `channels` ([ppo_deficit.md](../notes/ppo_deficit.md)). Это дефицит адаптации, а не показанная потеря обучаемости (замечание руководителя, [decisions.md](decisions.md)). Запусков нет.
- **Текущий спринт** — [sprint-03/plan.md](../sprints/sprint-03/plan.md): ответ руководителю на три вопроса, затем малые предрегистрированные проверки.

## 22 октября — DQN-пилот закрыт

- Кампания остановлена на полной natural-панели **576 исходов** (22.10, 22:48); процессов нет. Восстановление после сбоя 19.10 — [20261022_recovery.md](results/minatar-repair-20261007/monitoring/20261022_recovery.md).
- Итоги — [отчёт T0](results/minatar-repair-20261007/monitoring/20261022_full_576.md) и запись в [decisions.md](decisions.md). Новых запусков DQN нет; основная линия — PPO по [карточке](../docs/topic_card.md), текущий спринт — [sprints/](../sprints/README.md).
- Записи ниже — история кампании; их счётчики не являются текущим состоянием.

## Текущее состояние — 14 октября

- **Ночная остановка восстановлена; обе unfinished joint branches завершены.** Причина завершения Python неизвестна, power intervals подтверждены отдельно. На 07:48 UTC: root **376 outcomes**, additional_sources **184/192**, следующая recorded wave repeats 2/3 выполняется; supervisor9616/coordinator17984 и два workers здоровы. Final ledgers 201,755 / 201,438, package 4 **2,466,837 updates / ~18.95 h**; reconciliation 373,193 и remaining 30k учтены однократно, discarded 3,193 включён. Scientific code/config/seeds неизменны, failures 0. [Журнал и SHA](results/minatar-repair-20261007/monitoring/20261014_power_interruption.md). Pilot conclusion остаётся неопределённым, reserved закрыты. Текущие PID проверять по native identity.

- **После power off 14:28 UTC та же очередь возобновлена.** PID2400 уже принадлежал новому ChatGPT.exe renderer, поэтому исправлена внешняя проверка владельца lease; приложение не трогали. Controller revisions `568152b`/`e06f068`, 26 stdlib checks. На 15:30 UTC: root 360 outcomes, additional_sources 168/192, additional_repeats ожидает; один supervisor/coordinator, два workers продолжают verified 10k savepoints. Physical reconciliation 25,601 однократный, package 4 — 1,683,644 updates; 5,601 discarded observed updates сохранены в затратах. Scientific code/config и seeds неизменны. [Журнал и SHA](results/minatar-repair-20261007/monitoring/20261013_pid_reuse_recovery.md). Проверять actual process identity вместе с PID; не останавливать посторонние процессы с переиспользованным номером.

- **После аварийного OS interruption восстановлены NUL control files.** На проверенном срезе 13:23 UTC: root 336 outcomes, additional_sources 144/192, additional_repeats ещё ожидает; package 4 — 458,043 updates, два workers здоровы. Только две unfinished joint branches повторены от verified source; шесть raw completed rows сохранены, предыдущий observed расход 358,043 reconciled один раз. Helper `b29eba8`, 14 checks passed; scientific code/config unchanged. [Журнал и SHA](results/minatar-repair-20261007/monitoring/20261013_nul_control_recovery.md). Текущие PID брать из valid supervisor.json и проверять alive. Повторный helper apply к уже восстановленным файлам не запускать; original NUL bytes/PT находятся в ignored recovery quarantine.

- Первый screen **96/96 и confirmation 96/96 завершены**; fixed-TD 24 jobs/744 branches и synthetic LR 1728 branches завершены. Verified pilot — **192 online outcomes**, source seeds 0–2, четыре repeats, исходных histories по-прежнему три на игру. Расширение seeds 3–5/repeats 4–5 выполняется; его outcomes учитывать отдельным срезом.
- **Первоначальные optimizer findings не подтвердились в обеих играх:** Breakout −0.500 → +0.304167; Asterix +0.337500 → −0.316667, все шесть source histories сменили знак. Asterix joint +0.770833 → −0.150000. Польза выбора vs continue не подтверждена. [Полный sourced audit](results/minatar-repair-20261007/monitoring/20261011T1244Z.md); [CURRENT_STATUS.md](results/minatar-repair-20261007/CURRENT_STATUS.md).
- После выключения Windows 11 октября 16:46 MSK процессов не было. Scientific SHA/config, отсутствие живых leases и source savepoint Breakout seed 5 на 190k env verified; hidden supervisor возобновлён 12 октября, workers=2, одна существующая очередь и прежние seeds. Текущие PID проверять по `supervisor.json`/manifest и реальному alive, не по stale `worker_status=running`.
- Frozen runner закрыл старый пакет с 61.02 h календарного времени, включая простой, и открыл пакет 2. Строгий wall ceiling остаётся дефектом; не трактовать это как отрицательный repair outcome и не корректировать metadata задним числом.
- Активен **спринт 01**, следующий план остаётся предварительным до review/встречи. Продолжается существующая фоновая кампания и её уже авторизованная очередь; новых серий сверх неё и изменений методики до встречи не добавлять. `experiments/`/`configs/` не менять; исправления — только новая кампания/new output_root. Не публиковать intervals при <5 histories, не использовать auto-report intervals n=3; после автоматического пересоздания текста запускать `scripts/suppress_small_history_intervals.py`.
- Seaquest/Freeway закрыты: exploratory heterogeneous gate текущего runner не заменяет корректное подтверждение и freeze. Пробелы программы — [completion audit](../docs/research_program_completion_audit.md).
- Следующая полная natural панель — 576 branches: 6 source histories на игру × 2 ages × 6 repeats × 4 actions. До её завершения не смешивать cohorts с разным числом repeats и не объявлять missing histories отрицательным repair outcome. Decision `increase_resolution` записан самим runner при exploratory gate=uncertain.
- **Исправлен вход в recorded queue:** partial additional checkpoints превращали новый gate в incomplete и могли отменить уже принятое расширение. Supervisor теперь запускает внешний `scripts/resume_recorded_research.py`, который владеет worker.lock и replay принятого решения выполняет до frozen auto. Plan — `runs/minatar-repair-20261007/recorded_resume_plan.json`. Scientific files/config не менялись; helper revisions имеют отдельные hashes. [Операционный audit](results/minatar-repair-20261007/monitoring/20261012T1024Z_queue_recovery.md).
- **Live handoff retired:** текущий `scripts/handoff_recorded_queue.py` только read-only audit. Усилены plan/decision/budget validation и standalone live-lease guard; 9 stdlib tests прошли, stop helper проверен без выполнения. Здоровый controller не перезапускать ради scripts hardening; active и on-disk revisions учитываются отдельно. Heartbeat продолжает следить за durable plan; прямой `auto --resume` при pending plan не запускать.
- **Повторный resume 12 октября после сна 14:30–14:49 UTC:** все прежние Python завершены, точная причина неизвестна; новых numerical/MemoryError нет. Verified два branch savepoints на 35k, восстановлены прежние seed/task IDs, hidden supervisor workers=2. Новый запуск загрузил controller `88b50ab`; reconciliation 70k выполнен один раз. Срез 14:59 UTC: PID 8440/14436 и два children живы, 248 root outcomes; текущие PID всегда проверять по alive. [Журнал и hashes](results/minatar-repair-20261007/monitoring/20261012_sleep_interruption.md). Бывшие PID 13144/17736 в ранних записях исторические.

Ниже — прежние записи для контекста; их счётчики 48/72 не являются текущим состоянием.

## Возобновлено (2026-10-09, после короткой остановки)
Кампания `minatar-repair-20261007` остановлена пользователем и затем возобновлена (supervisor + coordinator перезапущены, очередь продолжается с savepoint; 48 исходов на момент перезапуска, Breakout полностью, Asterix начат).
Остановить: `powershell -ExecutionPolicy Bypass -File scripts/stop_campaign.ps1`; возобновить: `scripts/resume_campaign.ps1`. Внешний watchdog: `.venv/Scripts/python.exe scripts/campaign_watchdog.py`.
Известный дефект supervisor (исправлен в `scripts/research_supervisor.py`): пустая сигнатура ошибки засчитывалась как «та же ошибка», после трёх обычных перезапусков он объявлял `attention_required`.
Дебат: A2 и B3 завершены (handoff_A2.md, handoff_B3.md), преемники A3 и B4 стартуют по ним. Не править `experiments/`/`configs/` до конца кампании (SHA в `dependency_manifest.json`).

## Состояние
- Кампания `minatar-repair-20261007`: online screen идёт (supervisor `scripts/research_supervisor.py --workers 2`, перезапущен 2026-10-08 после ~22 ч простоя).
- Живые данные: `runs/minatar-repair-20261007/{progress.json,outcomes.jsonl,supervisor.json}`. Конфигурацию не менять.
- Завершено 48 из 96 основных веток (2026-10-09: Breakout полностью — seeds 0,1,2 x 50k/200k; Asterix начат); живой отчёт — `research/results/minatar-repair-20261007/REPORT.md`.

## Внимание: сон машины
Кампания приостанавливается, когда Windows уходит в сон (ночью 9–10 ч); это не сбой, возобновление корректно (E1–E4 в `research/debate/log.md`). Сон от сети отключён, значит усыпление ручное. Не менять питание без решения пользователя.

## Порядок действий при возвращении
1. Проверить `supervisor.json` (status, restarts) и что растёт `package_updates`. Если процессов Python нет — перезапустить supervisor.
2. Пересчитать `scripts/power_from_partial.py`; при новых полных историях обновить `notes/parallel_vectors_20261008.md`.
3. Только содержательные изменения — в REPORT и снимок monitoring; коммит.

## Реестр гипотез (статус)
| id | гипотеза | статус | чем проверяется |
|---|---|---|---|
| H-int | head×optimizer взаимодействие отрицательно | не реплицировано на 2-й истории (−0.05 против −0.56; n=2, неопределённо) | contrast по всем историям, затем fixed-TD |
| H-split | вред joint reset — моменты Adam head vs body | proposed, не запущено | development probe с раздельным reset |
| H-age | знак head reset зависит от возраста | не подтверждена на 3 историях (+0.21 / −0.07) | age×repair в анализе; к тому же age = epsilon = replay |
| H-cost | immediate drop head reset учитывать в utility selector | proposed | immediate/recovery декомпозиция |
| H-eps | сброс Adam вреден у нас из-за малого ε (1e-8; у Rainbow/Asadi 1.5e-4): первые шаги после сброса почти одинаковой нормированной величины | proposed (2026-10-09), не запущено; поддержка: Lyle et al. 2023 Fig. 1 и App. A.1 — при резкой смене задачи Adam (ε=1e-9) расходится и насыщает ReLU, помогает b2=0.9 и ε̄=1e-3 (supervised MNIST, не RL; 2026-10-18) | arm optimizer_reset при ε=1.5e-4 и с LR-warmup; логировать ‖Δθ‖/‖θ‖ первых шагов |
| H-timing | у Asadi сброс совпадал с обновлением target и повторялся; у нас один сброс в произвольный момент | proposed (2026-10-09), не запущено | ветка «сброс в момент обновления target» и повторные сбросы |

## Дебат A/B (research/debate/)
- Раунд 1 завершён (A и B); см. `research/debate/log.md`, `accepted.md`, `rejected.md`, решения — `research/decisions.md` (2026-10-09).
- Раунд 2: A пишет prereg диагностик + dry-run процедуры (`scripts/debate_prereg_dryrun.py`); B остановлен пользователем, не перезапущен.
- Патч whitelist (`research/debate/patch_whitelist.diff`) отложен до конца кампании (SHA в dependency_manifest).

- Проект следующей кампании (не запущен): [docs/next_campaign_proposal.md](../docs/next_campaign_proposal.md).
- Прогноз: после screen решить, нужна ли репликация (n>=24 историй, 4 repeats, >=100 eval-эпизодов); prereg v3 финализирует A.

## Контекст агентов дебата
Координатор измеряет контекст субагентов по полю `usage` в их транскрипте (без чтения транскрипта в основной контекст): input + cache_read + cache_creation последнего сообщения.
Замер 2026-10-09: A ~190k токенов (близко к пределу), у B2 транскрипт недоступен (0 байт). Правило: агенты ведут `research/debate/handoff_A.md` / `handoff_B2.md`;
при контексте ~150k+ агент дописывает handoff, возвращает отчёт и завершается, координатор запускает преемника с чистым контекстом по handoff-файлу.

## Правила
- Числа в выводах только из файлов с SHA; reserved-игры закрыты до freeze; повторы одной истории не независимы.
