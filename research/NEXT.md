# NEXT — читать первым в каждой сессии

## 2026-11-19 — Sprint 07 завершён

Проверены 12 новых веток по 32768 шагов, три контроля из проверенных префиксов спринта 06. Overhead mean 2,53%, median 3,00%, битовое равенство подтверждено. Контраст pulse−neither −0,0124, описательный CI95 [−0,1119; +0,0871], `practical_benefit_not_shown` при пороге +0,1. При отключении поправки mean TV 0,724. Один цикл PPO с резким отключением поправки не дал требуемой пользы на этом горизонте. Остались открытыми эффекты более длительного контроля и самого отключения. Проверка завершена, активных очередей нет. Команды воспроизведения: `-m scripts.sprint07_pulse calibrate` и `-m scripts.sprint07_pulse science` после PASS, корни `sprint07_pulse_{cal,live}_v1` зарезервированы под завершённые результаты. [Итоги](../sprints/sprint-07/review.md)

## 2026-11-18 — Sprint 06 завершён

Все 36 научных, 24 калибровочных и 6 пилотных веток проверены; семь тестов прошли. Очередей нет. Итоги: `sprints/sprint-06/review.md`, `notes/sprint06_results.md`; JSON/SHA в `research/results/sprint06-match/`. Основной matched−reset AUC +0.04577, описательный CI90 [−0.03723; +0.12878], CI95 [−0.06011; +0.15166], verdict `uncertain`: margin ±0.1 не выполнен. Средние AUC neither/reset/matched 1.40505/3.91462/3.96039. Начальные policy/value, первый batch/targets/RNG и evaluation return matched/reset совпали в 12/12 парах, anchors неизменны. Frozen поправка действует всю траекторию; чистый механизм исследования среды и пластичность тела не доказаны. Не расширять seeds/горизонт/margin ради эквивалентности. Следующий план не создан: сначала выбрать ценный вопрос для независимого подтверждения или более строгого контроля поведения. Старые результаты не менять; selector и новые серии не запускать автоматически. Тяжёлые ревью запрещены. Ниже — исторические записи.

## 2026-11-18 — Sprint 06: научная очередь запущена

Семь проверок, пилот s0/s1 и отдельная калибровка прошли. Mean overhead neither/reset/matched 0.3258%/0.5391%/0.4474%, median ≤5%, без исключения. Артефакты 24 калибровочных и 6 пилотных веток проверены. Идут 36 научных веток, два workers, `runs/causal-trainability/sprint06_match_live_v1_20261118`. Freeze `a1b57a0`. При перезапуске сначала проверить процессы, затем та же команда `-m causal_trainability.behavior_match queue --workers 2`; не менять пакет/протокол. После завершения: `scripts/export_sprint06_match.py` проверяет артефакты и независимо пересчитывает AUC/интервалы; затем заметка, review и решение по основному matched−reset критерию. Базовые результаты спринта 05 не менять. Ниже — исторические записи.

## 2026-11-18 — Sprint 06: равная начальная политика

План `sprints/sprint-06/plan.md`, протокол `notes/sprint06_match_prereg.md`, runner `causal_trainability/behavior_match.py`. Neither/reset/matched: matched сохраняет raw actor и использует фиксированную замороженную поправку для точного равенства начальной политики reset. Это меняет функциональное представление; не чистая медиация через исследование среды. Семь проверок прошли до пилота. Порядок CLI: `pilot`, `calibrate`, после обоих PASS `queue --workers 2` через `-m causal_trainability.behavior_match`. Корни `sprint06_match_{pilot,cal,live}_v1_20261118` под `runs/causal-trainability/`. Пилот s0/s1 по 2048 steps, калибровка s0 по 131072 steps; после gates 36 научных веток, прежние исключения overhead не действуют. Не менять пакет/протокол в том же output root, не запускать второй экземпляр. STOP научной очереди запрещает новые ветки; attempts сохраняются. Тяжёлые ревью не запускать. Спринт 05 не переоткрывается; его результаты не меняются.

## 2026-11-16 — Sprint 05 завершён вместе с H1

Дополнение H1 выполнено в этом же спринте по просьбе пользователя. 48/48 веток и 32 технических запуска завершены, артефакты проверены, основные контрасты независимо пересчитаны. Actor effect +2.54413, CI97.5 [1.62547; 3.46278], `beneficial`; critic +0.01319, CI97.5 [−0.11557; 0.14194], `uncertain`. Старое тело, свежий Adam у всех, actions round5, 131072 steps, шесть уже изученных историй. Overhead ≤5% во всех ячейках без исключения. Подробности и SHA: `notes/sprint05_heads_results.md`, `research/results/sprint05-heads/`; общий review `sprints/sprint-05/review.md`. Активных очередей нет. Базовые 72 лабораторные ячейки не изменены. Следующий вопрос — различить изменение поведения и сброс параметров actor; для него ещё нужен отдельный план/протокол, запусков нет. Спринт 06 не создавался. Тяжёлые ревью запрещены. Ниже исторические записи.

## 2026-11-16 — Sprint 05 переоткрыт: actor/critic factorial

По явному расширению задачи пользователем раздельный сброс голов включён в раздел «Добавлено в ходе спринта» плана 05; спринт 06 не создаётся. Протокол `notes/sprint05_heads_prereg.md`, runner `causal_trainability/head_reset_factorial.py`. 48 on-policy веток по 131072 steps, четыре ячейки, свежий Adam во всех. Перед наукой отдельная последовательная калибровка ≤5% для всех четырёх ячеек; прежнее исключение overhead не применяется. Команды: `-m causal_trainability.head_reset_factorial calibrate`, после PASS `queue --workers 2`. Output roots `sprint05_heads_cal_v1_20261116` и `sprint05_heads_live_v1_20261116` под `runs/causal-trainability/`. Не запускать два экземпляра; STOP в корне научной очереди запрещает новые ветки. Завершённую базовую сводку 72 ячеек не менять. Тяжёлые ревью не запускать.

## 2026-11-16 — Sprint 05 завершён

Итоги: `sprints/sprint-05/review.md`, `notes/sprint05_results.md`; переносимая сводка и SHA в `research/results/sprint05/`. Все 72 научных ячейки и 64 технических проверены; очередей нет. Результат: `practical_deficit_not_shown`, fresh−aged −0.16167, описательный CI [−0.21290; −0.11044], aged лучше на общей цели в 6/6 изученных историй. Это не независимое подтверждение и не живой return. Обе калибровки остаются FAIL по 5%; пользователь разрешил исключение только для этих 72 ячеек, не переносить его на другие серии. Следующий план ещё не создан: исходить из review, готовить проверку раздельного вклада actor/critic в пользу ремонта; не подбирать новые seeds/горизонт ради дефицита. Тяжёлые ревью запрещены. Ниже — исторические записи.

## 2026-11-16 — Sprint 05: фиксированный общий batch

План `sprints/sprint-05/plan.md`. Runner v2 и пять проверок готовы; обе калибровки FAIL по overhead. Пользователь явно разрешил исключение только для 72 ячеек: `notes/sprint05_overhead_exception.md`, freeze `eea5cbd`. Запуск: `.venv/Scripts/python.exe -m scripts.run_sprint05_exception`; output root `runs/causal-trainability/sprint05_common_exception_20261116`. Не запускать второй экземпляр без проверки процессов. Manifest сверяет код, протоколы, runner, исключение и исходную калибровку; STOP в корне прекращает новые группы. Незавершённые группы сохраняются как attempts, завершённые проверяются в summary. После завершения проверить артефакты, экспортировать summary, написать заметку и review, обновить статусы. Тяжёлые ревью не запускать.

## 2026-11-16 — Sprint 04 завершён в объёме плана

Итоги: `sprints/sprint-04/review.md`. Все четыре задачи плана выполнены: разбор, протокол, технический пилот, решение о следующем дизайне. Калибровка v2 завершена: PASS, mean/median fresh 0.4204%/2.4121%, aged 2.7941%/4.2949%. Проверены артефакты 16 запусков. Старый FAIL сохранён. Сводка: `research/results/sprint04/calibration_v2.json`; freeze `7c78c22`. Активных очередей нет. Научный протокол: `notes/sprint04_common_batch_prereg.md`; реализация общего batch, её тесты и собственный telemetry gate ещё не выполнены. Не переносить на него PASS on-policy runner. Дальнейшие тяжёлые ревью запрещены пользователем. Ниже — исторические записи.

## 2026-11-13 — Sprint 04 в работе

Разбор готовых данных завершён: `notes/sprint04_diagnosis.md`. Нулевые головы и свежий Adam обеспечили одинаковые исходные выходы и первый rollout на двух development-историях. Пилот завершён, процессов очереди нет; logging/resume побитово совпали. Overhead gate FAIL (fresh mean 7.3763% > 5%); научные серии не запущены. Протокол и решение: `notes/sprint04_prereg.md`, `notes/sprint04_pilot_result.md`; сводка: `research/results/sprint04/technical_pilot.json`. Код freeze `d4fa6dc`, output root `runs/causal-trainability/sprint04_pilot_20261113` не переиспользовать. Далее — новая фиксированная калибровка наблюдаемости и отдельный научный протокол контроля на общем batch с фиксированными targets; параметры и пороги ещё предстоит зафиксировать. Тяжёлые ревью не запускать.

## Sprint 03: проверки завершены

Итоги: `sprints/sprint-03/review.md`. Сводки с SHA256: `research/results/sprint03-final/manifest.json`. Все серии и анализ v3 завершены. Selector не запускается; потеря обучаемости в RL не установлена. Следующий шаг — разбор сопоставимости actor/critic до нового пилота. Ниже сохранены исторические инструкции запуска.


## 2026-11-09 — подготовка возобновляемой очереди Sprint 03

Пользователь разрешил оставшиеся малые серии и субагентов; первичное ревью выполнено; дальнейшие тяжёлые ревью запрещены пользователем. Протоколы: `notes/sprint03_live.md`, `notes/ppo_probe_v3.md`, `notes/two_state_v2.md`; ревью — `research/sprint03_review_20261109.md`. 14 новых тестов прошли. Эксперименты и `configs/` DQN не изменялись.

Контроллер: `.venv/Scripts/python.exe -m scripts.run_sprint03`. Состояние: `runs/causal-trainability/sprint03_controller_20261109/state.json`, логи этапов рядом, разрешения S6 — `gates.json`. Сначала три ПОСЛЕДОВАТЕЛЬНЫХ timing-only пилота без параллельного обучения; затем только серии со своим green overhead gate (mean и median<=5%, bit equality). Не запускать другие тяжёлые задачи рядом. Научные output roots: `two_state_v2_20261109`, `probe_v3`, `sprint03_live_20261109` под `runs/causal-trainability/`. Очередь live — максимум2 workers, прочие серии по1, между сериями последовательность.

Остановка: создать файл `STOP` в каталоге контроллера; для уже запущенной live очереди также в `runs/causal-trainability/sprint03_live_20261109/` (текущие ветки завершаются). Истории probe/S5 завершаются своим подпроцессом до перехода контроллера. Не останавливать процессы по старому PID без проверки владельца/identity. Контроллер перезапускается той же командой только после проверки отсутствия активного экземпляра; manifest не допускает смену кода. Завершённые артефакты проверяются; partial S5 требует ручного нового attempt, не удаления.

S3/S4 используют108 общих веток; offline v3 — follow-up тех же histories, не независимая confirmation. Отдельно выполнена содержательная смена sticky-action probability .1→.25. Optional возрастной контроль пока не включён. До анализа всех summary не закрывать спринт и не писать финальный научный review.

Обновлено: 2026-10-27. Практики: [docs/agent_research_practices.md](../docs/agent_research_practices.md).

## 27 октября — T7 закрыт, начат спринт 03

- **T7, дефицит адаптации PPO.** Этап A — дефицита нет; этап B — «дефицит есть» на `actions`, «практически нет» на `channels` ([ppo_deficit.md](../notes/ppo_deficit.md)). Это дефицит адаптации, а не показанная потеря обучаемости ([decisions.md](decisions.md)).
- **Текущий спринт** — [sprint-03/plan.md](../sprints/sprint-03/plan.md): критерии до экспериментов, затем малые предрегистрированные проверки.

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
