# Журнал решений

## 2026-10-27 — T7 закрыт; дефицит адаптации не равен потере обучаемости

1. **Вердикт T7 по предрегистрации** ([ppo_deficit.md](../notes/ppo_deficit.md), [stage_A.json](results/ppo-deficit/stage_A.json), [stage_B.json](results/ppo-deficit/stage_B.json)):
   - после одной смены (этап A) дефицит не показан: `channels` — «практически нет», `actions` — «неопределённо»;
   - после цепочки из 4 смен (этап B) на `actions` «дефицит есть»: +1.44, 97.5% [+0.48, +2.40], 6/6. На `channels` — «практически нет»: −1.42, старый агент лучше в 6/6.
2. **Этот вердикт не читается как потеря обучаемости.** Руководитель (ответ на план, 25.10): разница «свежий − старый» после обратимой перестановки может объясняться стартовой политикой и переносом. Кроме того, на этапе B старый агент старше (2M шагов) и прошёл 4 смены — дизайн не разделяет эти вклады. Разведка объяснения не различает:
   - стартовый return хуже примерно так же, как на этапе A, где дефицита не было;
   - от этапа A этап B отличают более уверенная политика (энтропия 0.24–0.57 против 0.62–1.02) и плато в 5 из 12 старых веток. Это совместимо и с ловушкой исследования, и с потерей обучаемости.
3. **Живые ветки ремонтов (T8) и лабораторный замер v2 (T9) не запускаются.** Сначала — шаги руководителя:
   - выгрузить все 576 исходов DQN с хешами кода;
   - показать побитовое продолжение PPO из настоящего чекпоинта (в T5 проверено только тестом на маленькой конфигурации, и на GitHub этого нет);
   - малый предрегистрированный тест «ремонт против `continue` сверх шума повторов»;
   - пример на двух состояниях с одинаковым горизонтом;
   - эксперимент, отличающий потерю обучаемости от плохого переноса: сопоставимый старт, обратная перестановка, содержательная смена.
4. **Предсказание по «встречной смене» действий не подтвердилось** ни на цепочках s3–s5, ни на раунде 5. Записано в `cf2e026` до того, как были вычислены AUC цепочек s3–s5 и раунда 5.

## 2026-10-23 — стенд PPO готов; первый лабораторный замер требует другого исхода

1. **Стенд PPO принят для следующего спринта** ([ppo_stand.md](../notes/ppo_stand.md)): все 8 прогонов пилота учатся (Breakout `cnn` 5.5 / 5.2 против 0.5 у случайной политики), 1M шагов — 3.5–4.5 мин в один поток. Гибриды на реальных чекпоинтах воспроизводят состояние побитово. Накладные расходы телеметрии в пределах 5% только по медиане (4.6%, одна конфигурация) — перемерить на ячейках следующего спринта.
2. **Лабораторный исход T4 в нынешнем виде для решений не годится** ([ppo_lab.md](../notes/ppo_lab.md)). Value-контраст по весам устойчив по знаку, но смешан с начальным loss (старый critic на данных θ₀ сдвинут по шкале); policy-контрасты упёрлись в потолок клипа. До живых веток — новая предрегистрация: исход value, не зависящий от начального loss, и меньший K для policy.
3. **Телеметрию базового обучения отложенных игры и архитектуры (Space Invaders, `cnn_ln`) видели** на графике пилота: rank, мёртвые нейроны, √v. Исходов ремонтов на них нет. Правило выбора ремонта не будет подбираться по этим кривым; если признак из них войдёт в правило, это нужно отметить при заморозке.

## 2026-10-22, вечер — DQN-пилот закрыт на панели 576

1. **Кампания `minatar-repair-20261007` остановлена** штатным `scripts/stop_campaign.ps1` на 576 исходах (22:48). Стадии `mechanisms`, `report` и `auto()` не запускались; reserved-игры не открывались. Новых вложений в DQN нет.
2. **Итоги разведочные** ([отчёт T0](results/minatar-repair-20261007/monitoring/20261022_full_576.md), 6 историй на игру, 20 интервалов без поправки):
   - оба сброса (голова + Adam): Asterix +0.26 [+0.10, +0.42], 5/6; Breakout −0.52 [−0.75, −0.28], 0/6. Знак одинаков во всех трёх парах повторов;
   - отдельный сброс Adam на финальный return обнаружимого эффекта не дал ни в одной игре;
   - выбор ремонта под чекпоинт не лучше постоянного ремонта на игру (Breakout −0.21 [−0.39, −0.03], 0/6): разброс между повторами сравним с разницей между ремонтами;
   - fixed-TD считает сброс головы вредным на всех 18 чекпоинтах, а онлайн-знак зависит от игры. Это согласуется с риском карточки «выигрыш на общем батче может не переноситься», но не проверяет его: другой горизонт и estimand, 4–5 историй, post hoc.
3. **Следствие для PPO-линии:** выбор ремонта нужно проверять против лучшего постоянного ремонта, а не только против `continue`. Шум между повторами одной истории нужно оценивать до того, как говорить о выборе под конкретный чекпоинт.

## 2026-10-22 — тема по карточке руководителя; телеметрия собирается всегда

1. **Основная линия — карточка руководителя** ([docs/topic_card.md](../docs/topic_card.md)): on-policy PPO, скрещивание частей состояния, проверка решения на отложенных средах. DQN-кампания `minatar-repair-20261007` — пилот в off-policy-режиме; её остаток (72 ветки) досчитывается, новых вложений в DQN нет. План — [спринт 02](../sprints/sprint-02/plan.md), ждёт согласования руководителя.
2. **Телеметрия по слоям и состоянию Adam собирается всегда, фоном, во всех новых запусках**, а не только в экспериментах, где она нужна сразу. Это нужно, чтобы помимо самих экспериментов потом провести сквозной анализ того, как менялась сеть. Правила — [observability_spec.md](../docs/observability_spec.md), раздел «Принцип»: единый формат и индекс запусков, запись отдельным потоком, побитовая проверка, накладные расходы ≤ 5%. Широкая телеметрия служит разведке; подтверждающие гипотезы по-прежнему проверяются только из prereg.

## 2026-10-14, 07:45 UTC — продолжить verified joint branches после прекращения процессов

Причина прекращения Python около 13 октября 18:16UTC неизвестна; confirmed sleep/wake не доказывает её, queued WER с апрельскими/сентябрьскими dumps не является новым BSOD. Все JSON valid, frozen hashes/recorded plan и два joint 35kSHA verified. Один hidden supervisor продолжил прежние IDs/seeds AST seed 5 age 200k repeats0/1, без scientific edits. Шесть completed rows импортированы неизменными (root368→374); physical373,193 reconciled один раз (2,063,644→2,436,837), observed discarded 3,193 учтён. Workers продвигаются, failures 0, package 4<24h/<12M. Expanded series неполна; pilot conclusion неопределённый, reserved закрыты. [Операционный журнал](results/minatar-repair-20261007/monitoring/20261014_power_interruption.md).

Восстановленные joint repeats0/1 завершены на срезе 07:48 UTC: root 376, physical ledgers 201,755 / 201,438, package 4 **2,466,837**; remaining 30k учтены, failures 0. Следующая wave выполняется, additional_sources 184/192 ещё неполно.


## 2026-10-13, 15:30 UTC — сохранить очередь при переиспользовании PID после power off

Windows 1074/42 и hybrid boot 27 объясняют новое прекращение Python. PID2400 теперь принадлежит новому ChatGPT.exe renderer, а не прежнему worker; не останавливать его. Внешний controller проверяет image/creation time, сохраняет unknown/Python owners и архивирует доказанный reused lease только под worker.lock. Исполнены commits `568152b`/`e06f068`, 26 stdlib checks и independent review; scientific code/config неизменны. Два verified savepoints continue Asterix seed5 age50k на 10k шагов продолжены с теми же seeds/IDs. Reconciliation 25,601 выполнен один раз (1,658,043 → 1,683,644), observed discarded 5,601 сохранены как ресурсные затраты. Root 360 без failures, expansion 168/192 неполон; итог pilot остаётся неопределённым, reserved закрыты. [Evidence](results/minatar-repair-20261007/monitoring/20261013_pid_reuse_recovery.md).

## 2026-10-13 — восстановить control metadata, сохранить completed repairs и физический расход

После подтверждённого unexpected Windows shutdown обнаружены полностью NUL supervisor и manifests/progress/resume JSON двух children. Root 328 outcomes и шесть local completed rows целы. Непроверенные scratch PT сохранены в quarantine; повторить только joint repeats 2/3 от verified Asterix seed 4, age 50k, сохранив original IDs/seeds. Внешний helper `b29eba8` восстанавливает completed jobs/counters по raw data и parent progress под двумя locks; 14 лёгких checks и independent review пройдены. Старые counters 179,051/178,992 reconciled один раз; после двух 50k reruns ledgers 229,051/228,992, total 458,043 включает 58,043 discarded updates. Не смешивать ресурсные counters с learner clocks. Package 3 закрыт с 26.98 h (downtime), package 4 открыт до fresh updates. На 13:23 UTC root 336, оба workers здоровы, FPE нет; серия неполна и reserved закрыты. Не менять frozen scientific code/config; durability исправлять следующей кампанией. [Evidence](results/minatar-repair-20261007/monitoring/20261013_nul_control_recovery.md).

## 2026-10-12, 14:55 UTC — восстановить прежние ветки после инфраструктурного прерывания

До запуска проверены отсутствие живых writers, frozen code/config и два branch SHA на completed_steps=35,000. Системный сон подтверждён, но причина завершения Python не установлена; старые queued WER не объявлять новым BSOD. Возобновлён existing hidden supervisor, workers=2, прежние task IDs/seed 4/repeats 2–3. Штатный restart загрузил подготовленную controller revision `88b50ab`, learner не менялся. Независимо подтверждён однократный reconciliation 70k updates (3,700,254 → 3,770,254); учёт ненаблюдаемой работы после последнего counter остаётся ограничением. На 14:59 UTC оба workers продолжили обучение, 248 root outcomes без failures. Прерывание учитывать отдельно от scientific results; reserved games закрыты. [Журнал и данные](results/minatar-repair-20261007/monitoring/20261012_sleep_interruption.md).

## 2026-10-12 — усиление controller и отказ от повторного live handoff

Исполненная версия orchestration сохранена commit `686a7a0`; дальнейшие усиления имеют отдельные hashes. После независимого замечания о TOCTOU live handoff полностью retired, CLI выполняет только read-only audit. Проверяем recorded decision по manifest и canonical операции, полный бюджет finished outcomes и только явный FloatingPointError как numerical failure. Standalone controller под worker.lock не создаёт Campaign при живых child leases. Девять stdlib tests прошли за 0.015 s без learner updates; stop script syntax и четыре workspace boundary examples проверены без остановки процессов. Замечание final reviewer о Thesis-copy/Thesis2 устранено. Healthy PID 13144/17736 оставлены выполнять source queue; будущая wrapper revision применяется при следующем штатном запуске. Heartbeat prompt обновлён с сохранением cadence/quiet intent. [Проверки и provenance](results/minatar-repair-20261007/monitoring/20261012T1024Z_queue_recovery.md).

## 2026-10-12 — записанное расширение восстанавливать до нового gate

Partial imported checkpoints превращают global readiness в incomplete; frozen auto не читает принятое increase_resolution и мог пропустить natural seeds 3–5/repeats 4–5 после restart. Дефект воспроизведён независимо. В scripts добавлена lock-owning orchestration принятого решения; она вызывает frozen Campaign methods, сохраняет task identities/accounting и не меняет scientific config/code. Supervisor возобновляет durable plan до auto. Текущие probes сохранены перед handoff; первая transient lock denial не выполняла learner updates, повторный запуск продолжил sources. Четыре лёгких tests прошли. Независимые замечания к первой handoff/plan-validation версии сохранены для усиления до её следующего использования; фактические hashes — [операционный audit](results/minatar-repair-20261007/monitoring/20261012T1024Z_queue_recovery.md).

## 2026-10-12 — полное confirmation и увеличение разрешения

1. **Снять screening benefits Asterix как подтверждённый результат.** На independent repeats Adam +0.337500 → −0.316667 (все три знака сменились), joint +0.770833 → −0.150000. Положительный pooled joint +0.310417 у всех трёх histories не заменяет независимое подтверждение: две histories отрицательны в confirmation. Вместе с Breakout результат остаётся «неопределённо», n=3/game, без intervals.
2. **Selection benefit не установлен.** Symmetric repeat-split winner vs continue: Asterix −0.020833, Breakout −0.252083. Малый Asterix contrast с SBS чувствителен к одной точной ничьей. Это outcome-label check на прежних histories, не diagnostic transfer; reserved games закрыты.
3. **Сохранить secondary recovery costs отдельно от primary final.** Breakout head/joint AUC effects отрицательны у всех трёх histories в обеих половинах: means −0.675625/−1.034583 → −0.658750/−0.804375. Final outcome после просмотра не заменяется AUC. Raw curve/harm summaries сохранены, zero panels отдельно.
4. **Продолжить авторизованное расширение.** Runner при gate=uncertain запустил seeds 3–5, затем repeats 4–5 на seeds 0–5; итоговая natural панель 576 branches, 6 source histories/game. Отличать variance внутри history от различий между агентами; не считать repeats новыми histories. Следом — запланированные LR controls, stress при отсутствии signal.
5. **Возобновить ту же очередь после выключения машины.** Windows event 1074 совпадает с прекращением progress. Живых процессов/leases нет, code/config hashes и source savepoint Breakout seed 5 на 190k verified. Hidden resume с workers=2; metadata/calendar downtime не переписывать, исправление strict deadline — новая кампания.

Источники: [192-branch audit](results/minatar-repair-20261007/monitoring/20261011T1244Z.md), SHA `a895b8b6ae778f3878ada2e8297b9945d0c4cad3931a885e4d2813ad6115cf16`; независимое воспроизведение stdlib read-only агентом. Learner/config не менялись, проверки не запускали optimizer updates.

## 2026-10-11 — знак эффекта Adam не подтвердился и очередь восстановлена

1. **Снять стабильный вред optimizer reset в Breakout.** На полных 48 confirmation branches repeats 2/3 mean effect +0.304167 против −0.500 в исходном screen; все три source histories сменили знак. Это новые continuation repeats тех же трёх histories. Положительную confirmation не объявлять доказанной пользой; итог — «неопределённо».
2. **Не объявлять selection signal по положительному контрасту с шумным SBS.** Checkpoint winner screen→confirm +0.029167 vs continue, обратное направление −0.533333; среднее −0.252083. Среднее +0.254167 vs training-half SBS не компенсирует отсутствие пользы vs continue. Это exploratory outcome-based label stability, не diagnostic predictor или перенос на unseen history.
3. **Восстановить ту же очередь после прекращения процессов.** Живых coordinator/workers/leases нет; проверены исходные SHA и два Asterix savepoints 40k/50k. Hidden supervisor возобновлён с workers=2, без правки learner/config/seeds. Перезагрузка Windows зафиксирована, точная причина остановки progress не установлена; не записывать её как MemoryError или numerical result.
4. **Оставить wall-budget defect явно открытым.** Пакет 1 автоматически закрыт с 7,561,975 updates/61.02 h календарного времени включая простой; новый пакет 2 начат перед wave. Строгий 24h deadline требует новой кампании. Исходные metadata сохраняются.
5. **Следовать текущему спринту:** продолжать разрешённую фоновую очередь, новых методических экспериментов до встречи не запускать. Исправления `experiments/`/`configs/` — только для нового output_root. Reserved transfer закрыт до корректного подтверждения signal/freeze; текущий exploratory gate не достаточен.

Данные: [audit](results/minatar-repair-20261007/monitoring/20261011T0954Z.md), compact JSON с исходным SHA `594efebfc577c50b9a73b348b34fbd9c6275f219c97019c9ec6781da0c8e9402`. Числа и matched-seed/budget checks воспроизведены независимым read-only агентом; n=3/game, intervals не вычислялись.

## 2026-10-05 — первая постановка после поиска литературы

1. **Не заявлять новизну fixed-budget plasticity.** Такая формализация уже есть у Lyle et al. (2023).
2. **Не заявлять новизну clone-and-intervene.** Plasticity Injection (2023) прямо описывает такой диагностический эксперимент.
3. **Не заявлять новизну любого adaptive repair.** Уже существуют ReDo, Adaptive RR, automatic soft reset и learned optimizers; нужны сравнения с ближайшими методами.
4. **Рабочий вклад:** prospective выбор среди нескольких типов repair по pre-repair features, с оценкой regret на held-out environment families. Это кандидат на вклад, не подтверждённая свободная ниша.
5. **Два уровня проверки:** fixed-data/fixed-target probes для оптимизации и online RL для фактической полезности.
6. **Первые actions узкие и воспроизводимые.** Параметрный repair и optimizer reset пересечь факторно; data/target interventions сначала считать лабораторными controls, пока нет доступного deployable варианта.
7. **Отдельно effect prediction и cause identification.** Успех selector допустим без доказательства четырёх исключающих друг друга причин.
8. **Сначала проверить heterogeneity.** Если один repair почти всегда лучший, сложный selector не нужен.
9. **Не начинать масштабное обучение по одному обзору.** Сначала проверка ближайших PDF/кода и небольшой пилот. Compute был неизвестен на первом проходе; замер и CPU-контроль записаны ниже.
10. **Сохранять отрицательные результаты.** Отсутствие преимуществ selector или необходимость комбинированного repair — полезные итоги, если дизайн корректен.

## 2026-10-05 — независимая проверка формализации

Исправлены четыре места: primary probe loss измеряет fitting, held-out loss отдельно generalization; data×target factorial использует shared target semantics; empirical winner не равен population oracle и не является строгой bound; practical utility использует total budget с вычетом diagnostics/repair costs. Также разделены held-out environment для selector и перенос агента в новую среду.

## 2026-10-05 — приложения, авторский код и первый выполненный контроль

1. **Узить selection claim:** TeLAPA уже выбирает по фактической short-budget adaptation. Наш кандидат — amortized prediction heterogeneous repair responses; добавить short-probe chooser и учёт offline label/training cost.
2. **Не переносить repair по названию:** Juliani `none` сбрасывает optimizer на shifts; Primacy меняет дополнительные states; Plasticine replacement Parameters не попадают в старый optimizer. Все findings относятся к указанным code commits, не доказывают происхождение paper results.
3. **Писать собственный маленький fork runner:** каталог диагностик/методов заимствовать после проверки definitions. Сторонние install/training scripts не запускались.
4. **Compute:** Python 3.13.12, torch 2.12.1+cu126; RTX 3050 Laptop с 4 GiB доступна. Первый контроль CPU float64/один thread; gymnasium/MinAtar пока не установлены. GPU не понадобилась; speed полного RL не измерялась.
5. **Контроль завершён:** 24 checkpoint/288 branches, шесть tests прошли, nonfinite failures нет, 36.22s. Response зависит от искусственного scenario и age; это preliminary semantics control, не RL validation.
6. **Timestep reset не очищает memory:** точный LR+epsilon schedule воспроизвёл t-reset. Этот control нужен для разделения clock/step-size и moment-memory эффектов.
7. **Не обучать selector на synthetic scenario labels:** diagnostics знают future training labels, histories связаны; 288 branches не независимые samples. Следующий шаг — fixed transitions/frozen TD, затем closed-loop return.
8. **Следующая опорная проверка:** final loss на common held-out objective и gain от unrepaired start; Double DQN freeze включает argmax selector. При data×target factorial сравнивать одинаковый evaluation objective, а не loss относительно разных teachers.
9. **Независимый шум для probing baseline:** short-ranking stream отделён от main continuation/final evaluation. Chooser получает short outcomes, основной diagnostic predictor их не видит; final/oracle outcomes закрыты обоим.

[Audit parameters](../notes/code_audit_parameters.md), [audit optimizer](../notes/code_audit_optimizer.md), [forward search](../notes/infrastructure_and_forward_search.md), [выполненный run](../docs/controlled_probe_results.md).

## 2026-10-09 — результаты дебата A/B (раунд 1) и решения координатора

1. **Power-оценка 2/7/62 истории отозвана** (единицы — повторы одной истории, z вместо t). `scripts/power_from_partial.py` переписан: единица — история, <3 историй -> insufficient.
2. **Шумовой пол принят:** SE парной разности final return 0.6–1.9 при 20 eval-эпизодах (A1, воспроизведено B). Ни один эффект на единственной завершённой истории не выходит за ~2 SE; классификация остаётся «неопределённо».
3. **AUC 0.972 `pre_future_training_loss` — утечка future labels, снято.** Baseline для будущего теста диагностик: age_only и pre_td_huber+recent_return.
4. **Whitelist признаков (патч B) проверен, но НЕ применён к `experiments/rl_analysis.py` во время кампании.** Причина: `runs/.../dependency_manifest.json` фиксирует SHA этого файла (76315e19…); правка в середине кампании нарушила бы провенанс результатов. Проверено: 21 реальный ключ diagnostics, имена whitelist совпадают с ними; на копии `research/debate/tmp/pkg0` (текущий код) тесты утечки красные (2 failures), на `pkg` (с патчем) 13/13 зелёные. План: применить после завершения screen к новому/отдельному анализу с записью обоих SHA; selector-анализ на blacklist-версии не считать подтверждающим.
5. **Атрибуция вреда head reset:** arm head_reset_sync как причина отвергнут (sync делает target мусором); нужны arms stale/sync/injection, вред головы — по разности stale−injection (согласование A/B в `research/debate/log.md`). Для development-probe, не для текущей кампании.
6. **Исследователь B остановлен пользователем** во втором раунде; не перезапускался. Артефакты B (патч, тесты, stale-target эксперименты) сохранены.

## 2026-10-09 — решения по аудиту B2 (отчёты, генератор, возобновление)
1. **Ручной текст отчёта вынесен** в `research/results/minatar-repair-20261007/NOTES_manual.md` (генератор пересоздаёт `REPORT.md` через `write_text` при ротации пакета ~2026-10-09 23:49 и может сам закоммитить results: `rl_runner.py:166-178, 519-547`). Поправки к неизменяемым снимкам — `monitoring/ERRATA.md`.
2. **«95% interval» при малом числе историй — не доверительный интервал** (размах повторов, `rl_analysis.py:153-185`); реальное покрытие 0.44/0.71/0.84/0.89/0.89/0.94 при 1/2/3/5/6/12 историях. Правка генератора (`rl_runner.py:519`, подпись PNG `rl_analysis.py:725`) — после кампании с записью нового SHA.
3. **Риск доступности F2:** `replace` без повторов в `save_state`/`source_savepoint` на OneDrive; три одинаковых сбоя -> supervisor выходит и кампания тихо стоит. Смягчение без правки кода: внешний `scripts/campaign_watchdog.py` (перезапуск supervisor, максимум 3 раза на одну ошибку).
4. **H-age неотделим от epsilon/replay при двух возрастах** (corr = ±1.0000 по 12 checkpoint). Разделить можно только возрастом 5k–100k (например 25k) — для нового development-пробы, не для текущего конфига.
5. **Расширение seeds 3,4,5 запускается только при gate=uncertain и learned_fraction>=0.5** (`rl_runner.py:610-617`): при heterogeneous/single_repair расширения не будет. Следствие: n=12 историй достижимо только в ветке «uncertain»; учитывать при планировании селектора.
6. **delta=0.2 sd недостижима и для оракула** (эффекты ячеек в sd возвратов: ~0.06–0.3, выигрыш оракула над константой <=0.07 sd) — критерий отрицательного вердикта prereg пересматривается A/B2.

## 2026-10-09 — причина простоев кампании: сон машины (B2, вход 3; проверено координатором)
1. Журнал питания Windows: сон 08.10 01:18–10:21 (9.05 ч) и 09.10 01:01–10:51 (9.85 ч, wake source: Power Button). Первый совпал с остановкой wave 5 («22-часовой простой»); во втором кампания пережила сон (те же PID), но не считалась.
2. Машина без системной батареи, питание от сети (`PowerLineStatus=Online`); сон при простое от сети отключён (индекс 0). Значит сон инициируется вручную (кнопка/меню), keep-awake через SetThreadExecutionState его не предотвратит; системные настройки питания **не меняются** без решения пользователя.
3. Следствие для планирования: ночью кампания не считается; календарная оценка = вычислительная x ~1.7 при ночных снах 9–10 ч. Чтобы кампания шла ночью, не усыплять машину (или отключить сон вручную).
4. **Инцидент B2:** в записи log.md обратные кавычки в неэкранированном heredoc выполнились шеллом, и `powercfg /change standby-timeout-dc 0` была запущена; команда завершилась ошибкой параметров (MSYS подменил `/change` путём). Координатор перепроверил: standby-timeout от сети 0, от батареи 0x258 (600 с) — без изменений. Правило: записи в log.md только через quoted heredoc/Write.

## 2026-10-09 — аудит B3 (selection/gate) и поправка к п.5 предыдущей записи
Проверено координатором по коду:
1. **ПОПРАВКА п.5:** фраза «при heterogeneous/single_repair расширения не будет» неверна. Условие расширения `rl_runner.py:610` — `learned_fraction >= .5 and gate == "uncertain"`; при 3 историях на игру gate структурно `uncertain` (`rl_analysis.py:348-350`), поэтому расширение seeds 3,4,5 произойдёт, пока нет `incomplete`. Реальный риск — `incomplete` (любая ветка с ошибкой/нечисловым `j_final`, расхождение бюджета, недостающая строка): тогда расширение не запускается и кампания закончится на 6 историях. В `Campaign.source` нет обработчика `FloatingPointError` (в ветке есть, `rl_runner.py:396`). Меры: перед `adaptive()` проверять natural-страту на ошибки; при изолированном сбое запускать расширение вручную с записью в decisions.
2. **D1 — параметры конфигурации не читаются:** `gate_practical_margin=0.05` и `minimum_histories_for_selector=6` (`configs/research_program.json:38`) нигде в `experiments/*.py` не используются (grep подтверждает); фактически порог gate = `heterogeneity_margin` по умолчанию 0.0 (`rl_analysis.py:446`), минимум историй зашит как «не меньше 5 на игру». Править код нельзя (SHA); формулировки в `docs/research_program.md` привести к коду, при анализе после кампании использовать заявленные 0.05 и 6 явно.
3. **D2 — gate не доказывает пользу selector:** `heterogeneous` может возникать из различий между играми (SBS одна на все игры в сырых единицах, срабатывает «любая игра»): симуляция B3 без эффектов внутри игры — positive 98/100 при различии между играми, 26/100 при различии между возрастами. Не считать `primary_gate = heterogeneous` свидетельством в пользу selector и не открывать frozen transfer на этом основании; добавить вторичный анализ преимущества над SBS по игре и игре x возраст (после кампании, с новым SHA).
4. **D4 — порог «выучен» мягок** (`rl_runner.py:301`): Asterix 50k j_pre 1.0/1.1/0.65 против random 0.75 статистически неотличимы (z≈0.7–0.9); учитывать при интерпретации Asterix.
5. **D6 — `immediate_change = j_immediate − j_pre` смещён** (continue: в среднем −0.867, отрицателен в 11 из 12 пар, не объяснено); использовать только парную разность с continue.
6. **Расхождения в документах** (правки текстов не внесены; список — `research/debate/log.md`, раздел B3-2): `docs/controlled_probe_results.md:40` (победители 7/4/9/4 верны для held-out MSE, не для training loss; в 12 из 24 checkpoint выигрыш <= 6.8%), `:44` (артефакта «независимого чтения» в репозитории нет), `docs/research_program_completion_audit.md:9` («все три Breakout выросли» — у seed 2 рост +1.4 в пределах шума). Утверждение «65 тестов» подтверждено `runs/final-tests.txt` (Ran 65), претензия B3 снята.
7. **Что выдержало проверку:** утечки SBS нет (история исключается целиком, `rl_analysis.py:217-218,439,288`), половины cross-fit не пересекаются; при нулевых эффектах cross-fit несмещён. **Можно утверждать по 48 строкам Breakout (3 истории):** head и joint reset сразу роняют политику (12/12 пар отрицательны; среднее −1.83 на 50k и −4.28 на 200k); знак эффекта optimizer reset на финальный return отрицателен у 3 из 3 историй (предварительно; p без поправки 0.041, при 3 историях знаково-перестановочный тест не может дать p < 0.25). **Нельзя:** вывод про joint, head, возраст, взаимодействие, диагностики, Asterix.
