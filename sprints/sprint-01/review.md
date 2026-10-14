# Спринт 01 — итоги (черновик на 2026-10-09; спринт до 2026-10-12)

**Дополнение 12 октября; финальный review и встреча ещё не зафиксированы.** Confirmation обеих игр завершено: 96 screen + 96 independent continuation branches. Asterix optimizer +0.337500 → −0.316667, joint +0.770833 → −0.150000; устойчивость screening benefits не подтверждена. Symmetric checkpoint-winner vs continue: Asterix −0.020833, Breakout −0.252083. Итог — **неопределённо**, n=3/game; intervals не показываем. По Breakout secondary AUC head/joint отрицателен у всех трёх histories в обеих половинах (means −0.675625/−1.034583 → −0.658750/−0.804375); primary final не заменяется этим outcome. Данные/SHA/график/независимое воспроизведение: [full audit](../../research/results/minatar-repair-20261007/monitoring/20261011T1244Z.md). Выполняется предусмотренное расширение source seeds 3–5/repeats 4–5. После выключения машины восстановлена прежняя очередь с verified source savepoint; learner/config не менялись. Следующий спринт не начат.

**Дополнение 11 октября; review остаётся черновиком до встречи.** Screen 96/96 завершён, fixed-TD и synthetic LR controls завершены, confirmation Breakout 48/48. Прежний вывод ниже о вреде Adam не подтвердился: −0.500 в screen → +0.304167 в новых продолжениях, все три history сменили знак. Exploratory repeat-split выбор −0.252083 vs continue в среднем двух направлений; пользу diagnostics не подтвердили. Итог — **неопределённо**. Числа, исходный SHA и график: [audit 11 октября](../../research/results/minatar-repair-20261007/monitoring/20261011T0954Z.md). Asterix confirmation ещё выполняется, n=3 исходных histories на игру, без intervals. После перезагрузки восстановлена прежняя очередь с verified 40k/50k savepoints; новых экспериментов до встречи не запускали. Исторические пункты ниже относятся к срезу 9 октября, не являются текущими выводами.

План: [plan.md](plan.md). Задачи спринта по договорённости с руководителем — **литературное ревью** и **анализ работ на смежные темы**; эксперименты выполнены сверх плана. Итоги собраны по `research/decisions.md`, `research/NEXT.md` и git-истории; финальная версия — в конце спринта.

**Операционное дополнение 13 октября; review остаётся черновиком.** Unexpected OS shutdown оставил NUL supervisor/child control JSON при целых root 328 и шести completed local rows. Проверены исходные SHA, IDs и отсутствие writers; helper `b29eba8` после 14 stdlib checks/independent review восстановил completed jobs и observed resource counters. Только unfinished joint повторены от verified source. На 13:23 UTC root 336, следующие workers здоровы, общий recovery cost 458,043 updates включает discarded 58,043; FPE нет. Package 3 calendar 26.98 h с downtime закрыт до новых updates, package 4 в пределах бюджета. [Журнал и срез](../../research/results/minatar-repair-20261007/monitoring/20261013_nul_control_recovery.md). Научный вывод пока не меняется: expanded серия неполна. Следующий спринт не начинается без окончательного review/встречи.

**Операционное дополнение 12 октября.** При восстановлении обнаружен дефект: новый incomplete gate мог отменить уже записанное расширение. Исправлен внешний controller, scientific SHA сохранены; исполнившаяся версия — commit `686a7a0`. После независимого review live handoff retired, добавлены canonical plan/budget/live-lease checks. Прошли 9 stdlib tests и syntax/path checks stop helper без выполнения остановки; healthy campaign не перезапускалась для hardening. [Queue recovery и отдельные revision hashes](../../research/results/minatar-repair-20261007/monitoring/20261012T1024Z_queue_recovery.md). Это восстановление существующей очереди, не новая научная методика; остальные completion gaps остаются открыты.

**Второе прерывание 12 октября.** После сна 14:30–14:49 UTC живых Python не осталось; точная причина завершения не установлена. Verified два branch savepoints на 35k, hidden supervisor восстановил прежние ветки и загрузил готовую controller revision `88b50ab`. Независимый ledger audit подтвердил однократный reconciliation 70k; в срезе 14:59 UTC два workers продолжили обучение, 248 root outcomes без failures. Новая научная серия не завершена, выводы по repair не меняются. [Журнал восстановления](../../research/results/minatar-repair-20261007/monitoring/20261012_sleep_interruption.md).

## Сделано
**По задачам спринта**
- **Анализ смежных работ** — тематические заметки (параметры, optimizer/targets, данные и selection, on-policy), аудит авторского кода, границы новизны с ближайшими работами (TeLAPA, PAME, Injection) — `notes/`, `docs/novelty.md`.

**Подробно (включая сверх плана)**
- **Обзор литературы** — 41 источник (36 с чтением разделов полного текста, 5 по аннотации) — `docs/literature_review.md`, `docs/novelty.md`, `bibliography/`.
- **Постановка вопроса**: выбрать вмешательство по диагностике до него лучше, чем лучший постоянный repair — `docs/formalization.md`, `docs/research_program.md`.
- **Синтетический контроль** — 24 checkpoint, 288 веток — `docs/controlled_probe_results.md`.
- **Стенд и кампания** — DQN на MinAtar, 4 действия (continue, Adam reset, head reset, оба), парные сравнения — `experiments/`, `configs/research_program.json`; на 2026-10-09 готово 72 исхода из 96 основных, 0 сбоев.
- **Независимый аудит** двумя парами агентов (Proposer/Skeptic) — `research/debate/log.md`, `accepted.md`, `rejected.md`.
- **Операционные инструменты** — supervisor (исправлен подсчёт перезапусков), watchdog, `scripts/resume_campaign.ps1`, `scripts/stop_campaign.ps1`.
- **Публикация** — https://github.com/muitiiifruckt/trainability-repair-thesis.

## Не сделано и почему
- Личное чтение ключевых статей и шпаргалка — в процессе.
- Forward citations ближайших работ — не выполнено (`docs/novelty.md`).
- Основной screen не завершён (Asterix частично): кампания стояла ~22 ч и ещё ~10 ч из-за сна компьютера.
- Prereg плана анализа диагностик не заморожен: dry-run v3 выполнен частично, ложные срабатывания на сценарии с гетероскедастичностью 3.5% [2.5; 4.8] при цели ≤ ~3%, независимого воспроизведения нет (`research/debate/handoff_A2.md`).
- Правки генератора отчёта и whitelist признаков отложены до конца кампании (SHA кода зафиксирован).

## Выводы
**Можно (предварительно), Breakout, 3 истории, 12 ячеек:**
- Head reset и joint reset сразу роняют политику: 12/12 пар отрицательны, в среднем −1.83 (возраст 50k) и −4.28 (200k).
- Adam reset не улучшает итог ни в одной из трёх историй: −0.51 / −0.68 / −0.31, среднее −0.50 при continue ≈ 7.3.
- Head reset в среднем нейтрален (+0.07), joint −0.70.

**Нельзя:**
- Взаимодействие head × optimizer и зависимость от возраста: кандидаты с одной истории исчезли на трёх.
- Asterix: агент почти не научился (≈ 1.1 против 0.75 у случайной политики) — сравнение не интерпретируется.
- Польза диагностик: данных нет. Для честной проверки нужно порядка 24 историй (мощность «информативности» ≈ 28% при n=12 и ≈ 75% при n=24, симуляции B2).

## Что пошло не так (процесс, инфраструктура)
- «95% интервалы» при 1–2 историях были размахом повторов (реальное покрытие 0.44–0.71) — исправлено подавлением интервалов.
- Утечка будущих данных в синтетической диагностике (AUC 0.972) — снято.
- Параметры `gate_practical_margin` и `minimum_histories_for_selector` в конфиге код не читает; `heterogeneous` может срабатывать от различий между играми.
- Случайный коммит копий кода в `research/debate/tmp` (исправлено), случайный запуск `powercfg` агентом (без эффекта).
- Сон компьютера останавливает кампанию.

## Переносится в следующий спринт
- Завершение основного screen и анализ полной панели.
- Заморозка prereg после воспроизведения.

## Предложения для плана следующего спринта
- Цель — закрыть первый screen и решить, нужна ли репликация на 24 историях.
- Подготовка к встрече с руководителем: личное чтение 4–5 ключевых статей.

## Операционное дополнение 13 октября: power off и reused PID

После power off 14:28 UTC Windows переиспользовала PID worker для renderer. Это инфраструктурная причина блокировки resume, не repair outcome. Исправлена внешняя owner verification под worker.lock, 26 stdlib checks/independent review; приложение не останавливалось. Verified savepoints 10k continue repeats2–3 seed5 AST сохранены, workers продвинулись. На срезе 15:30 UTC root 360, expansion 168/192 неполон; reconciliation 25,601 учтён один раз, discarded observed 5,601 не скрыт. Scientific pilot conclusion остаётся неопределённым, reserved закрыты. Controller `568152b`/`e06f068`; [evidence](../../research/results/minatar-repair-20261007/monitoring/20261013_pid_reuse_recovery.md).

## Операционное дополнение 14 октября

Ночная инфраструктурная остановка восстановлена без изменения frozen science/seeds: два verified 35k joint checkpoints продолжены, 6 completed raw rows сохранены и импортированы. На срезе 07:45 UTC root 374, additional_sources182/192 неполно, failures 0; resource373,193 reconciled один раз, observed discarded 3,193 включён. Причина завершения Python не установлена; sleep/wake подтверждены отдельно, не считать их repair outcome. Один supervisor/coordinator и два workers здоровы, package 4<24h/12M; pilot вывод остаётся неопределённым. [SHA и audit](../../research/results/minatar-repair-20261007/monitoring/20261014_power_interruption.md).

Восстановленные joint repeats0/1 завершены на срезе 07:48 UTC: root 376, physical ledgers 201,755 / 201,438, package 4 **2,466,837**; remaining 30k учтены, failures 0. Следующая wave выполняется, additional_sources 184/192 ещё неполно.
