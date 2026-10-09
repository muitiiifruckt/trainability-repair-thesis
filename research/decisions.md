# Журнал решений

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
