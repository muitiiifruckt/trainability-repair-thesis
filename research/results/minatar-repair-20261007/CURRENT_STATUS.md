# Состояние исследования, 11 октября 2026

**Первый screen завершён (96/96), новые продолжения Breakout тоже завершены (48/48). Вред сброса Adam из первого screen не подтвердился.** В root на зафиксированном срезе 144 online outcomes; confirmation Asterix ещё выполняется. Source histories по-прежнему три на игру: возраста и repeats не увеличивают число независимых histories.

Для Breakout mean final-return effects относительно matched continue, с равным весом возрастов/repeats внутри history:

- Adam reset: **−0.500 → +0.304167** в confirmation repeats 2/3. Все три history сменили знак.
- Head reset: +0.070833 → −0.512500.
- Head + Adam reset: −0.695833 → −0.191667.
- Head×optimizer interaction: −0.266667 → +0.016667.

Новый положительный знак Adam не является доказанной пользой: первые и новые независимые продолжения тех же histories противоречат друг другу. По всем четырём repeats mean Adam effect −0.097917; объединение не объясняет нестабильность. **Классификация: неопределённо**, intervals при n=3 не показываем. Прежнее различие игр из screen нельзя представлять как подтверждённый optimizer contrast. Asterix screen effects +0.3375/−0.004167/+0.770833 для Adam/head/joint остаются screening findings до завершения его confirmation.

Exploratory выбор checkpoint winner по двум trial repeats: screen→confirm +0.029167 vs continue, обратно −0.533333; среднее **−0.252083 vs continue**. Положительный контраст +0.254167 с training-half SBS не доказывает пользу: SBS тоже меняется между половинами. Это проверка устойчивости outcome labels, использующая пробные outcomes тех же histories; pre-repair diagnostic selector и перенос на unseen history здесь не проверялись.

У 144 завершённых веток нет failures, дублей или нечисловых finals; matched continuation/checkpoint/evaluation seeds и бюджеты 50k/50k проверены, evaluation means реконструированы по 20 episodes. Числа и checks независимо воспроизведены. Источник SHA256: `594efebfc577c50b9a73b348b34fbd9c6275f219c97019c9ec6781da0c8e9402`.

Полный журнал, компактные данные и график: [audit 11 октября](monitoring/20261011T0954Z.md), [JSON](monitoring/20261011T0954Z_confirmation.json), [history effects](monitoring/confirmation_effects_20261011T0954Z.png). Предыдущий screen сохранён отдельно в `monitoring/20261009T1310Z.*`; его описание относится только к repeats 0/1.

Fixed-TD 24 jobs/744 branches и synthetic LR controls 1728 branches завершены. Совместный reset хуже continue на fixed-TD после 2000 updates во всех 24 jobs; это fitting outcome, не return. Synthetic scale drop меняет optimizer effect с LR; проценты чувствительны к estimator. Пять float32 timestep controls превышают declared tolerance; FPE отсутствуют, длинный precision follow-up отложен до новой кампании. Подробный журнал — [controls audit](monitoring/20261009T1410Z.md).

11 октября процессов прежнего supervisor/coordinator/workers не было; последний root progress — с 9 октября. Windows перезагружалась, точная причина остановки progress не установлена. Проверены исходные SHA/config и обе сохранённые ветки Asterix на 40k/50k; возобновлена одна существующая очередь с двумя workers и прежними seeds. [Restart preflight](monitoring/20261011T0949Z_restart.json). Актуальные PID/рост clocks проверяются по live manifest/progress и реальным процессам.

Frozen runner закрыл пакет 1 с 7,561,975 accounted updates и 61.02 h календарного времени, включая длительный простой; пакет 2 начат перед следующей wave. Строгий 24h wall ceiling не обеспечен, metadata не переписывались. Потолок 12M updates и возобновляемая очередь остаются действующими. Инфраструктурные ограничения не являются отрицательными результатами repairs.

Продолжается существующая программа. Новых методических серий к ней до встречи не добавляем; активен спринт 01, следующий план предварительный. `experiments/` и `configs/` во время кампании не меняются; исправления требуют нового output_root. [Completion audit](../../../docs/research_program_completion_audit.md) остаётся открытым. Seaquest/Freeway закрыты до корректного подтверждения signal и protocol freeze; текущего exploratory gate недостаточно.

Автоматический `REPORT.md` смешивает доступные repeats и может вновь генерировать intervals при n=3. Такие intervals не используются; отдельный postprocessing убирает их из текущего текста, не меняя frozen analysis code. Научная интерпретация и сопоставление screen/confirmation хранятся в этом manual статусе и неизменяемых monitoring snapshots.
