# Состояние исследования, 9 октября 2026

Срез около 14:41 MSK: **72 из 96 screening-веток завершены**, исходные 12 checkpoints готовы. Coordinator и supervisor работают на двух workers. Прерванные после отсутствия процессов расчёты возобновлены с прежними seeds и matched savepoints; код learner, scientific config и очередь не менялись.

Полностью завершены три Breakout histories и одна Asterix history: каждая содержит два возраста и два начальных repeats. У второй Asterix history пока завершён только младший возраст. Numerical failures и расхождений бюджета нет. Confirm repeats 2/3 и механистические серии ещё впереди.

Средний final-return effect относительно continue в полных histories:

- Breakout: Adam reset **−0.500**, head reset **+0.071**, совместный reset **−0.696**; три histories.
- Asterix: Adam reset **+0.3875**, head reset **−0.0625**, совместный reset **+0.850**; одна history.

Это описательные числа, без population confidence intervals. Возрасты и repeats сначала усредняются внутри history; незавершённые histories исключены из этих средних. По Breakout ранний кандидат на устойчивое head×optimizer взаимодействие не подтвердился: знак interaction меняется между histories. Первое отличие Asterix может отражать особенности игры; диагностическая ценность индивидуальных признаков ещё не проверена. Все направления пока **неопределённы**.

Следующий шаг: завершить оставшиеся 24 screening-ветки, затем fixed-TD/LR controls и независимые continuation repeats. Seaquest/Freeway остаются закрытыми для development и policy evaluation до допустимого freeze.

Старый автоматически созданный `REPORT.md` содержит исторический срез 16 строк. Его интервалы при малом числе histories не используются для текущих научных выводов. Дополнительные аудиты — в `NOTES_manual.md`; этот файл и monitoring сохраняются отдельно от перезаписи генератором.

В первом пакете manifest зафиксирован wall time 25.56 h, включающий остановки. Строгая календарная граница 24 h в текущем runner не подтверждена; дефект учтён в completion audit. Update counters пакетов ниже 12M; metadata сохранены без исправления чисел задним числом.

Raw SHA, данные, per-history effects, состояние процессов и объяснение следующего теста: `monitoring/20261009T1140Z.json` и `monitoring/20261009T1140Z.md`. Воспроизводимый график: `monitoring/history_effects_20261009T1140Z.png`.
