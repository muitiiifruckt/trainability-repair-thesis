# Состояние исследования, 9 октября 2026

**Первый online screening завершён: 96/96 веток**, 12 checkpoints, шесть исходных histories — по три в Breakout и Asterix. У каждой history два возраста и два continuation repeats. Все 24 checkpoint×repeat панели полные, budgets по 50k transitions/updates, continuation/evaluation seeds внутри панели совпадают. Numerical failures, нечисловых finals и дублей нет. Код и scientific config соответствуют исходному manifest.

Средний final-return effect относительно matched continue; сначала усредняются возраста и repeats внутри source history:

| Repair | Breakout, 3 histories | Asterix, 3 histories |
|---|---:|---:|
| Adam reset | −0.500 | +0.338 |
| Head reset | +0.071 | −0.004 |
| Head + Adam reset | −0.696 | +0.771 |

Adam и совместный reset имеют положительное среднее во всех трёх Asterix histories и отрицательное во всех трёх Breakout histories. Head reset меняет знак между histories обеих игр. Head×optimizer interaction по Breakout: −0.563/+0.462/−0.700; по Asterix: +0.525/+0.088/+0.700. Знак среднего interaction одной history не означает согласия всех её отдельных панелей.

Это описательные результаты пилота, без population confidence intervals. Независимая единица — source history, а не 24 панели и не 96 веток. Различие игр создаёт кандидата для следующей проверки, но пока не доказывает пользу diagnostics сверх простого контекста игры или age/return. Нет оснований объявлять optimizer state «испорченным» только потому, что его reset помог. Классификация: **неопределённо**; полезность индивидуального selector не подтверждена.

Head и совместный reset сразу уменьшают return: средняя парная потеря −3.058 в Breakout и −0.850 в Asterix. Adam reset сохраняет функцию и immediate return. Поэтому gain от собственного post-repair старта не заменяет primary final return: разрушение функции создаёт искусственно большой recovery gain.

Fixed-TD и synthetic LR серии уже завершены, идут independent continuation repeats 2/3. На fixed-TD после 2000 updates совместный reset хуже continue во всех 24 jobs, несмотря на положительный online effect Asterix. Synthetic scale drop показывает смену знака optimizer effect с LR и чувствительность процентов к усреднению. Это пока не устанавливает online механизм. Пять float32 timestep-controls превышают declared tolerance; они сохранены как numerical discrepancies отдельно от FPE failures и требуют длинного precision control на безопасной границе. Полный журнал — `monitoring/20261009T1410Z.md`, compact data и явные estimators — в соседних JSON. Seaquest/Freeway остаются закрытыми до допустимого freeze и подтверждённого development signal.

Данные, график и журнал полного screening: `monitoring/20261009T1310Z.json`, `monitoring/history_effects_20261009T1310Z.png`, `monitoring/20261009T1310Z.md`. Root outcomes SHA256: `fe3ee030f585111795edc9fbe17c8833ea4068b8326e0cf4a4447d9264ea396b`. Снимок проверен независимо вторым агентом; воспроизводимые scripts не импортируют learner и выполняют ноль optimizer updates.

Автоматический `REPORT.md` обновлён на 96 строк и содержит bootstrap intervals при n=3/game. Эти интервалы не используются для научных выводов: coverage на таком количестве histories не подтверждён. Manual status и immutable monitoring сохраняются отдельно от генератора.

На срезе около 16:13 MSK coordinator 16652 и supervisor 10116 живы, новых инфраструктурных ошибок нет. Первый закрытый пакет имеет wall time 25.56 h, включающий остановки; строгая календарная граница 24 h текущим runner не обеспечена. Активный пакет — около 16.4 h и 4.04M accounted updates, ниже 12M. Metadata не переписывались. Обязательные пробелы реализации перечислены в `docs/research_program_completion_audit.md`; завершение текущей реализованной батареи ещё не означает выполнения всей программы.
