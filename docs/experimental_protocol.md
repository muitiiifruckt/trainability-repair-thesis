# Первый эксперимент: сначала проверить, нужен ли selector

Дата: 2026-10-05. Это проект RL-протокола. Этап 0 частично выполнен на supervised synthetic control; следующий fixed-TD и online RL этапы ещё не запускались. Числа RL-пилота уточняются после короткого замера скорости и вариативности.

## Проверяемые гипотезы

- **H1 — неоднородность:** лучший repair меняется между checkpoints/средами, и разница превышает шум продолжения обучения.
- **H2 — предсказуемость:** признаки до repair предсказывают различия его эффекта лучше checkpoint age и recent return.
- **H3 — перенос:** selector превосходит лучший постоянный action на полностью исключённых из обучения средах.
- **H4 — практичность:** выигрыш остаётся после учёта цены диагностики, немедленного ущерба и recovery horizon.

Если H1 не подтверждается, первоочередной результат — хороший простой repair и границы его применимости. Нельзя заранее считать, что always-optimizer-reset обязательно лучше continue.

## Этап 0. Проверка корректности вмешательств

На небольшой синтетической смене regression targets проверить cloning всей training state и независимость веток. Затем повторить на маленьком fixed-data TD-примере. Нужны как минимум несколько initializations; один deterministic trace годится только для проверки реализации.

Проверки: идентичность continue-веток при одинаковом RNG; независимость storage optimizer tensors/replay; сохранение predictions при optimizer-only reset; неизменность batch order в controlled probe; точный учёт затронутых target-network и scheduler states.

Использовать известные controlled stressors как sanity checks, но не учить selector только различать специально синтезированные «поломки». Такая классификация может переноситься плохо.

**Уже выполнено:** 24 synthetic checkpoint, 288 веток, шесть invariant tests, CPU float64. Timestep-only Adam reset точно совпал с LR/epsilon schedule control; weights×optimizer responses зависят от training history. [Результаты и ограничения](controlled_probe_results.md), [алгебра control](adam_reset_controls.md). Это не доказательство H1–H3 в RL.

## Этап 1. Fixed-input, fixed-target probe

Для каждого checkpoint заранее сохранить reference inputs и несколько frozen teacher target-функций с одинаковым масштабом. Обучение всех веток проходит на той же последовательности minibatches, с фиксированными targets и одним бюджетом K. Primary loss — на обучаемом probe objective; отдельный held-out subset с той же teacher function оценивает generalization. Для iid random-label stress-test оценивать training memorization, а не предсказание независимых unseen labels. Предлагаемый exploratory K: 100, 500 и 2000 updates; основной K выбирать по train/dev-пилоту, не по held-out test.

Два probe режима: старый optimizer state и fresh optimizer; сравнение показывает зависимость измерения trainability от optimizer. Probe выполняется на копии: исходный агент не получает эти gradients. Random targets — operational stress test, не доказательство пригодности к RL.

**Факторное ядро действий:**

1. `continue`: ничего не менять.
2. `optimizer_reset`: \(m=v=0\), optimizer timestep=0; scheduler и learning rate по отдельному зафиксированному правилу.
3. `parameter_repair`: выбранный head repair; остальные параметры и optimizer оставить. Это намеренно диагностический arm со старыми moments при новых весах.
4. `parameter_and_optimizer`: тот же head repair + тот же optimizer reset.

Для первого parameter repair выбрать один вариант на dev: фиксированный shrink-and-perturb либо head reset. Зафиксировать слои, seed и силу вмешательства до test. Не объединять в одну категорию всё от ReDo до полного network reset.

Старые moments после weight change могут быть несовместимы с новой геометрией; проверять численную устойчивость. Это исследуемый эффект взаимодействия, не рекомендация deployment.

При head reset сохранять Parameter identities либо явно переносить optimizer state на новые объекты. Перед каждым branch update проверять coverage optimizer по текущим trainable parameters. Не копировать найденные в benchmark-коде замены модулей без перепривязки optimizer. Fresh-head sampling и reset к собственной initial head — разные specifications; выбрать одну до dev tuning и сохранить её в config.

Опциональное пятое действие — function-preserving plasticity injection с точным описанием trainable/frozen частей и новой optimizer state. Его нельзя выдавать за weight-only repair.

## Этап 2. Разделить данные и targets лабораторными controls

На малом подмножестве checkpoints пересечь input schedule и target schedule. Задать frozen target rule \(g_0(s,a,r,s')\) и moving rule \(g_t(s,a,r,s')\) того же типа, с одинаковыми loss/scale; для первого controlled test moving-rule sequence записать заранее, чтобы она не зависела от repair branch:

- одинаковый input schedule + \(g_0\);
- одинаковый input schedule + \(g_t\);
- изменяющийся input schedule + тот же \(g_0\);
- изменяющийся input schedule + тот же \(g_t\).

Не заменять teacher semantics между arms. Live endogenous TD-target updates проверять вторым вариантом: они включают feedback и могут перестать быть одинаковыми между ветками.

Определить допустимый reference dataset без информации из будущего. Заморозка targets помогает изолировать оптимизационную динамику, но уже не равна полноценному RL. Хранить эту оценку отдельно от online utility.

**Ближайший fixed-TD контроль:** использовать snapshot transitions и cached targets `r+gamma*(1-terminal)*bootstrap`, рассчитанные до repairs. В Double DQN заморозить также argmax action selector; одного fixed target critic недостаточно. Для SAC аналогично фиксируются target actor, entropy term и sampled next actions. Иначе ветки получают разные training tasks, хотя target network формально frozen.

Сначала две небольшие factorial панели: weights×optimizer на одном dataset/teacher и data×target при фиксированных weights/optimizer. Для второй панели обе frozen target functions заранее применяются к обоим datasets; provenance reference teacher указывается явно. Primary comparison — final loss на общем held-out objective или gain от единого **unrepaired** checkpoint. Fit к собственным training targets и reduction от собственного post-repair старта — дополнительные outcomes; ошибки относительно разных teachers не складывать в один ranking. Privileged reference labels для оценки не передавать selector.

Сравнение old replay и fresh replay при прежних весах/optimizer — data intervention. Сравнение policy exploration schedules — occupancy intervention. Новый replay требует сбора transitions; его стоимость включать в бюджет. Вариант с expert trajectories — privileged laboratory control и не участник deployable selector, если таких данных нет при применении.

Target-only вмешательство на старте должно менять только target state/update rule при прежнем loss/head. Categorical loss, добавление LayerNorm и смена архитектуры — отдельные redesign/prevention baselines: checkpoint совместимость и preservation of predictions там требуют самостоятельного решения.

## Этап 3. Небольшой online RL-пилот

Предлагаемый первый домен — DQN на MinAtar: это упрощённый testbed для более доступных экспериментов ([Young & Tian, 2019](https://arxiv.org/abs/1903.03176), [авторский код](https://github.com/kenjyoung/MinAtar)). Выбор — инженерное предложение; наличие сильной деградации в нашем setup надо проверить.

Начать с двух train/development games, двух возрастов checkpoint, трёх training seeds и четырёх factorial actions. Это 48 continuation branches при одном continuation seed; такой запуск выявляет грубые проблемы и не даёт надёжного доказательства H3. На информативных checkpoints добавить независимые continuation seeds.

Измерить runtime на коротком фрагменте и подобрать horizon на dev так, чтобы у repair была возможность восстановиться. Если degradation не возникает, изучить high replay ratio/controlled reward shift на dev. Не подбирать failure-inducing режимы на test games.

Основной pilot checkpoint брать в заранее определённые моменты обучения. Plateau-trigger можно проверить вторично; выбор только «удачных сломанных» checkpoints создаёт selection bias. Включать и здоровые состояния, где разумно continue.

В каждой ветке фиксировать:

- return до вмешательства, сразу после и на сетке шагов;
- final return, AUC адаптации и return loss во время recovery;
- environment steps, gradient updates, wall-clock, диагностические расходы;
- pre-repair features и весь вектор исходов actions.

Для online policy-changing веток одинаковые seeds не означают одинаковые trajectories. Они могут уменьшать variance, но не удерживают данные постоянными. Начальные snapshots среды можно копировать лишь там, где backend действительно это поддерживает; иначе одинаковые reset states и распределение initial states описать явно.

## Этап 4. Selector и честные baselines

Начать с регуляризованного линейного predictor эффектов и небольшого дерева. Сложный neural selector без доказанного signal не нужен.

Baselines:

- continue;
- каждый always-action отдельно;
- **single best repair**, выбранный только по train/dev;
- random action с заранее заданной вероятностью;
- threshold baseline по dormancy или weight-norm;
- selector только по checkpoint age, recent return и plateau slope;
- short-probe chooser: копии всех доступных repairs, малый бюджет на каждую, выбор по short response;
- полный diagnostic selector;
- ограниченный menu oracle.

ReDo и adaptive injection timing подходят как ближайшие diagnostic policies; Adaptive RR имеет другое меню и бюджет updates, поэтому сравнивать в отдельно описанном budget regime. OPEN/automatic soft reset — полезные baselines второго этапа после согласования training history и затрат.

Short-probe chooser — наш menu baseline по мотивам [TeLAPA](https://arxiv.org/html/2604.15414v2), без policy archives. Probe horizon/ranking выбирать на dev. Основной вариант после выбора стартует от исходного repaired checkpoint; не переносить post-probe weights скрыто. Вариант reuse-probe-weights разрешён как отдельный baseline. В practical budget входят **все** probes, environments interactions и repair costs, не только выбранная ветка. Small probe score не равен population-oracle score.

Для ranking probes выделить отдельные RNG streams, episodes/data и evaluation seeds. После выбора и рестарта main continuation и final evaluation используют свежие независимые repeats; не переигрывать тот же случайный префикс, по которому выбрали победителя. Common randomness между actions внутри одного probe допустима. Chooser получает только заранее разрешённые short-budget outcomes; final/oracle outcomes остаются закрытыми. Этот baseline имеет иной информационный бюджет, чем pre-feature predictor, и стоимость получения outcomes учитывается явно.

До выбора действий проверить effect heterogeneity: есть ли воспроизводимый oracle advantage над single best repair, как часто меняется победитель при новых continuation seeds, какая доля случаев имеет почти равные исходы. Один выигравший action при шумных оценках не должен создавать жёсткую метку «истинная причина».

## Разбиение и предотвращение утечки

В основном тесте «unseen environment» означает: selector не обучался на этой среде, но получает checkpoint агента, уже обученного в ней, и разрешённые diagnostics текущего состояния. Это отличается от задачи «агента впервые переносят в неизвестную среду»: для неё отдельно задаются момент смены среды, доступная diagnostic exposure и distribution будущих задач. Не смешивать эти два claims.

Разделять **полные training trajectories и environment families**, включая все их checkpoints, seeds, shifted variants и связанные replay samples. Random split строк checkpoint даст слишком лёгкую задачу.

MinAtar leave-one-game-out — ранний тест переноса между играми ограниченного семейства; для сильного unseen-environment claim нужен второй domain/семейство. Два варианта friction той же Ant — слабее, чем held-out embodiment. Cross-algorithm/architecture перенос — дополнительный результат, не обязательное условие первого прототипа.

Normalizers признаков, feature selection, thresholds, repair strength, horizon и return normalization обучать только на train/dev. Primary selector не получает environment name/id. Secondary env-aware baseline можно показать явно.

Функция признаков в тестовой среде получает только доступное на момент решения. Если для неё нужен rollout в этой среде, это **online diagnosis on a held-out environment**, а не zero-shot без взаимодействия. Final continuation/oracle outcomes всех ремонтов разрешены только evaluator. Исключение для short-probe chooser ограничено его объявленными short-budget probes на независимом stream; основной pre-feature selector их не получает.

## Оценка качества и oracle bias

Главные результаты: regret в единицах return, improvement над continue и single best repair, доля вредных решений, интервалы неопределённости. Return scales между средами различаются: нормировать заранее доступными task anchors либо публиковать per-environment effects. Не оценивать нормировку по test oracle.

Из \(\max_a\widehat Y_a\) возникает оптимистическое смещение в ожидании. На test checkpoints разделить continuation repeats для выбора winner и независимой оценки его качества: это **cross-fitted estimated-winner value**, обычно ниже недоступного истинного population oracle \(\max_a\mathbb E[Y_a]\). Показать также naive plug-in maximum как оптимистичную estimate, а не строгую upper bound. На конкретной выборке ни одна из них не гарантированно ограничивает true oracle. Menu oracle — концептуальный reference, не обученный baseline; оценку regret сопровождать неопределённостью этого reference.

Отдельно bootstrap по environment family/trajectory; checkpoints одной trajectory зависимы. Ресэмплировать и continuation seeds на вложенном уровне. При малом числе сред интервалы плохо характеризуют перенос на новые семьи — ограничение нужно оставить явным. IQM/performance profiles можно добавить по методологии [Agarwal et al., NeurIPS 2021](https://arxiv.org/abs/2108.13264).

## Цена и критерии продолжения

Число веток: \(N=E\times S\times C\times A\times R\), где E — environments, S — training seeds, C — checkpoints, A — actions, R — continuation repeats. Например, 5×5×3×5×3 = 1125 веток; это пример сметы, не утверждённый запуск.

Два результата публиковать отдельно: scientific comparison при одинаковом продолжении после repair и practical comparison при одинаковом total budget. Во втором стоимость диагностики/repair вычитается из доступного continuation budget; baseline, который diagnostics не использует, не платит за неё. Расходы задавать вектором environment transitions/compute; gradient evaluations и updates логировать отдельно. Не просто добавлять overhead к графику одинакового продолжения. Для actions, меняющих replay ratio, невозможно одновременно фиксировать env steps и число updates без изменения задачи; публиковать оба budget regimes.

Selector training и сбор offline branch labels тоже имеют стоимость. Отдельно показать training compute, deployment cost/decision и число решений, после которого amortized prediction выгоднее repeated probing при выбранной метрике ресурсов. Target refresh clock, optimizer clock и environment clock логировать раздельно. Paper-specific schedules нельзя переводить в наши units без проверки.

Продолжать до полноценного selector, если: (1) есть устойчивое разнообразие эффектов; (2) дешёвые features несут signal сверх age/return; (3) выигрыш не исчезает при grouped split и учёте recovery cost. Отрицательный результат записывать, а не скрывать заменой split или menus после просмотра test.
