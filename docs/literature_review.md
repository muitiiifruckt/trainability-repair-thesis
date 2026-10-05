# Литературный обзор: от потери plasticity к выбору repair

Срез поиска: **5 октября 2026**. Это тематический обзор под исследовательский вопрос, а не законченный systematic review. Подробные карточки и границы проверки — в [библиографии](../bibliography/README.md) и [заметках](../notes/parameter_plasticity.md). Формулы и предлагаемый эксперимент вынесены отдельно.

## Главный вывод

У проекта есть содержательный вопрос: **предсказывать относительную пользу разных вмешательств по состоянию checkpoint до ремонта, а затем проверять выбор на новых средах**. Но большая часть исходной мотивации уже изучена: fixed-budget plasticity, reset отдельных компонентов, диагностика с помощью клонирования и adaptive interventions имеют прямых предшественников. Новизну надо искать в точной постановке selection и её проверке, а не в самом факте деградации обучаемости.

В данном поиске точный эквивалент всего предлагаемого протокола не обнаружен. Это ограниченное наблюдение, не основание писать «впервые» без дополнительного novelty audit. [Ближайшие работы](novelty.md).

## 1. Какие явления нельзя смешивать

**Plasticity / trainability** — способность обновлять predictions и учиться на новой задаче. **Forgetting** — потеря результата на старой задаче. **Negative transfer** — новая задача решается хуже после предыдущего обучения, чем из подходящего reference initialization. **Плохой return** — системный outcome, который может возникать также из-за exploration, данных и несовершенных targets.

Lyle et al. задают plasticity через конечный loss после фиксированного optimization process; в исследованиях используются probe objectives, а не только reward. Поэтому «проект впервые формализует trainability» — слишком сильная формулировка. [Understanding Plasticity, ICML 2023, §2.2–3.1](https://proceedings.mlr.press/v202/lyle23b/lyle23b.pdf).

Наш практический вывод: хранить отдельно probe-loss, online return и effect относительно continue. Нельзя объявлять plateau доказательством потери plasticity или делать сильный claim по одной proxy-метрике.

Для ориентации полезен обновлённый survey Klein et al.: v3 от апреля 2026 содержит таксономию более 50 mitigations и методологических проблем. В нём изменились авторский состав и структура относительно 2024; в базе фиксируется версия. Empirical claims ниже привязаны к исходным работам. [Survey v3](https://arxiv.org/html/2411.04832v3).

## 2. Параметры и состояние optimizer: разделение уже проверялось

**Primacy Bias** показывает пользу сброса learner при сохранении replay. В приложении уже есть optimizer-only / parameter-only ablation; отрицательный результат optimizer-only в этих задачах не согласуется с универсальной гипотезой «сброс Adam всё починит». [Nikishin et al., ICML 2022](https://proceedings.mlr.press/v162/nikishin22a.html).

**Resetting the Optimizer** получает пользу от сброса Adam state при target updates. **Adam on Local Time** показывает режимы, где полный reset хуже сохранения моментов со сбросом timestep. Различия learning-rate tuning и nonstationarity существенны. [Asadi et al., NeurIPS 2023](https://arxiv.org/html/2306.17833v2), [Ellis et al., NeurIPS 2024](https://arxiv.org/html/2412.17113v1).

Наш вывод: optimizer нельзя трактовать как один исправный/неисправный объект. Минимальный bank сравнивает full reset; расширенный — отдельные \(m\), \(v\), \(t\) interventions, с явным bias-correction rule и контролем update norm. Большой эффект reset может объясняться сменой effective step size.

**Уточнение второго прохода:** headline comparisons Asadi и AdamRel используют разные budgets/configs; в AdamRel PPO отдельно меняются LR, clipping и GAE. Это не один и тот же эксперимент с обратным ответом. Partial-reset clocks Asadi не удалось подтвердить в авторском коде. Для timestep-only reset мы вывели точный LR/epsilon control и проверили его на малой задаче; это собственный контроль, не новое эмпирическое утверждение статьи. [Audit](../notes/code_audit_optimizer.md), [вывод формулы](adam_reset_controls.md).

## 3. Dormancy, rank, curvature: полезные сигналы с ограничениями

**ReDo** адресно обновляет малоактивные units. **Continual Backprop** заменяет units по utility; с Adam изменяются и соответствующие optimizer statistics. Это уже диагностика, управляющая локальным intervention. [Sokar et al., ICML 2023](https://proceedings.mlr.press/v202/sokar23a.html), [Dohare et al., Nature 2024](https://doi.org/10.1038/s41586-024-07711-7).

**Implicit Under-Parameterization** связывает bootstrap learning и сокращение feature rank. **Dissecting High Update Ratios** показывает, что картина включает value divergence и optimizer dynamics; некоторые representation metrics восстанавливаются с новыми данными. [Kumar et al., ICLR 2021](https://arxiv.org/pdf/2010.14498), [Hussing et al., RLC 2024](https://arxiv.org/html/2403.05996v3).

Lyle et al. демонстрируют, что plasticity loss не сводится к saturation; более поздняя **Disentangling** изучает несколько механизмов и комбинации mitigations. [Understanding Plasticity](https://proceedings.mlr.press/v202/lyle23b.html), [Disentangling, CoLLAs 2024 / proceedings 2025](https://proceedings.mlr.press/v274/lyle25a.html).

Наш вывод: диагностические признаки — конкурирующие predictors treatment response. Каждый должен добавлять predictive value сверх возраста агента, reward history и LR. Feature rank, NTK rank, Hessian spectrum и gradient interference — разные объекты. Batch/distribution/слой измерения необходимо фиксировать.

## 4. Данные и targets — части обратной связи

Фиксированный replay не превращает TD-learning в стационарную supervised задачу. В **DR3** feature co-adaptation исследуется на offline data; важны свойства backup. **Adaptive RR** использует critic activity для изменения replay ratio по стадиям обучения. [Kumar et al., ICLR 2022](https://arxiv.org/pdf/2112.04716), [Ma et al., ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/5647763d4245b23e6a1cb0a8947b38c9-Paper-Conference.pdf).

**Stop Regressing** сравнивает scalar regression с разными categorical objectives. Смена loss/head — более сложное вмешательство, чем замена target values. [Farebrother et al., ICML 2024](https://proceedings.mlr.press/v235/farebrother24a.html).

Наш вывод: «данные/occupancy» и «targets» нужно операционализировать. Old/fresh replay mixture, exploration burst, target-network refresh и loss encoding меняют разные вещи. Frozen-input/frozen-target probe измеряет learner response; live RL измеряет итог всей системы. Экспертные данные и будущие Monte Carlo returns допустимы как лабораторный control, но не как доступный repair без учёта их получения.

## 5. Самые близкие предшественники diagnostic selection

**Plasticity Injection** — прямой prior для clone → intervene → compare. Помимо диагностики после intervention, приложение уже содержит heuristic выбора времени injection. Это особенно важное ограничение широкой идеи «pre-repair signal triggers repair». [Nikishin et al., NeurIPS 2023, §5.2 и Appendix B](https://papers.neurips.cc/paper_files/paper/2023/file/75101364dc3aa7772d27528ea504472b-Paper-Conference.pdf).

**On-Policy Study** сравнивает repairs и связывает diagnostics с outcomes. Анализ финальных метрик и prospective выбор из исходного состояния — разные дизайны; этот разрыв надо проверить в коде, а не приписывать себе любые диагностические корреляции. [Juliani & Ash, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ce7984e36d58659211a8dc7d5457cd6f-Abstract-Conference.html).

**No Representation, No Trust** использует fixed-budget capacity probe с fresh optimizer, то есть диагностическая процедура уже устраняет optimizer memory. **Data, Auxiliary Losses, or Normalization Layers** сравнивает 18 PPO interventions и их metrics; это tuned run-level configurations, а не найденный prospective checkpoint selector. Для второй работы проверены индексированные sections PDF, прямое открытие ограничено verification challenge. [Moalla et al., NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/81166fbd9cc5adf14031cdb69d3fd6a8-Abstract-Conference.html), [Pyatko et al., EWRL 2025](https://openreview.net/pdf?id=6CViR7tKj2). [Подробная оговорка и разбор](../notes/additional_onpolicy.md).

**OPEN** уже обучает optimizer по gradient/moment/activity signals с тестами переноса. **Automatic Soft Reset** учит интенсивность parameter drift из данных. Это сильные соседние approaches; конечный discrete repair menu не является общим форматом их решения. [Goldie et al., NeurIPS 2024](https://papers.nips.cc/paper/2024/file/09e1944b7f2372f9f81866470c59b663-Paper-Conference.pdf), [Galashov et al., NeurIPS 2024](https://arxiv.org/html/2411.04034v1).

**CPR** и **NeuMoSync**, свежие препринты июля–августа 2026, усиливают риск широкого novelty claim: utility задаёт частичный reset, learned controller управляет несколькими механизмами plasticity. NeuMoSync также использует snapshots и бюджетную адаптацию, но проверенная область — supervised CL. [CPR](https://arxiv.org/abs/2607.24996), [NeuMoSync](https://arxiv.org/abs/2608.04358).

Наш вывод: возможный вклад — конкретно **pre-intervention prediction относительных effects нескольких совместимых repairs и held-out RL selection regret**. Сопоставление с этими methods должно включать costs и training history.

Повторный поиск добавил **TeLAPA**: discrete origin selection использует короткие реальные adaptation probes нескольких archived candidates, с fresh optimizer. Это соседний prior для выбора по наблюдаемой response, и аргумент за baseline budgeted probing. **PAME** уже выбирает модуль и момент injection по activity/novelty thresholds. **SBP/P3O** совмещает reset с distillation recovery; стоимость этого repair переменная. [TeLAPA, Appendix D](https://arxiv.org/html/2604.15414v2), [PAME](https://ifaamas.csc.liv.ac.uk/Proceedings/aamas2025/pdfs/p2299.pdf), [SBP/P3O](https://proceedings.mlr.press/v267/zhou25am.html).

Уточнённый вопрос — **amortized prediction** эффекта: может ли дешёвая диагностика заменить дорогое пробное дообучение каждого варианта при переносе между средами? «Выбор по trainability» сам по себе уже недостаточно узкий claim. [Результаты поиска](../notes/infrastructure_and_forward_search.md).

## 6. Что нового в математической картине 2025–2026

В **C-CHAIN** рассматривается churn и спектральная деградация NTK. **Spectral Collapse** связывает доступные направления адаптации с бюджетом и residual alignment в упрощённой модели. **AdamO** рассматривает dynamical isometry и optimizer coupling. [Tang et al., ICML 2025](https://proceedings.mlr.press/v267/tang25g.html), [Prakash et al., ICML 2026](https://proceedings.mlr.press/v306/prakash26b.html), [Rosseau et al., ICML 2026](https://proceedings.mlr.press/v306/rosseau26a.html).

**Sample Weight Decay** даёт ещё одну теоретическую ветку: rank и attenuation gradients под input/target nonstationarity. Второй проход проверил Theorem 3 и proof: growing replay и exact previous minimizer существенны; затухание distribution-shift term не означает универсального затухания target-drift term или всех gradients в fixed-capacity replay. В AdamO также нужна оговорка, какая сторона rectangular isometry сохраняет norms при backpropagation. Это наши проверки границ аргументов; empirical результаты ими не опровергнуты. [Wu et al., ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/file/2886558a03d11f95d4020a460fe4a390-Paper-Conference.pdf), [подробный audit](../notes/code_audit_optimizer.md).

Наш вывод: полезная матчасть — finite-budget optimization, conditioning/spectra, interactions и statistical decision-making. Нет необходимости обещать полную теорию live RL. Для первого проекта достаточно operational outcome, предсказания response и корректной оценки transfer; спектральный анализ может объяснять выбранные случаи.

## 7. Инфраструктура и evaluation

**Plasticine v2** предоставляет реализации mitigations и benchmark scenarios; актуальный abstract перечисляет шесть metrics, тогда как старые описания v1 дают другие числа. В статически проверенном commit `aa00b4b` reset/injection заменяют Parameters после создания optimizer без его перепривязки. Полный checkpoint fork не реализован, MinAtar entrypoint нет. Поэтому это каталог для осторожного port, а не готовый runner нашего дизайна. Авторские training scripts не запускались; finding не устанавливает происхождение paper results. [Yuan et al., v2 2026](https://arxiv.org/abs/2504.17490v2), [точные code links и adoption checks](../notes/infrastructure_and_forward_search.md).

Аудит **On-Policy Study** выявил fresh AdamW при каждом task shift даже в `none` condition: baseline сохраняет weights, но сбрасывает optimizer. Реализация injection также отличается от основной frozen-head specification Nikishin. В Primacy resets затрагиваются дополнительные target/temperature/RNG/optimizer states. Сравнение по одному имени repair скрывает разные interventions. [Pinned code audit](../notes/code_audit_parameters.md).

Repair selection близок к per-instance algorithm selection. Стоит сохранить outcomes всех actions, feature costs и menus, как отдельный selection dataset. [Rice, 1976](https://www.sciencedirect.com/science/article/pii/S0065245808605203), [ASlib](https://arxiv.org/abs/1506.02465v3).

Для результата необходимы interval estimates и устойчивые aggregate metrics, а не только один mean по нескольким seeds. [Agarwal et al., NeurIPS 2021](https://arxiv.org/abs/2108.13264).

Наше дополнение к evaluation: grouped environment/trajectory split, baseline single best repair из train/dev, контроль post-treatment leakage, correction optimistic empirical oracle, separate immediate damage/recovery, diagnostic costs. Полный proposed protocol — [здесь](experimental_protocol.md).

## 8. Что реально следует исследовать дальше

1. Завершить аудит ещё не проверенных code paths Adaptive RR, Automatic Soft Reset и NeuMoSync; для Injection авторский код не найден. Уже прочитанные версии/commits указаны в трёх audit notes.
2. Перенести выполненный [синтетический контроль](controlled_probe_results.md) на fixed transitions/frozen TD targets, затем маленький online RL-пилот. Наблюдаемая synthetic heterogeneity ещё не оправдывает сложный RL selector.
3. Сравнить pre-repair features с age/return-only predictor, а не только строить корреляции metrics с reward.
4. Разделить fixed probe и live RL, показать перенос первого на второй либо честно зафиксировать его отсутствие.
5. Проверить held-out environment families до claims о generalization. Unseen seeds и новый checkpoint той же trajectory — другое, более лёгкое требование.

Рабочее название: **Predicting Repair Responses for Trainability Loss in Deep Reinforcement Learning**. Оно обещает измеримое предсказание эффектов; mechanistic diagnosis можно добавить, если отдельные результаты действительно его поддержат.
