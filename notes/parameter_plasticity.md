# Пластичность параметров: что уже известно и где остаётся место для selector

Проверено 5 октября 2026. Это рабочие заметки по первичным источникам, а не исчерпывающий обзор. Метаданные и ссылки находятся в `bibliography/parameters.json`. `full_text` означает, что проверены релевантные методы/результаты полного текста; это не означает проверку всех доказательств и воспроизведение экспериментов. Для Nature 2024 часть алгоритмических деталей проверена по предшествующему авторскому препринту, что отдельно отмечено в карточке.

## Самое существенное для проекта

Идея «пластичность = результат обучения за фиксированный бюджет» уже формализована. Диагностика с помощью вмешательства и сравнения копий checkpoint тоже уже существует. Потенциально сильный новый вклад — **по информации до вмешательства выбрать среди разных классов ремонта и показать перенос выбора на целиком отложенные среды**. Следует отдельно проверять, не решает ли задачу один универсальный ремонт: selector имеет смысл лишь при воспроизводимой неоднородности выгод вмешательств.

Ниже каждый абзац «Из источника» — краткий пересказ работы; «Для проекта» — наша интерпретация и критика.

## 1. Primacy bias: reset действительно помогает, но интервенция многосоставна

**Из источника.** [Nikishin et al., ICML 2022](https://proceedings.mlr.press/v162/nikishin22a.html) показывают зависимость RL от раннего опыта и улучшения после периодического сброса части сетей при сохранённом replay. Appendix B / Fig. 9 уже сравнивает optimizer-only и parameter-only resets на четырёх DMC задачах: первый там не помог, второй почти воспроизводил полный reset. Глубина и выбор сбрасываемых модулей влияют на эффект.

**Для проекта.** Это прямой предшественник факторного эксперимента. Выигрыш reset не идентифицирует единственную «поломанную» компоненту: меняются predictions, exploration и дальнейшие данные. Нужно воспроизвести этот baseline, а не объявлять разделение weights/optimizer новым. Отрицательный optimizer-only результат относится к конкретному режиму.

## 2. ReDo: дешёвая диагностика dormancy и адресный ремонт

**Из источника.** [Sokar et al., ICML 2023](https://proceedings.mlr.press/v202/sokar23a.html) измеряют нормированную среднюю абсолютную активацию нейрона на заданном распределении входов. ReDo переинициализирует входящие веса малоактивных нейронов и зануляет исходящие. При нулевой активности функция сохраняется, при положительном пороге — меняется. Работы с fixed data / меняющимися targets связывают dormancy с target nonstationarity; высокая replay ratio её усиливает.

**Для проекта.** Доля dormant neurons — естественный признак selector, но зависит от выбранного probe dataset. Низкая активность на данных текущей политики не доказывает глобальную бесполезность нейрона. Нужно проверить, предсказывает ли она именно **относительную** пользу ReDo против optimizer reset / no-op, а не просто возраст checkpoint.

## 3. Understanding Plasticity: бюджетная формализация и curvature

**Из источника.** [Lyle et al., ICML 2023](https://proceedings.mlr.press/v202/lyle23b.html), Eq. 5, определяют plasticity через ожидаемый финальный loss на probe objectives после заданного optimization algorithm, допускающего даже несколько шагов. Измеряют curvature и gradient interference; plasticity loss возможна без saturated units. Сравнивают resets, LayerNorm, weight decay, Shrink-and-Perturb и output parameterization. Appendix показывает режимы, где optimizer reset и Shrink-and-Perturb не помогают.

**Для проекта.** Это основной источник определения trainability и обязательный adversarial baseline для простой метрики dead units. Probe-loss и return нужно хранить как разные outcomes. LayerNorm улучшал ALE, но авторы прямо ограничивают причинную интерпретацию: это улучшение нельзя однозначно приписать восстановлению plasticity.

## 4. Plasticity Injection: уже предложена диагностика через counterfactual repair

**Из источника.** [Nikishin et al., NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/75101364dc3aa7772d27528ea504472b-Abstract-Conference.html) сохраняют начальные predictions с помощью суммы старой frozen head и разности двух одинаково инициализированных новых heads, одна из которых обучается. Число trainable parameters сохраняется, общее число растёт. Раздел 5.2 прямо предлагает клонировать plateaued agent, применить injection и сравнить дальнейшее обучение. На Atari выгода неодинакова по средам.

**Для проекта.** Самый близкий предшественник intervention-based diagnosis. Сохранение initial policy уменьшает confound от немедленного exploration jump; будущая occupancy всё равно меняется. Их диагноз получается **после** обучения с intervention. Наша цель отличается prospective prediction выбора ремонта до расходования его основного бюджета.

## 5. Implicit under-parameterization: rank полезен, но не универсален

**Из источника.** [Kumar et al., ICLR 2021](https://research.google/pubs/implicit-under-parameterization-inhibits-data-efficient-deep-reinforcement-learning/) связывают bootstrapped value learning с потерей feature effective rank, анализируют kernel/deep-linear абстракции и используют spectral feature penalty. Исследуются offline и online режимы, в том числе большой data reuse. Теория опирается на упрощающие предположения об оптимизации и регуляризации.

**Для проекта.** Rank — кандидат diagnostic, а не определение способности обучаться. Важно различать entropy effective rank, threshold rank и stable rank: численные значения несопоставимы без формулы. Feature rank, NTK rank и Hessian rank тоже отвечают на разные вопросы. Проверять надо prospective association с treatment benefit, а не только с return.

## 6. Continual deep RL: sparse activation footprint и CReLU

**Из источника.** [Abbas et al., CoLLAs 2023](https://proceedings.mlr.press/v232/abbas23a.html) изучают value-based agents на циклической последовательности Atari игр и вариантах nonstationarity. Измеряются weights, gradients и activations; разрежение activation footprint сопутствует ухудшению обучения. CReLU, использующая положительную и отрицательную ветви входа, оказывается эффективным mitigation в исследованных continual RL режимах.

**Для проекта.** Это ещё один механизм/ремонт для библиотеки кандидатов, но архитектурное изменение посреди trajectory сложнее чистого reset. Нельзя переносить вывод «активации решают проблему» на PPO или все среды. CReLU хорошо использовать как training-from-start baseline, отдельно от совместимых checkpoint interventions.

## 7. Shrink-and-Perturb: источник метода — generalization при warm start

**Из источника.** [Ash & Adams, NeurIPS 2020](https://papers.nips.cc/paper/2020/hash/288cd2567953f06e460a33951f55daaf-Abstract.html) исследуют растущий supervised dataset: warm start может хуже generalize, даже при сходном final training loss. Shrink-perturb initialization масштабирует старые параметры и добавляет случайную компоненту. [Авторский код](https://github.com/JordanAsh/warm_start) подтверждает реализацию и интерполяцию между warm start и новым initialization.

**Для проекта.** Отсюда нельзя напрямую заключать потерю способности оптимизировать RL targets: исходная проблема относится к generalization. Для ремонта записать формулу, масштаб perturbation и распределение шума. Периодический S+P и малые изменения после каждого update — разные treatments, не одна строка в результатах.

## 8. Continual Backprop: utility уже выбирает локальное вмешательство

**Из источника.** [Dohare et al., Nature 2024](https://doi.org/10.1038/s41586-024-07711-7) демонстрируют длительную потерю plasticity и замену небольшой доли менее полезных units через continual backpropagation. Utility учитывает вклад features и способность адаптироваться; используются maturity protection и малые replacement rates. Сопутствующие показатели — inactive units, norm growth и effective rank. [Авторский препринт](https://arxiv.org/abs/2306.13812) описывает сброс соответствующих Adam statistics при замене feature.

**Для проекта.** Уже существует diagnostics-driven selection **какие units** заменить. Это не равно выбору класса repairs. CBP с Adam — сочетание weight и optimizer interventions, поэтому его нельзя подписывать «weights only». Предотвращение деградации на длинном потоке не гарантирует эффективность одного ремонта старого checkpoint за малый бюджет.

## 9. Disentangling: несколько механизмов и комбинации уже исследованы

**Из источника.** [Lyle et al., CoLLAs 2024; proceedings 2025](https://proceedings.mlr.press/v274/lyle25a.html) разделяют target scale, preactivation shifts, unit death / linearization и parameter norm growth. Исследуют отдельные и комбинированные mitigations; LayerNorm с weight decay оказывается сильной комбинацией. Общим проявлением могут быть degeneracies empirical NTK. Отдельный механизм не считается универсальным объяснением.

**Для проекта.** Центральная работа для novelty audit. Наша таксономия weights/optimizer/data/targets — **места вмешательств**, а не четыре независимые физические причины. Например, target scale воздействует через веса и conditioning, optimizer после shift способен убить units. Комбинации repairs должны присутствовать хотя бы в небольшой факторной подвыборке.

## 10. On-policy study: диагностические корреляции не переносятся автоматически

**Из источника.** [Juliani & Ash, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ce7984e36d58659211a8dc7d5457cd6f-Abstract-Conference.html) сравнивают repairs в PPO under domain shifts. Методы, успешные off-policy, иногда хуже no intervention. Soft S+P с LayerNorm — сильный кандидат. Section 4.2 / Appendix D исследуют diagnostic correlations и GLM rewards: используются финальные метрики после десяти rounds; GLM содержит 27 агрегированных observations.

**Для проекта.** Корреляции финального состояния с финальным return не являются выбором ремонта по исходному checkpoint. Наш evaluation должен запретить post-treatment features и разносить среды, а не случайные строки одного trajectory. Различие on/off-policy — аргумент против универсальных thresholds и переноса лучшего метода без проверки.

## 11. Spectral collapse: актуальная бюджетная теория, не полный causal oracle

**Из источника.** [Prakash et al., ICML 2026](https://proceedings.mlr.press/v306/prakash26b.html) связывают failure с исчезновением существенных curvature directions. В linearized ReLU / fixed-gate модели residual сокращается по eigendirections Gram matrix, и достаточно быстрые направления определяются learning rate и числом updates. Важен также residual alignment с этими направлениями. Предлагается сочетание L2 и effective-rank regularization.

**Для проекта.** Особенно близко к finite-budget trainability: rank без alignment и масштаба spectrum может быть слабым predictor. Теорема для linearized модели не доказывает универсальную причину отказа live RL. Полный спектральный diagnostic дорог; сравнить его с дешёвыми layerwise proxies и учитывать стоимость измерения.

## 12. Weight vs unit reset: architecture меняет подходящий ремонт

**Из источника.** [Hernandez-Garcia et al., CoLLAs 2025; proceedings 2026](https://proceedings.mlr.press/v330/hernandez-garcia26a.html) сравнивают selective weight reinitialization, CBP и ReDo. В continual supervised экспериментах мелкий weight reset предпочтительнее unit reset для малых сетей и при LayerNorm. ViT эксперимент показывает дополнительные ограничения normalization и неполное восстановление plasticity.

**Для проекта.** Архитектура и norm statistics могут быть treatment-effect modifiers. Это не проверенный RL selector: перенос в deep RL — отдельный эксперимент. В библиографии сохранены first-publication и proceedings year, поскольку их нельзя смешивать.

## Наши выводы для постановки эксперимента

1. **Разделить phenotype и mechanism.** Плохой finite-budget probe loss — operational trainability phenotype. Выгода определённого repair — treatment response. Латентная «причина» гораздо сильнее обоих утверждений и требует дополнительных assumptions.
2. **Начать с совместимых interventions.** No-op, optimizer-only reset, head reset, ReDo, небольшой S+P, function-preserving injection. LayerNorm/CReLU/continuous CBP изначально сравнивать как prevention baselines, пока не определена корректная checkpoint conversion.
3. **Проверить function change.** Логировать immediate change in return, policy KL и prediction distance. Сохранить online/target networks, replay, RNG, optimizer/scheduler state и normalization state. Иначе «одна копия checkpoint» не означает один полный state.
4. **Две группы diagnostics.** Дешёвые: per-layer norm, normalized dormancy, activation diversity, weight-update ratio. Более дорогие: gradient interference, empirical NTK/GGN spectrum и короткий disposable learning probe. Измерять всё на одинаковом reference dataset и отдельно на occupancy dataset текущего агента.
5. **Предсказывать response, не ярлык причины.** Учить ожидаемую выгоду каждого repair или regret-aware ranking. Лучший repair может отличаться на малом и большом budget; небольшие ties не должны становиться жёсткими class labels.
6. **Нужен сильный baseline.** Всегда применять лучший repair, выбранный на training environments; простые thresholds; age+return-only selector; random selector. Сложная модель оправдана лишь при выигрышe над ними на новых средах.
7. **Проверить неоднородность раньше selector.** Если repair wins объясняются noise или один treatment доминирует, начинать обучение классификатора рано. Повторные continuation seeds и paired comparisons нужны уже на небольшой pilot-выборке.

Непроверенный пока research gap: в рассмотренных работах не обнаружен полноценный before-repair selector среди параметрных, optimizer, data и target interventions с validation на целиком unseen environments. Это ограниченный результат текущего поиска, не доказательство отсутствия такой работы.
