# Optimizer state и training targets: рабочее литературное ревью

Проверено 5 октября 2026. Использованы первичные статьи, официальные proceedings и авторские репозитории. `full_text` в библиографии означает проверку релевантных разделов полного текста, включая ограничения; это не воспроизведение результатов.

## Самый важный вывод для проекта

Не стоит ставить диагноз «испорчены weights / optimizer / targets» по одному успешному reset. В RL параметры, моменты Adam, bootstrap targets и будущие данные образуют связанную динамическую систему. Интервенция может исправлять взаимодействие или менять effective learning rate. Содержательная цель — предсказывать **выгоду вмешательства на конкретном checkpoint**, отдельно от объяснения механизма.

Самый полезный конфликт литературы — между `asadi2023_optimizer_reset` и `ellis2024_adam_rel`: полный reset moments помогает в одной экспериментальной постановке и мешает в другой. Это хорошая причина исследовать выбор repair, но ещё не доказательство, что selector будет лучше достаточно настроенного постоянного метода. Источники: [Asadi et al., §4](https://arxiv.org/html/2306.17833v2), [Ellis et al., §5.2–5.3](https://arxiv.org/html/2412.17113v1).

## Карточки первичных работ

### 1. Resetting the Optimizer in Deep RL — Asadi, Fakoor, Sabach, NeurIPS 2023

Авторы сбрасывают Adam state при синхронизации target network. В Rainbow отдельный reset второго момента полезнее reset только первого; оба вместе дают лучший результат. Измеряют cosine между gradient и первым моментом до/после target update. Это дешёвый диагностический кандидат. Ограничение: расписание известно заранее; причинная специфичность cosine и выбор лучшего repair на unseen checkpoints не проверены. Положительный результат не исключает эффект bias correction и step size. [Полный текст, §4.1–4.3](https://arxiv.org/html/2306.17833v2).

### 2. Adam on Local Time — Ellis et al., NeurIPS 2024

Adam-Rel сбрасывает только timestep `t`, сохраняя `m,v`. Сравнивается с Adam и полным Adam-MR; полный reset может ухудшать DQN/PPO. Объяснение — нестабильный размер update после изменения gradient scale; Adam-Rel также действует как LR schedule. Поэтому в repair bank нужен самостоятельный `t-only` arm и контроль размера update. Основные эксперименты относятся к дискретной смене objectives; перенос расписания на непрерывную нестационарность требует проверки. [§3–5 и §7](https://arxiv.org/html/2412.17113v1); [venue](https://proceedings.neurips.cc/paper_files/paper/2024/hash/f2733d3b0dde1d74995f35a9cf442d38-Abstract-Conference.html).

### 3. Dissecting Deep RL with High Update Ratios — Hussing et al., RLC/RLJ 2024

После усиленного обучения на раннем replay авторы находят Q divergence, связанное с unseen actions и optimizer statistics; SGD+momentum и RMSProp разделяют части Adam. Output Feature Normalization конкурирует с network resets. Важно: effective feature rank иногда восстанавливается с новыми данными, поэтому низкий rank не всегда необратимая поломка weights. OFN меняет геометрию модели; его успех не является чистым тестом targets. [§3–5, Appendix B.2–B.3](https://arxiv.org/html/2403.05996v3).

### 4. Implicit Under-Parameterization — Kumar et al., ICLR 2021

Классическая постановка: iterated bootstrap regression снижает effective rank learned features; вводится spectral penalty. Это обоснование включать rank в diagnostic, но feature rank — не то же самое, что NTK rank, и он не измеряет непосредственно gain после фиксированного бюджета. Теория использует упрощённые kernel/deep-linear модели; результаты не превращают любой низкий rank в доказанный bottleneck. [Статья, §3–5](https://arxiv.org/pdf/2010.14498).

### 5. DR3 — Kumar et al., ICLR 2022

Изучает feature co-adaptation на фиксированных offline data, исключая online exploration как объяснение конкретного эффекта. Существенно различаются observed-action SARSA и backups с out-of-sample actions, даже когда они in-distribution. DR3 штрафует dot product последовательных features. В приложении cosine similarity оказался менее информативным, чем исходный dot product. «Fixed replay» поэтому не равно «fixed supervised task»: bootstrap dynamics остаётся. [§3, Appendix A.6](https://arxiv.org/pdf/2112.04716).

### 6. Stop Regressing — Farebrother et al., ICML 2024

HL-Gauss превращает scalar value targets в сглаженную категориальную цель; сравнения с MSE, two-hot и C51 показывают, что categorical losses неодинаковы. Анализ включает target noise, growing target magnitude, frozen-feature probing и SARSA. Для нашего исследования это источник `targets/loss` interventions и probe tasks. Переключение scalar head на categorical меняет архитектуру и loss одновременно; его нельзя представлять чистой заменой training targets. [§3–5, Appendix A/B.4](https://arxiv.org/html/2403.03950v1); [proceedings](https://proceedings.mlr.press/v235/farebrother24a.html).

### 7. OPEN — Goldie et al., NeurIPS 2024

Наиболее близкая автоматическая политика в этой ветке литературы: meta-learned optimizer использует gradients, moments, dormancy, layer position и training progress. Проверяется перенос на другие environments и размеры networks. OPEN выбирает непрерывные updates и exploration noise, а не repair из конечного набора. Это соседний сильный baseline и prior для признаков; meta-training значительно дороже обучения простого selector. Dormancy авторы прямо не считают достаточной метрикой качества. [§5–7, Appendix F](https://papers.nips.cc/paper/2024/file/09e1944b7f2372f9f81866470c59b663-Paper-Conference.pdf); [код](https://github.com/AlexGoldie/rl-learned-optimization).

### 8. Reducing Churn / C-CHAIN — Tang et al., ICML 2025

Churn — изменение outputs вне minibatch после update. Работа связывает усиление churn с NTK collapse; C-CHAIN одновременно decorrelates gradients и корректирует step size. Полезно как диагностическая family и continual-training baseline. Но churn требует specification reference data, а реальный короткий update уже является probe intervention. Общая causal diagnosis и prospective repair selection не показаны; в рассмотренных MNIST settings метод слабее некоторых baselines. [§4–5](https://arxiv.org/html/2506.00592v1); [proceedings](https://proceedings.mlr.press/v267/tang25g.html); [код](https://github.com/bluecontra/C-CHAIN).

### 9. Dynamical Isometry / AdamO — Rosseau, Müller, Nowe, ICML 2026

Свежая геометрическая формализация связывает plasticity с anisotropy empirical NTK; AdamO отделяет orthogonality regularizer от моментов task gradients. Диагностики включают Jacobian/weight spectra и NTK; эксперименты — continual supervised learning и PPO MinAtar/Octax. Это комбинация optimizer/parameter intervention, преимущественно профилактическая. Изометрия — surrogate под предположениями о tasks/inputs, а не универсальная гарантия adaptation; pre-repair selector не проверяется. [§2, §4.4, §6](https://arxiv.org/html/2606.09762v1); [официальная публикация](https://proceedings.mlr.press/v306/rosseau26a.html).

### 10. DreamerV3 — Hafner et al., Nature 2025

Symlog и symexp two-hot decouple target scale от optimisation scale; Nature-версия использует также LaProp и adaptive clipping. Это сильный пример общей стабильности, но не доказательство восстановления plasticity конкретного checkpoint. Содержательный control — изменить target scale при одинаковой underlying task difficulty. Важно фиксировать версию: проверенный arXiv v2 датирован 2024 и не во всём совпадает с Nature-версией. [Robust predictions, arXiv v2](https://arxiv.org/pdf/2301.04104); [Nature Methods](https://www.nature.com/articles/s41586-025-08744-2); [авторская реализация](https://github.com/danijar/dreamerv3).

## Что отделить в терминологии

Это наши проектные определения, а не установленная авторами taxonomy:

- **Target generator:** Bellman backup, Monte Carlo, SARSA, target-network refresh / lag.
- **Target encoding:** scalar, two-hot, HL-Gauss, return distribution.
- **Loss geometry / scale:** MSE, Huber, cross-entropy, reward/value transforms.
- **Optimizer memory:** `m,v,t`, scheduler state и clipping.

Смешивание этих четырёх вещей в один «targets repair» сделает причинную интерпретацию слабой. Symlog разумно проверять отдельным scale arm; в текущем наборе статей его роль для выбора repair специально не установлена.

## Минимальная польза для experiment design

Рекомендации автора этой заметки, которые ещё предстоит проверить:

1. Для optimizer сравнить continue, full reset, `m-only`, `v-only`, `t-only`; точно записывать bias correction и фактический update norm. Partial resets должны иметь явную specification: что делать с общим timestep и коррекцией каждого момента.
2. Сначала измерять predictive признаки без изменения исходного checkpoint: gradient/moment alignment на одинаковых batches, gradient-scale/second-moment mismatch, предполагаемый Adam update norm, value/target scale, bootstrap disagreement.
3. Rank, NTK spectrum, churn и dormancy рассматривать как competing features, проверяя дополнительную прогнозную пользу сверх age/return/LR.
4. Сохранить отдельную ветку fixed replay + frozen targets. Если freeze помогает, это указывает на чувствительность к изменению targets; не доказывает, что weights и optimizer здоровы.
5. Для interventions, меняющих head или representation, требуются отдельные адаптация/дистилляция и контроль стоимости. Начать с вмешательств, которые сохраняют architecture.

Наиболее близкий автоматический предшественник здесь — **OPEN**; наиболее полезная мотивация выбора, а не постоянного reset — **эмпирический конфликт Adam-MR и Adam-Rel**. Ни одна из этих десяти работ не устанавливает сравнение diagnostic-before-repair → discrete repair → unseen environment в предлагаемом протоколе. Это ограниченное наблюдение по прочитанному набору, не доказательство отсутствия таких работ вообще.
