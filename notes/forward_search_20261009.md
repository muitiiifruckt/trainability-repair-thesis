# Forward citations 2025–2026: проход от 2026-10-09

Закрывает пункт «Forward citations» из `docs/novelty.md` («Незакрытый novelty audit») **частично**. Проход выполнен, прямого конкурента не найдено. Но API цитирований недоступны или неполны (§1), поэтому «не найдено» означает результат этого набора запросов и прочитанных текстов, а не отсутствие работы.

**Как читать.** `full_text` — читался текст arXiv-HTML указанной версии (разделы перечислены), не пересказ и не аннотация. `abstract` — только аннотация. «Наш вывод» — интерпретация, не утверждение авторов. Цитаты короткие, из прочитанного текста, с номером раздела.

**Шкала угрозы** (относительно четырёх различий из `docs/novelty.md`: prospective, относительная польза, несколько типов actions, transfer decision):
**нет** — не пересекается; **частичная** — совпадает с одним-двумя различиями или сужает допустимую формулировку, но не делает того же; **прямая** — prospective-предсказание относительной пользы нескольких типов repair с оценкой на отложенных средах. Прямых не найдено.

## 0. Итог

| Работа | Проверка | Угроза | Основание в одну строку |
|---|---|---|---|
| FAME, 2603.00903 (ICLR 2026) | full_text + статический код | частичная | выбор reset / finetune / meta-init **пробной оценкой каждого кандидата** на новой задаче; не prospective, не обучен |
| Three Regimes, 2510.01460 | full_text | частичная | ручное правило по двум измеренным до дообучения скалярам выбирает семейство вмешательств, 45/63 совпадений; offline-to-online RL, не regret |
| Optimization readiness, 2605.09044 | full_text | частичная (только как признак) | prospective ранжирование checkpoints по trainability; supervised, вмешательств нет |
| Ren et al., 2609.33620 | full_text (§5–§6) | частичная | диагностика коррелирует с выигрышем от **одного** reset (ρ = 0.83); LLM |
| Hofmann & Mäder, 2610.09621 | full_text | частичная (методология) | short probe + калиброванный baseline «текущее состояние»; вмешательств нет |
| Local redundancy, 2607.13432 | full_text (§3.6, §4–§5) | частичная (только как признак) | дешёвый предиктор trainability; supervised; корреляция слабая |
| Bu et al., 2609.32819 | full_text (§1, §3, §5–§6) | частичная | prospective предсказание дивергенции в RL; не repair, held-out = seed |
| Ma et al., 2606.25527 | full_text (§13–18) | частичная (рамка) | позиция «diagnosis-driven»; метода-селектора нет |
| Barriers, 2510.00304 | full_text | нет | теория ловушек; критерия выбора вмешательства нет |
| FIRE, 2602.08040 | full_text | нет | один новый repair; выбора между вмешательствами нет |
| CPR, 2607.24996 | сверено с карточкой | нет | один метод, `ρ` настраивается |
| SWD, 2604.01913 / Spectral Collapse, 2509.22335 | full_text (выборочно) | нет | методы и теория, не селекторы |
| He 2026, 2603.21173; Lanzillotta et al., 2607.05609 | full_text (выборочно) | нет | скептический prior и теория горизонта |

Главные выводы:

1. Идею «выбирать вмешательство по измерению до его применения» нельзя подавать как новую. FAME делает это пробной оценкой кандидатов, Three Regimes — правилом по двум скалярам. Обе работы вне нашей постановки (смена задачи и offline-to-online соответственно), но сужают допустимую формулировку.
2. Prospective предсказание trainability по состоянию сети в supervised CL уже есть (optimization readiness, local redundancy, Hofmann & Mäder). Для нашего вопроса они дают **кандидатов в признаки и методологию**, а не конкурента.
3. Survey v3 (Klein et al., 2026-04-18) называет наш пробел открытым: «We cannot yet predict ... which interventions will prove effective for a given task» (§6.1); «how to select interventions based on problem characteristics rather than exhaustive search» (§7). Это поддерживает мотивацию, но не доказывает отсутствия решения.
4. Простое правило по одному-двум скалярам — серьёзный конкурент learned selector (Three Regimes: 71% верных классов у правила по возвратам против 51% и 41% у альтернатив по Q-функции и BC).

## 1. Как искали и что не получилось

**Недоступно или неполно.**
- Semantic Scholar (Graph API и сайт): HTTP 403 с этого хоста.
- OpenAlex: работает, но `cited_by_count` у целевых работ 0–6 (Injection 1, Lyle 2023 6, On-Policy Study 1–2, OPEN 0–1, Asadi 2023 2–3). Граф цитирований для 2025–2026 явно неполон, выводы на нём не строились.
- Google Scholar не запрашивался. Список `awesome-plasticity-loss` последний раз обновлялся 2024-11-12.
- arXiv-API ищет только по метаданным (title/abstract/comments), не по полному тексту. Работы, которые цитируют ключевые статьи, но не упоминают plasticity/trainability в аннотации, могли быть пропущены.

**Что выполнено.**
- **arXiv API** (`export.arxiv.org`), `submittedDate:[2025-01-01 TO 2026-10-09]`, два прохода, 17 шаблонов: plasticity ∧ RL; «loss of plasticity»; trainability/plasticity ∧ reset; plasticity/trainability ∧ select/predict/portfolio/bandit; plasticity/dormant ∧ intervention/remedy/repair; nonstationarity ∧ Adam/optimizer ∧ RL; «plasticity injection»; «dormant neuron»; negative transfer/warm-start ∧ RL ∧ reset; «when to reset»/«adaptive reset»; learned optimizer ∧ RL; trainability ∧ checkpoint ∧ predict; optimizer reset/Adam state; «primacy bias»; plasticity ∧ benchmark/empirical study; plasticity ∧ «no single»/heterogeneous/combination; plasticity ∧ meta-learning/AutoML/hyperparameter. Всего 875 + 949 записей (1520 уникальных). Широкие шаблоны (select/predict и «which remedy», 1431 и 1649 совпадений) обрезаны на 600 последних, их recall хуже. Дальше: фильтр по ключевым словам, ручной просмотр нескольких сотен заголовков, чтение нескольких десятков аннотаций и 17 полных текстов (работы из §2–§3, Survey v3 и выборочно Chua et al.).
- **Запросы по авторам** за тот же период: Lyle, Juliani, Ash, Goldie, Nikishin, Sokar, Dohare, He, Luo, Hutter, Sutton, Mahmood, Castro, Elsayed, Calandra, Pascanu, Dabney, Courville, Foerster; и по авторам ближайших работ (Lillo/Cheney, Rohani, Zang, Galashov, Klein, Yuan, Sun/Kong, Tang, Ma).
- **WebSearch** (US-only, 9 запросов, mode standard):
  1. `"plasticity injection" Nikishin 2023 follow-up 2026 choose between reset, injection, shrink and perturb based on diagnostic reinforcement learning`
  2. `"Understanding plasticity in neural networks" Lyle prospective prediction of trainability from network diagnostics which intervention restores plasticity 2026`
  3. `"Disentangling the Causes of Plasticity Loss in Neural Networks" 2026 automatic selection of interventions layer normalization weight decay combination per environment`
  4. `"A Study of Plasticity Loss in On-Policy Deep Reinforcement Learning" Juliani Ash follow-up 2026 predict plasticity interventions PPO checkpoint`
  5. `OPEN "learned optimization" reinforcement learning plasticity Goldie follow-up 2025 2026 learned optimizer non-stationarity dormancy input`
  6. `predict which plasticity intervention will help a deep RL agent before applying it, diagnostics, held-out environments, arXiv 2026`
  7. `continual reinforcement learning decide whether to reset or finetune new task negative transfer adaptive warm-start selection 2025 2026 "Reset & Distill" follow-up`
  8. `"resetting the optimizer" Adam state reset moments plasticity loss deep reinforcement learning 2026 study when optimizer reset helps`
  9. `meta-controller bandit selects among plasticity remedies reset, shrink-and-perturb, layer norm, based on network diagnostics, deep reinforcement learning 2026 arXiv`

  Новых работ сверх arXiv-скрининга не принесли (запрос 7 вернул FAME, который уже был в списке). Ни один результат не описывает селектор вмешательств.
- Дополнительно: Survey v3 (Klein et al.) §5.12 и §6–§7 прочитан целиком — он сам цитирует все ключевые работы и датирован 2026-04-18.

**Что искали по пяти ключевым статьям.**

| Ключевая статья | Результат |
|---|---|
| Plasticity Injection | Используется как baseline: FIRE §4.3 («shows poor performance across discrete and control tasks»), SWD §6.5, Chua et al. 2605.26357 (под именем plasticity injection там реализован reset последнего слоя P-last, который при плавном дрейфе проигрывает EWC и synaptic consolidation, Fig. 3, 5, §5 — прочитаны подписи и абзацы результатов, не вся статья); работ, выбирающих между Injection и другими repair по диагностике, не найдено |
| Lyle 2023 | Цитируется OR, Ren et al., Local redundancy, Spectral Collapse; продолжения про prospective trainability — только supervised (см. §3) |
| Disentangling (Lyle 2024/25) | Продолжения: комбинации вмешательств в survey §5.12; автоматического подбора комбинаций по состоянию сети не найдено |
| Juliani & Ash | Продолжений с предсказанием вмешательства не найдено; цитируется He 2026 и survey §6.1 как пример неоднородности |
| OPEN | Единственный найденный продолжатель — 2609.35897 (аудит самооткрытого правила Disco103, «ported to a second rule (OPEN)»): про перенос рекуррентного состояния learned optimizer, не про выбор repair; прочие работы Goldie 2025–26 не про plasticity |

## 2. Работы из списка координатора

### 2.1 FAME — Principled Fast and Meta Knowledge Learners for Continual RL

- **Метаданные.** Ke Sun, Hongming Zhang, Jun Jin, Chao Gao, Xi Chen, Wulong Liu, Linglong Kong. arXiv 2603.00903v1, 2026-03-01; comment «Published in ICLR 2026». Код: github.com/datake/FAME, HEAD `d16f006102d2783b483aac6a350835eb4bd051b9` (2026-03-01).
- **Проверка.** `full_text`: §1–§5, App. A, D, E, F.1–F.2, G.1 (arXiv-HTML). Доказательства Prop. 1–3 (App. C) не читались. Код `MinAtar/FAME.py` (строки 240–420) прочитан статически, не запускался; код Atari и Meta-World не читался. HTML помечен v1 (2026-03-01); App. I сообщает, что в camera-ready добавлены ProgressiveNet, CompoNet и последовательность CW10.
- **Что сделано.** Два обучаемых агента: fast learner учится на текущей задаче, meta learner накапливает знание, минимизируя catastrophic forgetting (Prop. 1–2). На старте каждой новой задачи — «adaptive meta warm-up».

**Главный вопрос: как делается выбор.** Ответ: **пробной оценкой каждого кандидата на новой задаче и сравнением результатов. Это не измерение диагностик состояния сети и не обученный предиктор.**

- Кандидаты (§3.1.2): «chooses the most effective warm-up strategy among the preceding meta learner, a random learner (i.e., reset), and the preceding fast learner (i.e., finetune)». Meta-инициализация в value-based случае реализуется BC-регуляризацией на первых `L` шагах, в policy-based — прямой инициализацией (§3.2.2).
- Правило (§3.1.2, Eq. 6): «framed as a one-vs-all hypothesis test based on policy evaluation during the early interaction with a new environment», `H0: V^M ≤ max{V^f, V^r}`. Далее: «picking the best warm-up strategy according to the empirical ranking often performs favorably».
- Бюджет измерения: MinAtar `n = 600` шагов policy evaluation (App. F.1; Table 7 — абляция `n ∈ {300, 600, 1200, 5000}`), Atari `n = 1200`, Meta-World «policy evaluation for 10 episodes among a random policy, the preceding fast policy, and the meta policy» (§4.2). Данные оценки пишутся в replay fast learner, но (App. E): «the policy evaluation occurs at the cost of reducing the policy optimization update steps in total» — то есть измерение вычитается из общего бюджета. Оценка по нашему счёту: ≈ 2·600 из 500k шагов задачи ≈ 0.24% (в коде оцениваются fast и meta).
- Код (MinAtar/FAME.py, наше чтение): по умолчанию `--use_ttest 0`, то есть сравнение `mean_return_meta > mean_return_fast` (эмпирический ранг); каждый кандидат прогоняется жадной политикой `--detection_step` шагов. Для random-кандидата в коде **не rollout**, а Q-значение жадного действия случайной сети в начальном состоянии (`Avereward_rand`), что расходится с Eq. 6 статьи (`V^r = E_{π0}[R]`). Это наблюдение по одному файлу одного коммита; расходится ли это с кодом Atari/Meta-World и с опубликованными числами, не проверялось.
- Что значит «reset»: «reinitialize the parameters of all involved learners» (App. F.1) — полная реинициализация сети; в коде вместе с ней создаётся новый Adam.
- Эксперименты: MinAtar (DQN; breakout, spaceinvaders, freeway; 10 последовательностей по 7 задач, 3 seeds, 500k шагов на задачу), ALE (PPO; 10 режимов SpaceInvaders и 7 режимов Freeway; 3 seeds), Meta-World (SAC; 3 случайные последовательности по 10 задач, 10 seeds). Метрики: average performance, forward transfer (площадь под кривой относительно Reset), forgetting. Доля выбора на MinAtar (§4.1, Fig. 2 справа): «the meta warm-up is chosen with a 95.1% probability», если релевантные данные уже в meta buffer; для новой задачи «the random initialization is more commonly selected».
- Чего нет в проверенных разделах: regret или oracle; абляции «FAME с фиксированным выбором warm-up» (абляции только по `λ`, `L`, `N`, `n`, App. F.2); обученного селектора; признаков состояния сети (поиск по `diagnos|dormant|rank|Adam|optimizer` в тексте даёт только ссылки).

| Признак | FAME |
|---|---|
| Inputs до вмешательства | Нет диагностик; **rollout каждого кандидата в новой среде** (online probe) |
| Несколько типов repair | 3 инициализации: meta / finetune / полный reset. Head-only, optimizer-only, комбинации — нет |
| RL | Да; continual RL, известные границы задач |
| Оценка на отложенных средах | Правило не обучается; те же последовательности задач; regret нет |

**Угроза: частичная.** Что нельзя утверждать как новое: «адаптивный выбор между reset и продолжением по измерению до дообучения», «измерение выбора включено в бюджет». Различия: у нас prospective признаки состояния без rollout кандидатов; решение принимается mid-training внутри одной среды, а не при смене задачи; меню включает действия, не меняющие функцию; оценка — regret на отложенных средах.

**Наш вывод для протокола.** Evaluation-only probe в духе FAME не может отличить optimizer-only reset (и Plasticity Injection) от `continue`: функция в момент 0 совпадает, значит измеренный возврат тоже. Такой probe разделяет только действия, меняющие начальную функцию (head reset, shrink-and-perturb, полный reset). Поэтому наш short-probe chooser должен включать короткое дообучение (как TeLAPA и probe у Hofmann & Mäder); evaluation-only вариант FAME стоит держать как дешёвый отдельный baseline и ожидать, что он не различит state-only repairs.

### 2.2 «Predicting Plasticity in Deep Continual Learning» (optimization readiness)

- **Метаданные.** Jiuqi Wang, Jayanth Srinivasa, Claire Chen, Shuze Daniel Liu, Ali Payani, Shangtong Zhang. arXiv 2605.09044v1, 2026-05-09, 21 стр.
- **Проверка.** `full_text`: §1, §2, §4, §5, §6, §7, App. C.1 (arXiv-HTML). Доказательства (App. B) не читались.
- **Что сделано.** Trainability определена как `k`-step gain: `G^(k)(θ;T) = (L(θ) − E[L(θ_k)]) / L(θ)` при `k` шагах SGD на целевой задаче (Def. 2.1). Теоретически (§3) показано, что label-agnostic метрики (effective rank, 99%-energy rank представлений, eNTK rank) могут давать «хорошие» значения при нулевом прогрессе градиентного спуска. Предложена метрика optimization readiness.

**Формула.** С `g` — популяционный градиент потери `L` на целевой задаче, `ĝ_B` — градиент минибатча, `V_B = E‖ĝ_B − g‖²` (Def. 4.3–4.5):

- gradient strength `S(θ;T) = ‖g‖² / L`;
- gradient reliability `R(θ;T) = ‖g‖² / (‖g‖² + V_B) = ‖g‖² / E_B‖ĝ_B‖²` (при несмещённости, Assumption 4.1);
- `OR(θ;T) = S · R = ‖g‖⁴ / (L · E_B‖ĝ_B‖²)` (0, если `L = 0` или `E‖ĝ_B‖² = 0`).
- Thm 4.6: при β-гладкости и `0 < η < R/β`: `G^(1)(θ;T) ≥ α(1−α)/β · OR(θ;T)`, `α = ηβ/R`.

**Оценка и стоимость (App. C.1).** `g` — градиент по всему validation set (`N_val = 10 000`), `E‖ĝ_B‖²` — по `R = 128` независимым минибатчам `m = 4`. В ablation (§5, subsampling): 10 / 100 / 1000 примеров и всего 16 минибатчей `m = 1`; OR остаётся лучшим на SCR при 0.1% данных, на P-MNIST точность падает, особенно при больших `k`. Стоимость самой OR отдельно не приводится; в App. C.2 указаны затраты на весь эксперимент (<3 ч на seed на CPU для SCR, ≈10.5 ч на seed на GPU для P-MNIST), куда входит и вычисление ground-truth `k`-step gain по 128 rollouts.

**Где проверено.** Slowly-Changing Regression (MLP 2×5) и Permuted MNIST (MLP 3×100), Adam, batch 1; 20 train-последовательностей, 30 (SCR) / 10 (P-MNIST) отложенных validation-задач, чекпойнты каждые 5 задач (201 и 161 на run). Pairwise ranking accuracy (Table 2, `k = 1 / 10 / 100`): SCR — OR 0.987 / 0.988 / 0.977 против лучшего baseline (eNTK 99%-energy rank) 0.765 / 0.768 / 0.779; P-MNIST — OR 0.906 / 0.905 / 0.899 против representation effective rank 0.858 / 0.864 / 0.873. Авторы (§7): «testing whether the same observations hold in continual reinforcement learning ... is an important next step» и «diagnostics of plasticity should be evaluated not only as retrospective explanations ... but also as prospective predictors of future optimization gain».

| Признак | OR |
|---|---|
| Inputs до вмешательства | Да, но нужны данные и потеря **целевой** задачи |
| Несколько типов repair | Нет вмешательств вообще; ранжируются checkpoints, не actions |
| RL | Нет |
| Оценка на отложенных средах | Отложенные задачи есть; метрика — ranking accuracy |

**Угроза: частичная, только на уровне признака.** Нельзя заявлять «prospective предсказание trainability» как новое. Предсказание относительной пользы repair в статье не рассматривается.

**Можно ли посчитать для нашего DQN-checkpoint (наш анализ, не из статьи).** Да, по `TrainingState.snapshot()` (веса, target-сеть, replay, optimizer) без новых данных из среды:
1. Взять `N` (например 1024–10 000) переходов из replay, посчитать TD-цели замороженной target-сетью (`td_targets`), `L̂` = Huber-потеря, `ĝ` = градиент по всей выборке. В `TrainingState.diagnostics()` это уже есть: `pre_td_huber`, `gradient_norm` (1 backward, 1024 примера).
2. Добавить `R` (128 как в статье) минибатчей размера 32 (= `DQNConfig.batch_size`) и среднее `‖ĝ_B‖²`; стоимость — ещё `R` backward на мини-батче, то есть порядка 129 backward на крошечной CNN; `diagnostic_environment_steps = 0`.
3. `OR = ‖ĝ‖⁴ / (L̂ · mean‖ĝ_B‖²)`.

Оговорки: (i) несмещённость (Assumption 4.1) выполняется при i.i.d. минибатчах из фиксированного снимка replay при замороженной target-сети, то есть OR осмысленна только внутри текущего target-периода; (ii) теорема — про один шаг SGD, наш learner использует Adam; OR от состояния Adam не зависит, поэтому для optimizer reset нужны признаки состояния оптимизатора (`gradient_m_cosine`, `m_norm`, `sqrt_v_norm`, возраст — уже в `diagnostics()`); (iii) OR описывает текущую обучаемость, а не состояние после head reset — связь с выигрышем каждого repair проверяется только эмпирически; (iv) вне статьи нигде не проверено на RL.

### 2.3 «Barriers for Learning in an Evolving World»

- **Метаданные.** Amir Joudaki, Giulia Lanzillotta, Mohammad Samragh Razlighi, Iman Mirzadeh, Keivan Alizadeh, Thomas Hofmann, Mehrdad Farajtabar, Fartash Faghri. arXiv 2510.00304, v1 2025-09-30, **читалась v3 (2026-05-17)**; код github.com/ajoudaki/loss-of-plasticity.
- **Проверка.** `full_text`: аннотация v3, §1–§4, App. A.4–A.5, B.1.
- **Что сделано.** LoP определена как «entrapment of gradient dynamics within invariant sub-manifolds of the parameter space»: frozen-unit и cloned-unit многообразия (Thm 2.1, 2.2). Rank–plasticity tension (§3). Эксперименты: MLP, CNN, ResNet, ViT на 40 пятиклассовых задачах Tiny ImageNet, bit-flipping, клонирование.

**Критерий «когда вмешательство восстанавливает пластичность».** Операционального критерия нет. Есть качественная классификация (Def. A.2): по знаку Hessian в нормальных к многообразию направлениях ловушка «stable» (возмущения возвращаются), «unstable» (возмущения уходят) или «saddle» («escape is direction-dependent»). Сами авторы (§2): «a comprehensive theoretical analysis of the stability of empirically observed LoP manifolds is beyond the scope of this work ... Certain types of noise or symmetry-breaking operations can help models escape these manifolds». Выход проверяется на Noisy SGD, dropout, Continual Backprop. §4: «the optimal strategy for maintaining or recovering plasticity might be context-dependent».

**Угроза: нет.** Для нас полезно: Remark 2.4 и Remark A.3 утверждают, что точные клоны остаются на многообразии при SGD, momentum и Adam, «as long as the optimizer is initialized at the onset of cloning» (состояния клонов совпадают). **Наш вывод:** на точной симметричной ловушке сброс состояния Adam сам по себе не должен её разрушать, а возмущение весов (head reset, noise) может. Это проверяемое предсказание для нашего синтетического контроля, не результат статьи.

### 2.4 FIRE — Frobenius-Isometry Reinitialization

- **Метаданные.** Isaac Han, Sangyeon Park, Seungwon Oh, Donghu Kim, Hojoon Lee, Kyung-Joong Kim. arXiv 2602.08040, v1 2026-02-08, **читалась v3 (2026-04-01)**; comment «ICLR'26 (oral)».
- **Проверка.** `full_text`: §1, §2, §3.3–3.4, §4.1–4.4, App. E.3; §3.1–3.2 (меры SFE и DfI, Thm 1–4) и доказательства (App. A) не читались, только их формулировки в аннотации и §1.
- **Что сделано.** Reinitialization как условная оптимизация: `min ‖W − W̃‖²_F` (stability, SFE) при `W̃ᵀW̃ = I` (plasticity, DfI = 0); решение — полярное разложение, приближается итерациями Ньютона–Шульца (5 итераций, «less than 1%» времени). Оценено на CIFAR-10/ResNet-18, GPT-0.1B и RL: DQN на Asterix/BeamRider/DemonAttack, SAC+SimBa на 3 задачах HumanoidBench, 5 seeds.
- **Выбор между вмешательствами:** нет. «We perform a single intervention (Full Reset, S&P, FIRE) at the midpoint of learning» (App. E.3). Baselines: full reset, S&P (энкодер S&P + reset FC-слоёв для дискретного управления), Plasticity Injection. Протокол близок к нашему: «we reinitialized the network using the same checkpoint and replay buffer» (§4.3).
- SFE и DfI используются как цель оптимизации и как постфактум метрики в абляции (Fig. 5b), не как признаки выбора до repair.
- Обращение с состоянием Adam при reinit в проверенных разделах не описано.

**Угроза: нет.** Для меню: новый weights-only repair. Для мотивации неоднородности: «unlike in continual visual learning (Section 4.1), full reset performs poorly in this setting» (§4.2, LLM); «Plasticity Injection ... shows poor performance across discrete and control tasks» (§4.3).

### 2.5 Calibrated Partial Resets (2607.24996) — карточка уже есть

Карточка `mccutcheon2026cpr` (`bibliography/data_selection.json`) сверена с текстом: utility = «mean gradient magnitude of its incoming weights», layer-normalized и сглажена EMA (Eq. 5–6, §3); частичный возврат к инициализации с коэффициентом `ρ·φ(u)`; бенчмарки SlipperyAnt/SlipperyHumanoid (400M шагов, 15 seeds), Continual MetaWorld, Continual MinAtar. Расхождений нет, дублировать не нужно. Не отражено в карточке, но полезно: App. H — «We did not find base optimizer parameter resets necessary for our method»; §5 — «modest gains can be made by adjusting ρ per environment» и «Future work should isolate roles of utility estimation, partial reinitialization, optimizer-state handling». Угроза: нет (один метод, силу задаёт `ρ`).

### 2.6 Sample Weight Decay (2604.01913) и Spectral Collapse (2509.22335)

Обе работы уже в реестре (`wu2026sampleweightdecay`, `prakash2026spectralcollapse`) на уровне `full_text`. Доп. проверка на forward-вопрос:

- **SWD**, §6.5: «SWD combined with S&P yields the best result, validating its orthogonality to NTK-based methods» (Humanoid Run) — свидетельство, что data-side и weights-side repairs комбинируются. Выбора между вмешательствами и предсказания нет. Угроза: нет.
- **Spectral Collapse**, v3 HTML обрывается после §5 (Permuted MNIST); экспериментальные разделы здесь не перечитывались. Def. 4.4: fast-собственные значения `λ ≥ ε_{K,γ} ≈ −log γ / (ηK)`; Lemma 4.7 — необходимое условие успешного обучения на новой задаче: доля начального residual в fast-подпространстве. Это **prospective, зависящее от бюджета `K`** условие, но для линеаризованной ReLU-сети с замороженными вентилями; выбор вмешательства не рассматривается. Угроза: нет.

## 3. Найдено сверх списка

### 3.1 The Three Regimes of Offline-to-Online RL — ближайшее по форме претензии

- **Метаданные.** Lu Li, Tianwei Ni, Yihao Sun, Pierre-Luc Bacon. arXiv 2510.01460, v1 2025-10-01, **читалась v4 (2026-07-03)**. `full_text`: аннотация, §1–§6 (приложения B–D не читались).
- **Что сделано.** Режим определяется по измеренным **до онлайн-дообучения** возвратам предобученной политики `J(π0)` и датасета `J(π_D)`: Superior / Comparable / Inferior (t-test с margin δ = 0.05, §5). Режим диктует, какое семейство design choices выигрывает: π0-centric (online data warmup, offline RL regularization), D-centric (offline data replay, replay + **reset**) или смешанное. §1: «By first determining the regime, one can select or design fine-tuning strategies that match its stability-plasticity requirements.»
- **Оценка.** 63 settings (21 пара датасет×задача × 3 алгоритма предобучения, 10 seeds). Table 1: 45/63 верных предсказаний, 15 соседних, 3 противоположных; «71% correct predictions with only 5% opposite mismatches». Альтернативные таксономии (Table 2): по Q-функции 51% / 30% противоположных, по BC 41% / 5%. §5.2: «offline data replay with reset» лучше «replay» в 13 из 23 settings Inferior-режима.

| Признак | Three Regimes |
|---|---|
| Inputs до вмешательства | Да: два скаляра (возвраты), дёшево |
| Несколько типов repair | Семейства design choices; reset — одна из опций |
| RL | Да, но offline-to-online, не деградация онлайн-агента |
| Оценка на отложенных средах | Правило не обучается; метрика — согласие классов, не regret; сравниваются сильнейшие представители семейств в каждом setting |

**Угроза: частичная.** Это ближайший к «prospective choice по pre-intervention измерению с валидацией на многих settings» результат в RL. Он не закрывает наш вопрос (другие действия, нет обучаемого селектора, нет regret, нет mid-training repair), но: (i) сужает формулировку; (ii) показывает, что **простое правило по 1–2 скалярам может быть сильным конкурентом**; (iii) задаёт метрику «согласие с лучшим классом», которую стоит публиковать рядом с regret.

### 3.2 Beyond One-Size-Fits-All: Diagnosis-Driven Online RL with Offline Priors (позиционная статья)

- Guozheng Ma, Lu Li, Zilin Wang, Pierre-Luc Bacon, Dacheng Tao. arXiv 2606.25527v1, 2026-06-24. `full_text`: §13–§18 (Fig. 2, §16.1–16.2); приложения не читались.
- Тезис: «No single reliance configuration is universally optimal» (§16.1); контролируемые эксперименты в offline-to-online RL — бинарные выборы (инициализация весами offline-политики против случайной, консервативный штраф, удержание offline-данных) дают обратные результаты на разных задачах. §16.2 различает «prior-deployment match assessment» **до** онлайн-обучения и «tension dynamics monitoring» в ходе. Вывод (Conclusion): новый класс вкладов — «not a method that wins on a benchmark, but a signal that predicts when a design choice helps or hurts».
- Метода-селектора нет. **Угроза: частичная, на уровне рамки:** нельзя заявлять сам тезис «выбор вмешательства должен опираться на диагностику» как новый; наш вклад — конкретная реализация и проверка на plasticity-repair меню.

### 3.3 Ren et al. — Learning Dynamics of Continual Learning (Lyle — соавтор)

- Yi Ren, Wenlong Deng, Guanzhe Hong, Clare Lyle, Yarin Gal. arXiv 2609.33620v1, 2026-09-27. `full_text`: аннотация, §5–§6 (Fig. 6–7), §1 выборочно; §2–§4 и App. A–F не читались.
- Диагностика readout transmission `R_D = E_{u∼D}[‖wᵀg_u‖² / ‖g_u‖²]` (Eq. 7; `w` — readout, `g_u` — «сила» токена на выходе); Fig. 7(c): «Across models, tasks, and checkpoints, the two quantities are strongly correlated (Spearman ρ = 0.83 overall; 0.73 on OOD tasks)» между падением `R_D` и выигрышем от restoring the readout.
- Оговорки: языковые модели, **одно** вмешательство, которое авторы называют «a mechanistic intervention rather than a deployment strategy»; корреляция агрегирована, не обучен и не оценён отложенно предиктор, regret нет. **Угроза: частичная:** ближайший пример «диагностика до repair ↔ относительная польза repair», но для одного действия и вне RL.
- **Идея для признаков (наша):** transmission через head — дешёвый кандидат для действия `head_reset` у DQN.

### 3.4 Hofmann & Mäder — когда история обучения предсказывает будущее обучение лучше текущего состояния

- Martin Hofmann, Patrick Mäder. arXiv 2610.09621v1, 2026-10-07. `full_text`: §1–§3, §4.3, §5–§8 (подробности результатов Stage 1–2 в §4.1–4.2 и приложения не читались).
- **Probe как измерение:** копия сети обучается `K = 100` обновлений на новой задаче **со свежим состоянием оптимизатора**; отклик `Y = (1/K) Σ (L0 − Lk)`. Перед чтением результата probe проверен: positive control (функционально-сохраняющее масштабирование юнитов даёт монотонный отклик), ICC 0.940 [0.903, 0.997] в наименее надёжном классе (среднее трёх повторов).
- **Результат:** компактная история не улучшила калиброванную модель текущего состояния (gain −21.4% [−91.9, 8.1] при требуемых +10%). Baseline B1 использует 748 признаков (возраст обучения, потери, нормы весов, статистики активаций, ранги, нормы и косинусы градиентов, статистики моментов, изменение loss после одного пробного шага, выходы на 64 фиксированных входах).
- Ограничения (§7, §4.3): малые MLP на синтетике, 6 тестовых histories, нужно 195–276 тестовых histories для 80% power при gain 10%; exploratory. Важно для нас: «100 to 200 updates after a re-initialisation, the probe no longer separated those histories» (§6) — след вмешательства исчезает из probe за сотни обновлений.
- **Угроза: частичная, методологическая.** Вмешательств нет. Полезно: протокол валидации probe (positive control, ICC) для нашего short-probe baseline; аргумент в пользу baseline «калиброванное текущее состояние»; напоминание, что малые выборки разрешают только большие эффекты (их оценка мощности относится к их постановке; сопоставимость с нашей оценкой в ≈24 независимых обучения не проверялась).

### 3.5 Local redundancy (Cheng, ICML 2026 Spotlight)

- Jiaxuan Cheng. arXiv 2607.13432v1, 2026-07-15. `full_text`: §3.6, §4.1–4.3, §5, App. A.5 (теория §3.1–3.5 не читалась).
- Мера пластичности, оцениваемая **снизу** средним квадратом нормы градиента на синтетической memorization-выборке (метки сэмплируются из предсказательного распределения модели; один backward, +1.65% времени обучения, A.5). Continual ImageNet (MobileNetV3): остаточная корреляция с будущей точностью после вычитания номера задачи 8.9 ± 0.4 (×100, Pearson) против 7.3 для distance from init. Выбор чекпойнта (ETT): 0.1937 против 0.1952 по минимальной val-loss; linear probing 0.1918, но требует разметки целевой задачи.
- Оговорки самого текста (Limitations): «a controlled study that intervenes on plasticity directly ... is needed to test causation». **Угроза: частичная, только как признак**; эффект малый. Признак дешёвый (один backward), не требует данных целевой задачи; для DQN-регрессии можно строить аналог с шумом вокруг собственных предсказаний (наш вывод, не проверено).

### 3.6 Bu et al. — ранний сигнал дивергенции в high-UTD SAC

- Tianqi Bu, YuXuan Peng, Junteng Tu, Henghui Xiao. arXiv 2609.32819v1, 2026-09-26. `full_text`: §1, §3, §5–§6; результаты §4.1–4.5 читались по резюме в §1; приложения не читались.
- Наклон `log10|Q|` в окне (5k, 15k] шагов ранжирует seeds по времени до срабатывания guard: Harrell C 0.78 (Walker2d), 0.98 (Ant) при UTD = 4; growth-abort экономит 9.9% compute на Walker2d out-of-sample и «stays net-positive live». Dormancy: «Dormancy carries no seed-level warning of a flag» (§1). «Out of sample» = leave-one-seed-out **внутри одной конфигурации** (§3). Эксперименты: 612 прогонов SAC/TD3, 5 задач MuJoCo, критик шириной 2048 без нормализации; DQN-пилот на MinAtar «never split» (App. I).
- **Угроза: частичная:** prospective предсказание отказа в RL по телеметрии; выбора repair нет; held-out среды нет. Для нас: (i) признак «скорость роста |Q|» требует истории телеметрии; (ii) подтверждает, что dormancy может не работать как предиктор; (iii) подчёркивает разницу между «out-of-sample по seed» и переносом на среды.

### 3.7 Прочие прочитанные работы (угроза: нет)

- **He 2026, «Rethinking Plasticity in Deep RL»** (2603.21173v1; `full_text` §1–§3.2, HTML со сноской «An incomplete version», возможно усечён). Тезис: «plasticity loss is highly task-specific; notably, networks with high dormancy rates in one task can achieve performance parity with randomly initialized networks when switched to a significantly different task». Fig. 1: PPO с ReLU при более высокой dormancy сходится не хуже PPO без активаций. Один автор, малые демонстрации. Для нас: скептический prior против универсального диагностического сигнала.
- **Lanzillotta et al., «To Retain or to Adapt?»** (2607.05609v1; `full_text` §1, §2.4, §3; §2.1–2.3 и §4–§7 не читались). Lemma 6: `N∞_max = (Δ̄(ITL) − Δ̄(JTL)) / Ī` — горизонт задачи, за пределами которого retention (warm-start) становится помехой; Instability и Transient Error оцениваются постфактум по ансамблям прогонов, не по диагностике. Проверено на CLEAR, MD5 и Meta-World MT10. Для нас: **горизонт — явная ось выбора reset vs retain**, поэтому regret нужно публиковать при нескольких горизонтах.

### 3.8 Просмотрено по аннотации и не внесено в реестр

| arXiv | Название | Почему не внесено |
|---|---|---|
| 2605.06834 | Attribution-Based Neuron Utility for Plasticity Restoration (GXD) | выбирает **какие юниты** сбросить (оценка функциональной цены замены), не тип repair |
| 2607.17922 | PRIME (He et al.) | одна процедура reset по двум диагностикам; «gains ... to the gradient signal ... rather than to the specific reset operator» |
| 2605.26357 | Fast and Slow Successor Features (Chua, Precup, Richards) | при плавном дрейфе «methods favoring stability ... outperform ... parameters resetting» — свидетельство режимной зависимости; селектора нет |
| NeurIPS 2025 | Dual Nature of Plasticity Loss (Wang et al.) | два типа LoP по знаку FTLE, один метод (Generalized Mixup); выбор метода не описан (аннотация через WebFetch) |
| 2512.24445 | Adaptive Learning Guided by Bias-Noise-Alignment Diagnostics | теория без экспериментов; модулирует шаг, не выбирает repair |
| 2605.11978 | Predicting Post-training Potential of Pre-trained LLMs (RuDE) | LLM; прогноз качества базовой модели, не repair |
| 2507.20057 | What Can Grokking Teach Us about Learning under Nonstationarity? | один метод (рост effective LR) |
| 2608.07086 | ROSER (Zhao, Ma и др.): «the efficacy of different components exhibits significant task-dependency» | свидетельство неоднородности; селектора нет |
| 2610.01583, 2512.01034, 2602.11137, 2509.22562, 2605.04712, 2608.18319, 2606.25335, 2606.09932, 2603.20860 | нейроэволюция; AltNet; weight decay для LM; активации; SPHERE; SingularClip; KNIFE; Rejuvenation; targeted re-init для transfer | единичные методы, не выбор между ними |

## 4. Что меняется в формулировке вклада: предложения правок для `docs/novelty.md`

`docs/novelty.md` не изменялся. Ниже — текст для вставки.

**4.1. Раздел «Нельзя заявлять как новый вклад» — добавить:**

> - Дискретный выбор между reset, продолжением и переносом инициализации по измерению на новой задаче: [FAME, ICLR 2026, §3.1.2, App. E–F](https://arxiv.org/abs/2603.00903) — evaluation-only оценка кандидатов, выбор по эмпирическому рангу, reset = полная реинициализация.
> - Правило выбора семейства вмешательств по измеренным до дообучения величинам с валидацией на десятках settings: [Three Regimes, §3.2, Table 1](https://arxiv.org/abs/2510.01460) (offline-to-online RL, 45/63); тезис «diagnosis-driven»: [Ma et al., 2026, §16.2](https://arxiv.org/abs/2606.25527).
> - Prospective предсказание trainability checkpoint'а по дешёвым метрикам с проверкой на отложенных задачах в supervised CL: [optimization readiness](https://arxiv.org/abs/2605.09044), [local redundancy](https://arxiv.org/abs/2607.13432); протокол «короткий probe + калиброванный baseline текущего состояния»: [Hofmann & Mäder, 2026](https://arxiv.org/abs/2610.09621).
> - Связь диагностики с выигрышем от одного reset: [Ren et al., 2026, §5, Fig. 7(c)](https://arxiv.org/abs/2609.33620) (LLM, Spearman 0.83).
> - Постановка «по характеристикам задачи предсказать, какое вмешательство сработает» как открытая проблема: [Survey v3, §6.1, §7](https://arxiv.org/abs/2411.04832v3).

**4.2. Раздел «Кандидат на конкретный вклад».** Заменить абзац-формулировку на:

> По дешёвым разрешённым diagnostics состояния чекпойнта — **без rollout кандидатов** — предсказывать ожидаемую пользу каждого действия из небольшого repair menu, **включающего state-only действия** (сброс состояния Adam, который не меняет функцию и потому не виден evaluation-only пробе), заменяя пробное дообучение всех вариантов; обучаться на controlled checkpoint branches; оценивать return/regret и total cost на полностью отложенных RL environments. **Решение принимается mid-training внутри одной среды**, а не при смене задачи или старте fine-tuning.

И к списку различий добавить пятый пункт:

> 5. **Вход и момент решения:** признаки состояния сети (не возвраты кандидатов, как в FAME, и не пара скаляров о качестве priors, как в Three Regimes); решение о repair на уже обучавшемся агенте.

В абзаце про short-probe chooser добавить: «Evaluation-only вариант (в духе FAME) считать отдельным дешёвым baseline: он не отличает optimizer-only repair от `continue`; основной short-probe chooser включает короткое дообучение.»

**4.3. Раздел «Главные конкурирующие объяснения» — добавить:**

> - Весь выигрыш селектора воспроизводится правилом по одному-двум скалярам (возврат, plateau slope; ср. Three Regimes).
> - Победитель определяется горизонтом continuation, а не признаками (Lanzillotta et al., Lemma 6): оценивать при нескольких горизонтах, горизонт фиксировать до просмотра winners.

**4.4. Раздел «Незакрытый novelty audit» — заменить первые три пункта:**

> - Forward citations: частичный проход выполнен 2026-10-09 ([notes/forward_search_20261009.md](../notes/forward_search_20261009.md)). API цитирований недоступны (Semantic Scholar 403) или неполны (OpenAlex); использован скрининг arXiv-метаданных 2025-01 … 2026-10-09 и запросы по авторам. Прямого конкурента не найдено. Открыто: экспорт citation graph вручную (Google Scholar / Semantic Scholar), скрининг по полному тексту.
> - Свежие preprints 2026: просмотрены; ближайшие — Three Regimes, FAME, Ren et al., optimization readiness. Не исчерпывающе.
> - Prospective predictor response в коде и appendix: проверены FAME (`MinAtar/FAME.py` @ d16f006, App. E–F), optimization readiness (App. C.1), local redundancy (App. A). Не проверены: приложения Three Regimes (B–D), Ren et al. (App. F), Bu et al., код FAME для Atari и Meta-World.

**4.5. Побочные предложения для `docs/experimental_protocol.md` (не правились):** кандидаты в признаки — optimization readiness (считается из существующих `pre_td_huber`, `gradient_norm` плюс 128 минибатчей), local redundancy, transmission через head, наклон `log|Q|` по окну телеметрии; baseline «правило по двум скалярам»; positive control и повторные измерения для short-probe (по Hofmann & Mäder).

## 5. Что не проверено

- Citation graph не экспортировался; отсутствие найденного конкурента не доказывает его отсутствия.
- Не читались: доказательства (FAME App. C, OR App. B, FIRE App. A, Barriers A.1–A.3), приложения Three Regimes, Ren et al. App. F, Bu et al. App. A–K, код FAME для Atari и Meta-World.
- Расхождение «random-кандидат оценивается Q-значением» в `MinAtar/FAME.py` не сверялось с опубликованными числами FAME.
- OR для DQN не вычислялась, только проанализирована применимость по коду; результаты статей не воспроизводились.
- Spectral Collapse v3: HTML обрывается после §5, эксперименты v3 здесь не перечитывались.
