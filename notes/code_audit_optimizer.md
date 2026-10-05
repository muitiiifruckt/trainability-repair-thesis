# Audit optimizer repairs: semantics, код и смешивающие факторы

Проверено 5 октября 2026. Это аудит релевантных формул и исходников, не воспроизведение Atari/PPO результатов. Загруженный сторонний код только прочитан; он не выполнялся. Исходники и metadata сохранены в `research/external/optimizer/`. Предыдущий обзор и JSON-библиография не изменены.

## Что непосредственно следует для пилота

Нужно хранить **моментные буферы, часы их bias correction и глобальное расписание LR отдельно**. «Reset optimizer» недостаточно точное описание вмешательства. При частичном reset общий счётчик Adam может создать сильное изменение update size даже на неизменном градиенте. Победивший repair обозначает полезное вмешательство в данном протоколе, а не доказанную единственную причину деградации.

Рекомендуемый основной банк — `continue`, `full_reset`, `m_reset_split_clock`, `v_reset_split_clock`, `t_reset`. Частичные reset с общим временем можно добавить отдельными arms, явно указав семантику. Ниже split clocks — **наше проектное определение**, не заявка на воспроизведение частичных ablations Asadi.

## 1. Asadi: что проверено и что остаётся неоднозначным

В Algorithm 1 full reset означает `i=0, m=0, v=0` при начале нового target-defined objective; online weights сохраняются. Частичные moment ablations §4.3 не уточняют, что происходит с `i`; авторский patch с подтверждённым commit не найден. Нельзя приписывать им split clocks или общий clock reset. [Asadi et al., Algorithm 1 и §4.3](https://arxiv.org/html/2306.17833v2).

Rainbow настройки: LR `6.25e-5`, Adam epsilon `1.5e-4`, betas `(.9,.999)`, replay `200000`, batch `64`, update period `4`, horizon `3`. DQN LR `2e-4`. В большом сравнении baseline использует `K=8000`, reset Adam — `K=1000`, reset RAdam — `K=4000`; это разные target schedules. §4.1 называет K числом gradient updates, Appendix называет target update period: единицы нельзя считать подтверждёнными через авторский код. [§4.4 и Appendix 7.1](https://arxiv.org/html/2306.17833v2).

## 2. Adam-Rel: точные часы и условия сравнения

Algorithm 1 сбрасывает **только `t`**, сохраняя `m,v`; Adam-MR сбрасывает все три. В PPO это происходит перед обучением на новом rollout batch, до всех его epochs/minibatches, а не перед каждым minibatch. При неизменном objective t-reset тоже создаёт schedule. [Ellis et al., Algorithm 1 и §4](https://arxiv.org/html/2412.17113v1).

LR настраивается отдельно для DQN методов; для PPO также подбираются clipping и GAE. Atari PPO Adam: LR `.00025`, max grad norm `.5`, GAE `.95`; Rel/MR: `.002`, `5`, `.9`. DQN: replay `1e6`, batch `32`, target refresh `1000` env steps, train frequency `4`; это `250` updates. Appendix B сравнивает 40M против 120M frames и оценивает чужие scores по curves: full-reset результаты близки, но их Adam baseline сильнее. [Appendices B, C, E, F](https://arxiv.org/html/2412.17113v1).

Две ссылки авторов из Appendix D — [CleanRL fork](https://anonymous.4open.science/r/cleanrl-8719/) и [PureJaxRL fork](https://anonymous.4open.science/r/rl-nn-dynamics-8D0B/) — проверены через публичный repository API: оба вернули `repository_expired`. Поэтому **код, SHA и точные Adam epsilon/defaults не подтверждены**. PPO `epsilon` в таблице не следует автоматически интерпретировать как Adam epsilon.

Вывод аудита: противоречие не является независимым повторением одного эксперимента с обратным результатом. Для нашего исследования нужны два отчёта: одинаковый LR/config для механистического разбора и отдельная настройка каждого arm на training/validation groups для сравнения полезности selector. Выбор LR по test checkpoint outcomes — утечка.

## 3. Частичные reset: практическая спецификация

Adam update в пилоте определяем явно:

\[
m\leftarrow\beta_1m+(1-\beta_1)g,\qquad
v\leftarrow\beta_2v+(1-\beta_2)g^2,
\]

\[
t_m\leftarrow t_m+1,\quad t_v\leftarrow t_v+1,\qquad
\Delta\theta=-\alpha_{\mathrm{global}}
\frac{m/(1-\beta_1^{t_m})}{\sqrt{v/(1-\beta_2^{t_v})}+\epsilon}.
\]

При отсутствии частичных reset и `t_m=t_v` это обычная формула Adam. Библиотечная формула использует общую `t` в обеих коррекциях; изменение на два счётчика надо документировать как собственную модификацию. [Официальная формула PyTorch Adam](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Adam.html).

Спецификация arms, применяемая **до первого post-checkpoint gradient update**:

- `continue`: сохранить `m,v,t_m,t_v`.
- `full_reset`: обнулить `m,v,t_m,t_v`.
- `m_reset_split_clock`: обнулить `m,t_m`; сохранить `v,t_v`.
- `v_reset_split_clock`: обнулить `v,t_v`; сохранить `m,t_m`.
- `t_reset`: сохранить `m,v`, обнулить `t_m,t_v`. При одинаковых clocks до вмешательства это t-only Adam-Rel semantics; это намеренное изменение коррекции, а не «восстановление unbiased estimates».

`m_buffer_only_keep_t`, `v_buffer_only_keep_t`, `m_buffer_reset_shared_t`, `v_buffer_reset_shared_t` — допустимые дополнительные interventions, но это другие arms. Ни одна из этих четырёх операций не подтверждена как точная реализация partial ablations Asadi.

Все arrays клонируются независимо. Состояние содержит также global training step, LR scheduler, betas, epsilon, clipping, RNG и модельный state. Reset moment clock не должен случайно перезапускать глобальное расписание LR. Для первого пилота разумно исключить AMSGrad/weight decay; если они нужны, заранее определить reset `v_max`, decoupled decay и scheduler. Градиенты с `None` и параметры без update могут иметь собственные per-parameter clocks; dense NumPy pilot проще, но это ограничение переносимости.

### Проверка масштаба на постоянном градиенте

Следующие числа — **наш аналитический sanity check**. Предполагаются mature state `m≈g`, `v≈g²`, общий большой `t`, постоянный ненулевой scalar `g`, betas `(.9,.999)`, epsilon пренебрежимо мал:

- `continue` и `full_reset`: первый `|Δθ|/α≈1`.
- `m_buffer_only_keep_t`: `≈.1`.
- `v_buffer_only_keep_t`: `≈31.6228`.
- `t_reset`: `≈.316228`.
- `m_buffer_reset_shared_t`: `≈.0316228`.
- `v_buffer_reset_shared_t`: `≈10`.
- split-clock partial resets: `≈1` в обоих случаях.

Это следует подстановкой обновлённых `m,v` в формулу выше. Поэтому «v reset ухудшил обучение» может означать overshoot из-за семантики clocks. «m reset помог» может означать полезное уменьшение шага. При заметном epsilon числа меняются; сравнивать нужно фактические updates.

Полезные проверки реализации: constant-gradient числа; полное совпадение clone+continue с исходным продолжением; веса равны сразу после optimizer-only repair; arrays не разделяют память; изменение одной ветки не влияет на другие. Отдельно проверить скачок gradient magnitude и поворот gradient direction: constant-gradient sanity не моделирует нестационарность.

### Как отличить память от изменения шага

До вмешательства сохранять `||g||`, cosine(`g,m`), distributions `|g|/sqrt(v+epsilon²)`, norms и cosine **предложенного** Adam update для каждого arm. Это расчёт по probe gradient без обновления основной ветки; стоимость probe включается в diagnostic budget. Cosine при нулевой норме должен быть missing/undefined, а не искусственным нулём.

Логировать первые post-repair updates, включая `||Δθ||/||θ||` и изменение loss на общих reference data. Дополнительный control — LR-only schedule, согласующий первый update norm с reset arm. Такое нормирование само является вмешательством и не доказывает равенство всего будущего процесса; оно лишь проверяет альтернативное объяснение через начальный масштаб.

Нельзя обучать selector на ярлыках «optimizer broken» из победителя reset. Корректная supervised target — outcome/gain или regret каждого **точно заданного arm**. Если несколько arms статистически неразличимы, сохранять uncertainty/ties. Diagnosis механизма требует дополнительных factorial controls и устойчивости к LR tuning.

## 4. OPEN: авторский код подтверждён на фиксированном commit

Проверенный [commit `b65037bd89c09f31e80c5d49f1d78cc9c6dc43e0`](https://github.com/AlexGoldie/rl-learned-optimization/commit/b65037bd89c09f31e80c5d49f1d78cc9c6dc43e0), snapshot текущей main; не подтверждено, что это конкретный commit из experiments paper.

В [network.py, строки 48–128](https://github.com/AlexGoldie/rl-learned-optimization/blob/b65037bd89c09f31e80c5d49f1d78cc9c6dc43e0/rl_optimizer/network.py#L48) defaults: step/exp multipliers `.001`, MLP hidden `16`, GRU `8`, EMA decays `.1,.5,.9,.99,.999,.9999`; init сохраняет параметры, обнуляет EMA state и создаёт GRU carry. Входы включают gradients, moments, parameters, dormancy и progress. [Строки 145–255](https://github.com/AlexGoldie/rl-learned-optimization/blob/b65037bd89c09f31e80c5d49f1d78cc9c6dc43e0/rl_optimizer/network.py#L145).

Update задаётся GRU/MLP; mask добавляет noise к actor, затем mean всех actor+critic updates вычитается глобально. EMA и GRU carry сохраняются между updates. Поэтому это не Adam и его state не имеет простой интерпретации `m,v,t`. [Строки 257–379](https://github.com/AlexGoldie/rl-learned-optimization/blob/b65037bd89c09f31e80c5d49f1d78cc9c6dc43e0/rl_optimizer/network.py#L257).

Dependency `learned_optimization` закреплена submodule commit `4bcaeb06799bf68456cc61ffb36886d40d316e50`; [common.py, строки 25–83](https://github.com/google/learned_optimization/blob/4bcaeb06799bf68456cc61ffb36886d40d316e50/learned_optimization/learned_optimizers/common.py#L25) подтверждает first-moment EMA с нулевым init и увеличением собственного `t`, без bias correction. В OPEN используется `.m`; `_second_moment_normalizer` — RMS текущих features по tensor dimensions, а не исторический Adam `v`.

В [train.py, строки 254–263](https://github.com/AlexGoldie/rl-learned-optimization/blob/b65037bd89c09f31e80c5d49f1d78cc9c6dc43e0/rl_optimizer/train.py#L254) larger network имеет hidden `32`, GRU `16`; actor/critic gradients clipping применяется раздельно [строки 459–463](https://github.com/AlexGoldie/rl-learned-optimization/blob/b65037bd89c09f31e80c5d49f1d78cc9c6dc43e0/rl_optimizer/train.py#L459). `optax.adam(b1=.99)` [строки 815–825](https://github.com/AlexGoldie/rl-learned-optimization/blob/b65037bd89c09f31e80c5d49f1d78cc9c6dc43e0/rl_optimizer/train.py#L815) относится к **внешнему ES meta-optimizer**. Его нельзя описывать как Adam learner baseline.

Asterix config содержит 64 envs × 128 rollout steps, 4 epochs, 8 minibatches, max norm `.5`, hidden size `64`, ReLU. Флаг `ANNEAL_LR=True` присутствует, но поиск в проверенном train.py не нашёл его использования: presence flag не подтверждает annealing. [configs.py, строки 4–23](https://github.com/AlexGoldie/rl-learned-optimization/blob/b65037bd89c09f31e80c5d49f1d78cc9c6dc43e0/rl_optimizer/configs.py#L4).

Следствие для проекта: OPEN — полезный prior на diagnostics и соседний adaptive baseline, но прямое сравнение требует meta-training budget, pretrained optimizer state и шумового протокола; простой t-only reset не воспроизводит OPEN.

## 5. SWD: проверка assumptions и текущей реализации

Theorem 3 опирается на growing replay `|D_k|=k` и exact previous minimizer: old-risk gradient в proof равен нулю. `1/k` стоит только перед distribution-shift term; target drift исчезает при terminal `h=H`. Универсальное затухание всех RL gradients из этого не следует. [SWD, Proposition 1, Theorem 3, Appendix B.3](https://proceedings.iclr.cc/paper_files/paper/2026/file/2886558a03d11f95d4020a460fe4a390-Paper-Conference.pdf).

Наша проверка границы утверждения: при bounded new-example gradient получаем `O(1/k)` для первого term; `Θ(1/k)` требует нижней границы на его norm. Для SWD weights с положительным floor `c`, growing replay даёт `sum w_i≥kc`, значит `p_new≤1/(kc)`. Поэтому decay+floor не устраняет асимптотическую dilution в неограниченном buffer. Fixed-capacity ring buffer — другая постановка.

Авторская ссылка ведёт на CleanRL-JAX. Проверенный [commit `18ca645a72d4e78d4d4a63a5f6745f17837dd935`](https://github.com/wzhhasadream/CleanRL-JAX/commit/18ca645a72d4e78d4d4a63a5f6745f17837dd935):

- [ReplayBuffer.py, строки 49–69](https://github.com/wzhhasadream/CleanRL-JAX/blob/18ca645a72d4e78d4d4a63a5f6745f17837dd935/cleanrl_jax/utils/ReplayBuffer.py#L49): defaults capacity `1e6`, decay `0` (uniform), floor `.1`.
- [Строки 107–142](https://github.com/wzhhasadream/CleanRL-JAX/blob/18ca645a72d4e78d4d4a63a5f6745f17837dd935/cleanrl_jax/utils/ReplayBuffer.py#L107): ring overwrite; clock растёт раз за вызов `add`, не раз за transition при нескольких envs.
- [Строки 186–200](https://github.com/wzhhasadream/CleanRL-JAX/blob/18ca645a72d4e78d4d4a63a5f6745f17837dd935/cleanrl_jax/utils/ReplayBuffer.py#L186): положительный decay даёт `max(floor,1-age/T)`, нормализованные sampling probabilities, draw с replacement; отрицательный decay предпочитает старые данные.
- [train_sac.py, строки 38–51](https://github.com/wzhhasadream/CleanRL-JAX/blob/18ca645a72d4e78d4d4a63a5f6745f17837dd935/train_sac.py#L38): default decay `80000`; [строки 149–151](https://github.com/wzhhasadream/CleanRL-JAX/blob/18ca645a72d4e78d4d4a63a5f6745f17837dd935/train_sac.py#L149) передают его buffer.

Snapshot содержит SAC/TD3; полное соответствие опубликованным Double-DQN экспериментам и paper commit не подтверждено. SWD следует называть intervention в sampling distribution, а не чистым увеличением gradient magnitude: меняются state/action coverage и task weighting.

## 6. AdamO: формула и ограничение теоретического аргумента

В §4.4 моменты обновляются только task gradient, а Gram-penalty gradient добавляется отдельным parameter step; это не moment reset. Формула Eq.20 использует regularizer в текущих weights, `eta_iso=eta` по умолчанию, но допускает tuning. Eq.16 выбирает smaller Gram: `WᵀW-I` для tall matrix, `WWᵀ-I` для wide. Теория NTK опирается на isotropic/high-dimensional input prior; гарантированного revival всех dead gates нет. [AdamO, §2.3, §4.2–4.4](https://arxiv.org/html/2606.09762v1).

Наш контрпример к обобщению Eq.14 на произвольную rectangular isometry:

\[
B=\begin{bmatrix}1\\0\end{bmatrix},\quad
v=\begin{bmatrix}0\\1\end{bmatrix};\qquad
B^TB=1,\quad B^Tv=0,\quad \|v\|=1.
\]

Следовательно, `BᵀB=I` не гарантирует сохранение norm всех output gradients при backpropagation. Для этого нужно `BBᵀ=I`, в частности square orthogonal B. Это локальная математическая оговорка к формулировке §3.1, не опровержение empirical AdamO результатов. В ReLU также `sigma_min(D)>0` из condition-number bound §4.1 нарушается на закрытых gates. Поэтому orthogonality diagnostic не является универсальной сертификацией trainability.

## 7. Контракт воспроизводимости pilot

Это проектные рекомендации:

1. Сохранить один immutable checkpoint; arms получают независимые copies. Diagnostic вычисляется до выбора repair на фиксированном probe set.
2. Зафиксировать LR, epsilon, betas, update budget, minibatch schedule, evaluator и RNG seeds; явно записать момент repair относительно gradient update/target refresh.
3. Separate logging: env steps, optimizer updates, target refreshes, moment clocks, scheduler clock. В toy pilot фиксированные data позволяют убрать collector dynamics; это ограничивает выводы о live RL.
4. Первый контроль — одинаковая nominal LR; второй — LR tuning каждого arm на grouped train/validation. Оба нужны, потому что один отвечает о механизме, другой — о полезной repair policy.
5. Сохранять scores всех arms и ties; label лучший repair только внутри конкретного budget/config. Не выводить четыре causal classes из пяти optimizer arms.
6. Generalization split должен отделять task families/environments, а не только seeds соседних checkpoint одного training run. Начальный toy proof-of-concept не доказывает unseen-environment RL transfer.
