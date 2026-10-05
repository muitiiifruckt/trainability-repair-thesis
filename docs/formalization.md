# Формализация и минимальная матчасть

Дата: 2026-10-05. Всё ниже — предлагаемая постановка проекта, кроме явно указанных предшественников. Она требует проверки экспериментами.

## 1. Что считать состоянием агента

Checkpoint должен включать больше, чем веса:

\[
s=(\theta,o,\bar\theta,\mathcal R,n,h,z).
\]

Здесь \(\theta\) — параметры online-сети, \(o\) — optimizer state (для Adam: моменты \(m,v\), счётчик \(k\)), \(\bar\theta\) — target network, \(\mathcal R\) — replay buffer, \(n\) — состояния нормализации, \(h\) — learning-rate/exploration/target-update schedules, \(z\) — RNG и при необходимости состояние среды.

Occupancy \(d^\pi\) — распределение посещаемых состояний — не просто отдельный массив в checkpoint. Оно возникает из взаимодействия policy со средой. Замена данных из replay и изменение будущего occupancy — разные вмешательства.

## 2. Trainability имеет несколько уровней

**Способность решать фиксированную задачу обучения.** Пусть \(q\) задаёт распределение probe-задач \((X,y)\), \(U_K\) — ровно \(K\) обновлений, а \(\ell_q\) — loss на фиксированном обучаемом probe-objective. Можно измерять:

\[
T_K(s;q)=\mathbb E_q[\ell_q(\theta)-\ell_q(U_K(s;q))].
\]

Также сохраняем конечный training loss: один только gain может быть большим из-за плохого исходного loss. Отдельный held-out loss измеряет generalization и не заменяет ability to fit. При iid random labels оценивается training memorization; для held-out probe targets нужна одна frozen teacher function на обоих subsets. Начальный уровень, сложность и масштаб targets должны совпадать или быть учтены. Сравнение со свежей сетью той же архитектуры полезно как reference, но не как универсальный верхний предел.

Определение plasticity через качество после фиксированного optimization budget уже есть у [Lyle et al., ICML 2023, §2.2–3.1, eq. 5](https://proceedings.mlr.press/v202/lyle23b/lyle23b.pdf). Наша формула — прикладная адаптация этой идеи, а не новая теория plasticity.

**Способность улучшить RL-поведение.** Для среды \(e\), intervention \(a\) и бюджета \(B\):

\[
Y_B(a;s,e)=\mathbb E_\xi[J_e(\pi_{U_B(I_a(s),\xi)})],\qquad
G_B(a;s,e)=Y_B(a;s,e)-J_e(\pi_s).
\]

Бюджет \(B\) должен явно задавать environment steps, gradient updates и compute. Значения нельзя безоговорочно смешивать в одно «число шагов». \(\xi\) — случайность продолжения обучения и оценки.

**Эффект repair относительно продолжения:**

\[
\tau_B(a;s,e)=Y_B(a;s,e)-Y_B(a_0;s,e),
\]

где \(a_0\) — continue. Это практический вопрос «помогло ли вмешательство». Он отличается от утверждения «мы обнаружили единственную причину деградации».

## 3. Учитывать немедленный ущерб от reset

Измеряем отдельно \(J_{\rm pre}\), \(J_{0^+,a}\) сразу после repair и \(J_{B,a}\). Тогда:

\[
\Delta_{\rm instant}=J_{0^+,a}-J_{\rm pre},\quad
\Delta_{\rm learn}=J_{B,a}-J_{0^+,a},\quad
\Delta_{\rm total}=J_{B,a}-J_{\rm pre}.
\]

Reset может создавать большой \(\Delta_{\rm learn}\), сначала уничтожив качество policy. Поэтому выбор repair оптимизирует конечное качество или AUC адаптации, а не только «скорость от плохого старта». Plasticity Injection полезен как baseline, сохраняющий predictions в момент вмешательства; его диагностическое использование уже описано в [Nikishin et al., NeurIPS 2023, §5.2](https://papers.neurips.cc/paper_files/paper/2023/file/75101364dc3aa7772d27528ea504472b-Paper-Conference.pdf).

## 4. Selector

Диагностика \(x=D(s)\) собирается **до repair** из состояния и разрешённой небольшой выборки. Selector \(f(x)\) выбирает действие из заранее определённого конечного меню \(\mathcal A\).

\[
a_B^*(s,e)\in\arg\max_{a\in\mathcal A}Y_B(a;s,e),\qquad
R_B(f)=\mathbb E_{s,e}[\max_aY_B(a;s,e)-Y_B(f(D(s));s,e)].
\]

Oracle оптимален только среди указанных actions и для этого горизонта. Не оптимален среди всех методов RL. Если действие выигрывает на одном seed, это ещё не оценка его ожидаемого эффекта.

Эти формулы задают научное сравнение при одинаковом **бюджете продолжения**. Для practical end-to-end результата задаём total resource budget \(B_{\rm total}\), цену диагностики \(c_D(s)\) и repair \(c_I(a,s)\):

\[
\widetilde Y_{B_{\rm total}}(f;s,e)
=\mathbb E_\xi J_e\!\left(\pi_{U_{B_{\rm total}-c_D-c_I}(I_{f(D(s))}(s),\xi)}\right).
\]

Стоимость измеряется в согласованных единицах, либо как вектор ресурсов с покомпонентным ограничением. Продолжение допустимо только при неотрицательном остатке; infeasible actions исключаются заранее. Baseline без диагностики имеет собственный нулевой \(c_D\). Gradient evaluations для diagnostics и environment transitions считаются, даже если не обновляют основную сеть. Scientific equal-continuation результат и practical equal-total-budget результат публикуются отдельно.

Практический predictor лучше начать с \(\widehat\tau_a(x)\), затем выбирать \(\arg\max_a\widehat\tau_a(x)\), включая \(\tau_{a_0}=0\). Полный вектор исходов лучше жёсткой метки argmax: близкие по качеству repairs не надо превращать в искусственно разные классы.

Это частный случай выбора алгоритма по признакам instance; общий принцип существует со времён [Rice, 1976](https://www.sciencedirect.com/science/article/pii/S0065245808605203). В обучающем наборе, где проведены все repairs, задача имеет full-information outcomes. Contextual bandit нужен позже, если наблюдаем результат только выбранного действия.

## 5. Признаки и то, чего они не доказывают

Предлагаемый недорогой набор:

- Слой-за-слоем: нормы весов относительно инициализации, доля малоактивных units, нормы градиентов и обновлений.
- Adam: нормы \(m,v\), квантили effective step scale, \(\cos(g,m)\), распределение \(\|\Delta\theta_l\|/(\|\theta_l\|+\epsilon)\), возраст optimizer state.
- Данные: возраст transitions, reward/action coverage, доля terminal states, различие признаков свежих данных и replay. Сам feature extractor может исказить эту оценку.
- Targets: масштаб/дисперсия TD-target, TD error, изменение targets на одном reference batch после target update, согласованность online и target predictions.
- Если compute позволяет: feature rank, gradient interference, sensitivity/churn.

Effective rank не один стандартный показатель. Для сингулярных чисел \(\sigma_i\) можно заранее выбрать entropy rank:

\[
p_i=\sigma_i/\sum_j\sigma_j,\quad r_{\rm eff}=\exp(-\sum_i p_i\log p_i).
\]

Нужно фиксировать centering, batch size, layer, нормировку и handling нулевой матрицы. Низкий rank может отражать простой data distribution; сам по себе не удостоверяет loss of trainability.

Не путать три разных объекта:

\[
F_{ij}=\phi_j(x_i),\quad
K_{ij}=\langle\nabla_\theta f(x_i),\nabla_\theta f(x_j)\rangle,\quad
C_{ij}=\frac{\langle\nabla_\theta\ell_i,\nabla_\theta\ell_j\rangle}{\|\nabla_\theta\ell_i\|\|\nabla_\theta\ell_j\|}.
\]

Это feature matrix, empirical NTK Gram и нормированная матрица сходства loss-gradients. Hessian — четвёртый объект; его спектр не равен спектру этих матриц. Для нулевых градиентов правило обработки задаётся заранее. Полный Hessian не нужен для первого пилота.

## 6. Почему эффект вмешательства не раскрывает единственную причину

В factorial probe веса и optimizer можно пересечь. Для ожидаемого исхода \(Y_{ij}\), где \(i\) — weight repair, \(j\) — optimizer repair:

\[
\Gamma=Y_{11}-Y_{10}-Y_{01}+Y_{00}.
\]

\(\Gamma\ne0\) означает взаимодействие **этих вмешательств, на этой шкале и горизонте**. Это не универсальный decomposition причин. Weight reset меняет predictions, gradients и дальнейшие данные. Optimizer reset меняет preconditioning, bias correction и иногда effective learning rate. Data repair меняет Bellman weighting. Target freezing делает другую задачу.

Контролируемые клонированные ветки позволяют оценить условные эффекты заданных repairs; mechanistic claim требует дополнительных экспериментов и определения того, что действительно удержано постоянным.
