# Почему Adam timestep reset требует отдельного control

Это наша алгебра для используемого в [синтетическом контроле](controlled_probe_results.md) Adam, а не новый optimizer algorithm. Она уточняет интерпретацию timestep-only intervention из [Adam on Local Time](https://arxiv.org/html/2412.17113v1).

## Формула update

Пусть после очередного gradient update moments равны `m,v`. Для bias-correction clock t и epsilon снаружи sqrt:

\[
\Delta\theta(t)=-\alpha\,
\frac{m/(1-\beta_1^t)}{\sqrt{v/(1-\beta_2^t)}+\epsilon}
=-\alpha c(t)\frac{m}{\sqrt v+\epsilon d(t)},
\]

где \(d(t)=\sqrt{1-\beta_2^t}\), \(c(t)=d(t)/(1-\beta_1^t)\). Reset t не стирает m или v: он меняет их bias correction и величину шага.

Для ветки со сброшенным clock следующий update использует local time u, для control — продолжающийся global time T. Сохраняем moments и global clock, но задаём

\[
\alpha'_u=\alpha\frac{c(u)}{c(T)},\qquad
\epsilon'_u=\epsilon\frac{d(u)}{d(T)}.
\]

Тогда numerator multiplier и denominator совпадают точно. При одинаковых gradients совпадает update; по индукции совпадают будущие parameters и gradients на общих batches. Одного LR scaling недостаточно для точного равенства при ненулевом epsilon.

Условия нашей проверки: weight decay=0, одинаковая Adam recurrence, все parameters внутри group имеют общий clock, без AMSGrad/fused variants и иных скрытых состояний. При decoupled weight decay изменение LR меняет и decay update: нужен дополнительный control. Для параметров с разной историей updates один LR/epsilon на всю group в общем случае не даёт такой эквивалентности.

## Partial reset: общий clock создаёт неоднозначность

Если обнулить только m, но оставить большой t, первый новый gradient получает multiplier примерно `(1−β1)` вместо fully bias-corrected fresh m. Если обнулить только v при большом t, свежая v примерно `(1−β2)g²` без fresh bias correction; при β2=.999 и малом epsilon это может увеличить нормированный шаг примерно в 31.6 раза. Эти оценки относятся к отдельным компонентам correction, не гарантируют полный update ratio для произвольного m,v.

Поэтому каждое partial action должно явно выбирать clock policy: keep shared t, reset shared t или отдельные moment clocks. Последний вариант уже модификация стандартного Adam. В нашем первом runner m-only/v-only arms не включены.

## Что это меняет в проекте

Успех reset не означает, что moments содержали вредную память: вмешательство могло изменить step-size schedule. Для mechanistic attribution нужны сравнения с явно согласованными LR/epsilon controls, first-update norm и subsequent learning curve. Full m,v,t reset не воспроизводится одной scalar schedule в общем случае, поскольку он меняет coordinate-wise memory.

В DQN отдельно логировать environment steps, optimizer updates и target updates; в PPO — rollout index, epoch и minibatch updates. В AdamRel PPO clock reset задан перед epochs/minibatches нового rollout batch, а не при каждом minibatch. Paper-specific units нельзя переносить по одному имени `step`. [Разбор источников и кода](../notes/code_audit_optimizer.md).
