# Журнал решений

## 2026-10-05 — первая постановка после поиска литературы

1. **Не заявлять новизну fixed-budget plasticity.** Такая формализация уже есть у Lyle et al. (2023).
2. **Не заявлять новизну clone-and-intervene.** Plasticity Injection (2023) прямо описывает такой диагностический эксперимент.
3. **Не заявлять новизну любого adaptive repair.** Уже существуют ReDo, Adaptive RR, automatic soft reset и learned optimizers; нужны сравнения с ближайшими методами.
4. **Рабочий вклад:** prospective выбор среди нескольких типов repair по pre-repair features, с оценкой regret на held-out environment families. Это кандидат на вклад, не подтверждённая свободная ниша.
5. **Два уровня проверки:** fixed-data/fixed-target probes для оптимизации и online RL для фактической полезности.
6. **Первые actions узкие и воспроизводимые.** Параметрный repair и optimizer reset пересечь факторно; data/target interventions сначала считать лабораторными controls, пока нет доступного deployable варианта.
7. **Отдельно effect prediction и cause identification.** Успех selector допустим без доказательства четырёх исключающих друг друга причин.
8. **Сначала проверить heterogeneity.** Если один repair почти всегда лучший, сложный selector не нужен.
9. **Не начинать масштабное обучение по одному обзору.** Сначала проверка ближайших PDF/кода и небольшой пилот. Compute был неизвестен на первом проходе; замер и CPU-контроль записаны ниже.
10. **Сохранять отрицательные результаты.** Отсутствие преимуществ selector или необходимость комбинированного repair — полезные итоги, если дизайн корректен.

## 2026-10-05 — независимая проверка формализации

Исправлены четыре места: primary probe loss измеряет fitting, held-out loss отдельно generalization; data×target factorial использует shared target semantics; empirical winner не равен population oracle и не является строгой bound; practical utility использует total budget с вычетом diagnostics/repair costs. Также разделены held-out environment для selector и перенос агента в новую среду.

## 2026-10-05 — приложения, авторский код и первый выполненный контроль

1. **Узить selection claim:** TeLAPA уже выбирает по фактической short-budget adaptation. Наш кандидат — amortized prediction heterogeneous repair responses; добавить short-probe chooser и учёт offline label/training cost.
2. **Не переносить repair по названию:** Juliani `none` сбрасывает optimizer на shifts; Primacy меняет дополнительные states; Plasticine replacement Parameters не попадают в старый optimizer. Все findings относятся к указанным code commits, не доказывают происхождение paper results.
3. **Писать собственный маленький fork runner:** каталог диагностик/методов заимствовать после проверки definitions. Сторонние install/training scripts не запускались.
4. **Compute:** Python 3.13.12, torch 2.12.1+cu126; RTX 3050 Laptop с 4 GiB доступна. Первый контроль CPU float64/один thread; gymnasium/MinAtar пока не установлены. GPU не понадобилась; speed полного RL не измерялась.
5. **Контроль завершён:** 24 checkpoint/288 branches, шесть tests прошли, nonfinite failures нет, 36.22s. Response зависит от искусственного scenario и age; это preliminary semantics control, не RL validation.
6. **Timestep reset не очищает memory:** точный LR+epsilon schedule воспроизвёл t-reset. Этот control нужен для разделения clock/step-size и moment-memory эффектов.
7. **Не обучать selector на synthetic scenario labels:** diagnostics знают future training labels, histories связаны; 288 branches не независимые samples. Следующий шаг — fixed transitions/frozen TD, затем closed-loop return.
8. **Следующая опорная проверка:** final loss на common held-out objective и gain от unrepaired start; Double DQN freeze включает argmax selector. При data×target factorial сравнивать одинаковый evaluation objective, а не loss относительно разных teachers.
9. **Независимый шум для probing baseline:** short-ranking stream отделён от main continuation/final evaluation. Chooser получает short outcomes, основной diagnostic predictor их не видит; final/oracle outcomes закрыты обоим.

[Audit parameters](../notes/code_audit_parameters.md), [audit optimizer](../notes/code_audit_optimizer.md), [forward search](../notes/infrastructure_and_forward_search.md), [выполненный run](../docs/controlled_probe_results.md).
