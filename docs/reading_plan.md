# Как войти в тему и что читать первым

Цель чтения — прийти к проверяемому эксперименту, а не накопить список papers. Ниже порядок с конкретным результатом каждого блока; это не календарь и не оценка времени.

## Сначала три основных текста

1. [Understanding Plasticity — Lyle et al., ICML 2023](https://proceedings.mlr.press/v202/lyle23b/lyle23b.pdf). Читать §2.2–3.1, case studies и falsification/limitations. Выписать точное probe definition, optimizer/budget и контрпримеры простым metrics. Результат: определение trainability и различие с reward/generalization.
2. [Plasticity Injection — Nikishin et al., NeurIPS 2023](https://papers.neurips.cc/paper_files/paper/2023/file/75101364dc3aa7772d27528ea504472b-Paper-Conference.pdf). Читать construction, §5.2 и Appendix B. Выписать, какие states сохраняются, какие меняются и как выглядит function-preserving intervention. Результат: baseline, с которым сравнивается наша диагностика.
3. [Disentangling — Lyle et al., CoLLAs 2024 / proceedings 2025](https://proceedings.mlr.press/v274/lyle25a.html). Читать controlled mechanisms и сочетания методов. Результат: заменить «четыре независимые причины» на явно заданные места вмешательств и interactions.

Survey [Klein et al., v3 April 2026](https://arxiv.org/html/2411.04832v3) читать выборочно параллельно как карту терминов; не подменять им первичные доказательства.

## Затем прочитать конфликт и ближайших конкурентов

4. [Primacy Bias, ICML 2022](https://proceedings.mlr.press/v162/nikishin22a/nikishin22a.pdf), особенно Appendix B / Fig. 9. Результат: что именно известно об optimizer-only и parameter-only reset.
5. [Resetting the Optimizer, NeurIPS 2023](https://arxiv.org/html/2306.17833v2) **в паре** с [Adam on Local Time, NeurIPS 2024](https://arxiv.org/html/2412.17113v1). Выписать target-update schedule, LR tuning, m/v/t и bias correction. Результат: specification optimizer actions без смешения step-size эффектов.
6. [On-Policy Study, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ce7984e36d58659211a8dc7d5457cd6f-Abstract-Conference.html). Особенно момент измерения diagnostics, unit of observation GLM и intervention setup. Результат: точная граница prospective vs retrospective predictor.
7. [Adaptive RR, ICLR 2024](https://arxiv.org/abs/2310.07418), [OPEN, NeurIPS 2024](https://papers.nips.cc/paper/2024/file/09e1944b7f2372f9f81866470c59b663-Paper-Conference.pdf) и [Automatic Soft Reset, NeurIPS 2024](https://arxiv.org/abs/2411.04034). Результат: ближайшие policies и справедливое сравнение budget/training history.
8. [NeuMoSync, August 2026](https://arxiv.org/abs/2608.04358). Результат: проверить learned multi-mechanism control и snapshot evaluation как риск novelty. Ограничить claims проверенной областью supervised CL.

Для PPO отдельно: [Moalla et al., 2024](https://arxiv.org/html/2405.00662v3) и [Pyatko et al., 2025](https://openreview.net/pdf?id=6CViR7tKj2); [заметка](../notes/additional_onpolicy.md). Проверить optimizer, используемый capacity probe, и отличие глобального hyperparameter tuning от выбора действия для текущего checkpoint.

Добавить после Injection **[TeLAPA, Appendix D](https://arxiv.org/html/2604.15414v2)**: ближайшее сравнение с реальным short-probe choice. Затем [PAME](https://ifaamas.csc.liv.ac.uk/Proceedings/aamas2025/pdfs/p2299.pdf) для selective-module trigger. Результат — точное различие дешёвого predictor и выбора через пробное обучение.

## После этого выбрать одну глубокую ветку

- **Параметры/представления:** ReDo → Continual Backprop → Spectral Collapse. Сопоставить dormancy, utility, spectrum и residual alignment. Подробности: [заметка](../notes/parameter_plasticity.md).
- **Optimizer/targets:** DR3 → Stop Regressing → C-CHAIN/AdamO. Разделить target generator, encoding, loss и memory. Подробности: [заметка](../notes/optimizer_targets.md).
- **Данные/selection:** high-UTD value divergence → ROER → CPR/SNR → algorithm-selection framing. Проверить доступность repair без privileged information. Подробности: [заметка](../notes/data_and_selection.md).

## Матчасть, которая нужна для первого результата

1. TD/Bellman backup и target network: почему target может меняться даже при фиксированных transitions.
2. Adam: моменты, bias correction, timestep и effective update; чем различаются m-only/v-only/t-only reset.
3. Локальная оптимизация: Jacobian, Gram/NTK, spectrum, conditioning, почему finite horizon зависит от eigenvalues и alignment.
4. Причинные эффекты заданных interventions: treatment response, interactions и ограниченность mechanistic attribution.
5. Statistical decision-making: noisy argmax, regret, uncertainty, grouping trajectories и selected-winner bias.

Не обязательно начинать с доказательств для полного nonlinear RL. Для пилота достаточно корректно определённого outcome, воспроизводимых interventions и честного сравнения.

## Что записывать после каждого paper

- Какая именно форма деградации измеряется, на каких данных и горизонте?
- Какие компоненты реально меняет intervention, включая optimizer/targets/schedules?
- Diagnostic вычисляется до intervention или после; сколько стоит и требует ли будущих данных?
- С чем сравнивается, есть ли LR/hyperparameter tuning и как устроены seeds/tasks?
- Где метод не помогает; какие claims авторы сами ограничивают?
- Какой один факт меняет наш протокол или опровергает нашу гипотезу?

## Первое решение после чтения

Первый code audit и synthetic control уже выполнены: [Plasticine](../notes/infrastructure_and_forward_search.md), [resets](../notes/code_audit_parameters.md), [optimizer](../notes/code_audit_optimizer.md), [результаты контроля](controlled_probe_results.md). Следующее решение — fixed-TD panel, затем маленький online factorial pilot из [протокола](experimental_protocol.md). Решение о большом selector принимать по устойчивой RL effect heterogeneity, а не по привлекательности идеи.
