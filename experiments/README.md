# Малый контроль перед RL-пилотом

`controlled_probe.py` — собственная supervised teacher-regression задача. Она проверяет перенос состояния, branch invariants и различия repairs. Это не репликация paper, не RL и не проверка unseen-environment selector.

Запуск из корня: `python -m unittest discover -s tests -v`, затем `python experiments/controlled_probe.py`.

Первый run уже выполнен: 24 checkpoint, 288 веток, без nonfinite failures. `python scripts/summarize_controlled_probe.py` пересобирает descriptive summary и график. [Результаты и ограничения](../docs/controlled_probe_results.md).

Требуется установленный PyTorch. Default run использует CPU, один thread, float64; CUDA не нужна. Ничего из `research/external/` не импортируется и не запускается.

Меню: continue; full optimizer reset; head reset in place; head + full optimizer reset; timestep-only reset; его эквивалентный LR/epsilon schedule без reset state. Гиперпараметры Adam одинаковы до intervention. Все продолжения внутри repeat получают одинаковые данные, frozen targets и порядок minibatches.

Файлы результатов содержат все branch losses, immediate damage, curves, pre-repair diagnostics, config, code hash и runtime. Training loss и held-out loss разделены. Pre-diagnostics здесь знают labels новой teacher-задачи; такое предположение нельзя переносить на live RL.

Partial m-only/v-only resets намеренно не включены: при общем Adam timestep их semantics неоднозначны без явной спецификации bias correction. Next stage может сравнить keep-clock и separate-clock варианты отдельно.
