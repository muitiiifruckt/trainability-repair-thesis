# Diagnosis-guided repair of trainability in deep RL

Исследовательский репозиторий: почему RL-агент теряет способность дообучаться и можно ли **до вмешательства** выбрать подходящий способ восстановления.

## Рабочий вопрос

> Может ли диагностика текущего checkpoint выбрать вмешательство в параметры, optimizer state, данные или targets, которое на новой среде даёт больший выигрыш за фиксированный бюджет, чем лучший постоянный repair?

Это гипотеза проекта. Её новизна и экспериментальная состоятельность пока не доказаны.

## Навигация

- **[Спринты](sprints/README.md)** — недельные планы и итоги; правила работы — [AGENTS.md](AGENTS.md).

- [Синтез литературного обзора](docs/literature_review.md) — что известно, что спорно и что это меняет в проекте.
- [Ближайшие работы и границы новизны](docs/novelty.md).
- [Определения, формулы и диагностические признаки](docs/formalization.md).
- [План первого эксперимента](docs/experimental_protocol.md).
- [Результаты выполненного синтетического контроля](docs/controlled_probe_results.md) и [Adam timestep controls](docs/adam_reset_controls.md).
- [Порядок чтения и следующие шаги](docs/reading_plan.md).
- [Реестр источников](bibliography/README.md), [JSON](bibliography/papers.json), [BibTeX](bibliography/references.bib).
- Подробные заметки: [параметры](notes/parameter_plasticity.md), [optimizer и targets](notes/optimizer_targets.md), [данные и selection](notes/data_and_selection.md).
- [Дополнение по PPO и сравнению interventions](notes/additional_onpolicy.md).
- Аудит авторского кода: [параметры и resets](notes/code_audit_parameters.md), [optimizer и теория](notes/code_audit_optimizer.md), [Plasticine и повторный поиск](notes/infrastructure_and_forward_search.md).
- [Журнал поиска](research/search_log.md) и [решения](research/decisions.md).
- [Исполняемая исследовательская программа](docs/research_program.md), [конфигурация](configs/research_program.json) и [runner](experiments/rl_runner.py).
- [Схемы данных и manifest](schemas/README.md).
- [Совместимость native runtime](docs/native_runtime_compatibility.md).

## Статус на 7 октября 2026

Тематический обзор: 41 уникальный источник, включая актуальные препринты 2026 года. Второй проход добавил TeLAPA, PAME и SBP/P3O, уточнил границы новизны и проверил несколько авторских code paths с pinned commits. Это старт исследовательской базы, не исчерпывающий systematic review. Проверка полного текста означает чтение релевантных разделов, а не воспроизведение результатов; уровень проверки указан для каждой работы.

Выполнен supervised control: 24 checkpoint, 288 веток. Выведен и проверен LR/epsilon control для Adam timestep reset; наблюдены разные repair responses на искусственных задачах. [Отчёт с ограничениями](docs/controlled_probe_results.md).

Добавлен собственный CPU DQN с полным checkpoint, независимыми RNG streams, четырьмя основными repairs, cached-target probes и групповым анализом. Native MinAtar 1.0.15 проверен во всех пяти играх: по 500 совпадающих переходов после клонирования, включая sticky actions. Wheel SHA и версии сохранены в [проверке окружения](research/results/minatar_environment_validation.json).

Smoke обучения завершён: 2 000 transitions, 1 937 optimizer updates за 15.26 секунды, около 127 updates/s на полном training loop. Это проверка инфраструктуры, не результат о деградации обучаемости. Длинные серии и gates отражаются в живом отчёте `research/results/minatar-repair-20261007/REPORT.md`; незавершённые серии не объявляются экспериментальным успехом.

В директории уже был пустой локальный Git-репозиторий. Источники и документы сохранены локально; репозиторий опубликован: https://github.com/muitiiifruckt/trainability-repair-thesis (ветка main = codex/trainability-review на момент публикации 2026-10-09). Большие артефакты (checkpoints, replay, runs/) в Git не входят.

Рабочая ветка: `codex/trainability-review`. Author-code snapshots в `research/external/` исключены из Git; их commits и первичные ссылки записаны в audit notes. Малый synthetic результат сохранён вместе с runner/config/hash.

## Как пополнять базу

1. Найти первичную публикацию и проверить название, авторов, год, версию, venue.
2. Записать, что реально проверено: abstract, отдельные разделы PDF или авторский код.
3. Разделить результат авторов, собственную интерпретацию и гипотезу нашего проекта.
4. Добавить карточку в соответствующий `bibliography/*.json`, затем выполнить `python scripts/build_bibliography.py`.
5. При изменении постановки обновлять обзор, протокол и журнал решений вместе.

Большие checkpoints, replay buffers и результаты запусков в Git не включать. Для них хранить конфигурацию, seed, происхождение и контрольную сумму.

## Запуски и возобновление

Используется проектное `.venv` с native MinAtar и доступным PyTorch. Точные эффективные версии фиксируются в manifest каждого запуска.

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe -m experiments.rl_runner auto --resume
```

Можно отдельно выполнить `smoke`, `screen`, `mechanisms`, `confirm`, `synthetic`, `select`, `report`. `--resume` сохраняет прежние seeds и пропускает завершённые jobs. Конфигурация существующей кампании неизменна; для другой конфигурации нужен новый `output_root`.

Живое состояние: `runs/minatar-repair-20261007/progress.json`, очередь и контрольные суммы: `manifest.json`. Полные checkpoints/replay остаются в `runs/`; компактные исходы, графики и отчёты экспортируются в Git после серий. Максимум пакета — 12M updates или 24 часа; после границы runner сохраняет отчёт и начинает следующий пакет. Reserved-game обучение разрешается только после development signal и freeze протокола.
