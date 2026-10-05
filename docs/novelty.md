# Новизна: что уже сделано и где остаётся исследовательский вопрос

Проверено 2026-10-05, два прохода поиска. Это audit найденных работ, не доказательство отсутствия prior art. Все различия ниже — наше прочтение опубликованного дизайна. Приложения и ряд code paths уже проверены; полный citation-graph screening и replication не выполнены.

## Нельзя заявлять как новый вклад

- Fixed-budget измерение plasticity: [Lyle et al., 2023, eq. 5](https://proceedings.mlr.press/v202/lyle23b/lyle23b.pdf).
- Reset weights отдельно от optimizer: [Primacy Bias, 2022, Appendix B](https://proceedings.mlr.press/v162/nikishin22a/nikishin22a.pdf).
- Clone checkpoint и диагностическое вмешательство: [Plasticity Injection, 2023, §5.2](https://papers.neurips.cc/paper_files/paper/2023/file/75101364dc3aa7772d27528ea504472b-Paper-Conference.pdf).
- Любой repair, запускаемый diagnostic threshold: ReDo и adaptive injection timing уже это делают.
- Сравнение нескольких mitigations и нескольких metrics: [On-Policy Study, 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ce7984e36d58659211a8dc7d5457cd6f-Abstract-Conference.html).
- Широкое сравнение data/loss/normalization interventions в PPO: [Pyatko et al., 2025](https://openreview.net/pdf?id=6CViR7tKj2). Capacity probe со свежим optimizer также есть у [Moalla et al., 2024](https://arxiv.org/html/2405.00662v3).
- Несколько механизмов и комбинации методов: [Disentangling, 2024/2025](https://proceedings.mlr.press/v274/lyle25a.html).
- Adaptive training knob по plasticity proxy: [Adaptive RR, 2024](https://arxiv.org/abs/2310.07418).
- Learned plasticity control: [OPEN, 2024](https://papers.nips.cc/paper/2024/file/09e1944b7f2372f9f81866470c59b663-Paper-Conference.pdf), [Automatic Soft Reset, 2024](https://arxiv.org/abs/2411.04034), [NeuMoSync, 2026](https://arxiv.org/abs/2608.04358).
- Discrete выбор старта через короткое пробное обучение: [TeLAPA, 2026, Appendix D](https://arxiv.org/html/2604.15414v2). Diagnostic выбор модуля и времени injection: [PAME, 2025](https://ifaamas.csc.liv.ac.uk/Proceedings/aamas2025/pdfs/p2299.pdf).

## Кандидат на конкретный вклад

> По дешёвым разрешённым diagnostics предсказывать ожидаемую пользу каждого действия из небольшого доступного repair menu, заменяя пробное дообучение всех вариантов; обучаться на controlled checkpoint branches; оценивать return/regret и total cost на полностью отложенных RL environments.

Внутри этой формулировки четыре проверяемых различия:

1. **Prospective:** все features получены до main repair, а не из финального состояния.
2. **Относительная польза:** модель предсказывает effects относительно continue / альтернативных repairs, а не только «агент плохо обучается».
3. **Несколько типов actions:** достаточно совместимых parameter и optimizer actions в первом прототипе; data/targets расширять только при корректной операционализации.
4. **Transfer decision:** оценка на held-out средах с regret, uncertainty и сравнением с single best repair.

Если удастся только первое/второе в одной среде, это полезный preliminary result, но ещё не заявленный unseen-environments вклад. Если neural predictor не нужен и работает одно правило, это может быть сильнее инженерно.

Нужен прямой конкурент: **short-probe chooser** пробует каждое действие на копии checkpoint и выбирает по наблюдаемой short-budget response. В основном practical сравнении его probe cost вычитается из общего бюджета. У selector может быть отдельный offline training cost: публиковать его и break-even число решений, а не считать amortization бесплатной. Это наш repair-menu baseline по мотивам TeLAPA, не точная репликация policy-archive метода.

## Главные конкурирующие объяснения положительного результата

- Features просто кодируют checkpoint age, LR schedule или environment identity.
- Меню содержит один слабый/ненастроенный baseline и один сильный универсальный метод.
- Repair выигрывает только потому, что получает больше updates/данных.
- Post-repair features либо связанные checkpoints попали в train и test.
- Oracle выбран по шумному максимуму, поэтому его разрыв преувеличен.
- Test horizon выбран уже после просмотра winners.
- Controlled failures искусственно соответствуют names действий, а естественные checkpoints дают смешанные эффекты.

Каждая альтернатива превращается в baseline/control в [протоколе](experimental_protocol.md).

## Как не завысить причинный claim

Фраза «diagnostic определяет, что сломан optimizer» требует больше, чем успешный optimizer reset. Более защищённая формулировка: «diagnostic предсказывает положительный conditional effect optimizer reset в таком-то режиме и бюджете». Mechanistic evidence — отдельная часть: fixed-input/target controls, partial moment resets, update-size controls, factorial interaction и prediction-preserving arms.

Точная классификация четырёх взаимоисключающих причин пока не является обоснованной моделью. Weights/optimizer/data/targets лучше считать местами вмешательства в связанную систему.

## Незакрытый novelty audit

- Forward citations для Injection, Lyle2023/Disentangling, On-Policy Study и OPEN.
- Свежие proceedings/preprints 2026 про automated repair, meta-control и portfolios; поиск не считать исчерпывающим.
- Проверить, не существует ли prospective predictor response уже в коде/appendix ближайших работ, даже если abstract его не выделяет.
- Связать paper versions с code commits. Дата proceedings иногда отличается от года первой публикации.
- Перед thesis proposal перечитать таблицы/appendices ближайших работ и сформулировать отличие на уровне inputs, actions, labels и evaluation split.

Второй проход уже проверил Injection Appendix B, pinned Primacy/On-Policy/OPEN/Plasticine code и 24 дополнительных targeted queries. [Параметры](../notes/code_audit_parameters.md), [optimizer](../notes/code_audit_optimizer.md), [поиск и инфраструктура](../notes/infrastructure_and_forward_search.md). Negative search result не подтверждает отсутствие подходящей работы.
