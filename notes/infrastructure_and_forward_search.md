# Plasticine и повторный поиск ближайших предшественников

Дата проверки: 5 октября 2026. Это ограниченный targeted search и статическое чтение авторского кода; воспроизведение опубликованных результатов не проводилось.

## 1. Plasticine v2: пригодность как экспериментальной основы

[Актуальный текст v2](https://arxiv.org/html/2504.17490v2), 10 февраля 2026; [авторский репозиторий](https://github.com/RLE-Foundation/Plasticine). Paper предлагает 13+ методов, шесть диагностик, C51/ALE и PPO/continual Procgen/DMC. Это удобный каталог реализаций; наш протокол checkpoint forks, ремонта и выбора действия требует дополнительной инфраструктуры.

Исходники скачаны **только для чтения** в `research/external/data/plasticine/`. Не запускались training scripts, install scripts, imports из этого репозитория или его зависимости. Проверенный `main`:

- commit `aa00b4bb18f7fe298a47e1ce36c32ba55ce064e8`;
- commit time `2026-02-09T17:20:59+08:00`;
- subject `Update README.md`;
- лицензия MIT;
- источник `https://github.com/RLE-Foundation/Plasticine.git`.

Для ссылок ниже используется именно этот commit, а не движущийся `main`.

### Что есть и чего не найдено в текущем дереве

Три training entrypoint: `c51_atari_plasticine.py`, `ppo_continual_procgen_plasticine.py`, `ppo_continual_dmc_plasticine.py`; каждому соответствует `_base.py` с моделью/методами. Отдельно находятся `plasticine_envs/`, `plasticine_metrics/`, `buffers.py`, `trac.py`, `utils.py`, requirements для каждого benchmark.

В дереве нет MinAtar, CartPole или специального CPU toy baseline. Есть `cuda` flag и fallback на CPU; это выбор устройства, а не подтверждение скорости или совместимости окружений на Windows. C51 модель фиксирует `(4,84,84)` и Atari preprocessing, поэтому заменить только `env_id` на MinAtar недостаточно. [C51 model](https://github.com/RLE-Foundation/Plasticine/blob/aa00b4bb18f7fe298a47e1ce36c32ba55ce064e8/plasticine/c51_atari_base.py#L18), [device/env construction](https://github.com/RLE-Foundation/Plasticine/blob/aa00b4bb18f7fe298a47e1ce36c32ba55ce064e8/plasticine/c51_atari_plasticine.py#L190).

`save_model_state` клонирует только `named_parameters` в памяти для weight difference. Это не durable checkpoint: нет optimizer, target network, replay buffer, RNG, среды, scheduler и даже model buffers. У C51 есть флаг `save_model`, но поиск его использований не обнаружил сохранения. [Функция](https://github.com/RLE-Foundation/Plasticine/blob/aa00b4bb18f7fe298a47e1ce36c32ba55ce064e8/plasticine/utils.py#L140).

### Вмешательства и состояние optimizer

Самый существенный результат статического аудита: C51 `plasticine_reset_layers` **создаёт новые модули**, а optimizer создан один раз раньше и после reset не перепривязывается. Injection также создаёт новые параметры, без добавления их в optimizer. Следовательно, в этой последовательности новый head не будет обновляться optimizer. В `Injector.forward` original head сохраняет gradient flow, хотя описанный в paper PI блокирует его; `new_b` блокируется через `detach`. Это расхождение надо устранить до использования baseline. [Reset/PI](https://github.com/RLE-Foundation/Plasticine/blob/aa00b4bb18f7fe298a47e1ce36c32ba55ce064e8/plasticine/c51_atari_base.py#L161), [optimizer initialization](https://github.com/RLE-Foundation/Plasticine/blob/aa00b4bb18f7fe298a47e1ce36c32ba55ce064e8/plasticine/c51_atari_plasticine.py#L200), [repair calls](https://github.com/RLE-Foundation/Plasticine/blob/aa00b4bb18f7fe298a47e1ce36c32ba55ce064e8/plasticine/c51_atari_plasticine.py#L333).

Аналогичный pattern замены actor/critic и однократного создания optimizer присутствует в проверенных PPO файлах. [DMC reset](https://github.com/RLE-Foundation/Plasticine/blob/aa00b4bb18f7fe298a47e1ce36c32ba55ce064e8/plasticine/ppo_continual_dmc_base.py#L156), [DMC optimizer](https://github.com/RLE-Foundation/Plasticine/blob/aa00b4bb18f7fe298a47e1ce36c32ba55ce064e8/plasticine/ppo_continual_dmc_plasticine.py#L170). Это вывод о коде данного commit, а не проверка экспериментальных результатов авторов.

SnP и ReDo меняют `.data` существующих параметров. В прочитанных реализациях не найдено обнуления соответствующих Adam moments или step counters. Значит, их следует трактовать как weights intervention с сохранённым optimizer state, пока явно не задан другой вариант. ReDo в таком виде нельзя автоматически считать чистым воспроизведением всех деталей оригинального метода. [C51 ReDo](https://github.com/RLE-Foundation/Plasticine/blob/aa00b4bb18f7fe298a47e1ce36c32ba55ce064e8/plasticine/c51_atari_base.py#L300).

### Диагностики: переносить с точным именем и определением

- `compute_stable_rank` возвращает число singular values, необходимое для 99% их суммы. Это spectral-mass rank, а не математический stable rank `||F||_F² / ||F||_2²`, и не 99%-variance PCA rank, который суммирует квадраты. `compute_effective_rank` использует entropy нормированных singular values. Обе функции требуют batch size ≥ feature width; полностью нулевая матрица не обработана отдельно. [rank.py](https://github.com/RLE-Foundation/Plasticine/blob/aa00b4bb18f7fe298a47e1ce36c32ba55ce064e8/plasticine_metrics/rank.py#L4).
- FAU для ReLU — доля ненулевых элементов batch×features. Это отличается от доли нейронов, активных хотя бы на одном примере. RDU вычисляется по усреднённой абсолютной активности, нормированной средней по слою, затем слои усредняются с одинаковым весом. [units.py](https://github.com/RLE-Foundation/Plasticine/blob/aa00b4bb18f7fe298a47e1ce36c32ba55ce064e8/plasticine_metrics/units.py#L8).
- Weight difference — RMS изменения параметров; это не абсолютная weight norm. C51 после изменения архитектуры ловит исключение этой метрики и пишет ноль: такой ноль не следует подавать selector как реальное отсутствие изменений. [norm.py](https://github.com/RLE-Foundation/Plasticine/blob/aa00b4bb18f7fe298a47e1ce36c32ba55ce064e8/plasticine_metrics/norm.py#L6), [metric logging](https://github.com/RLE-Foundation/Plasticine/blob/aa00b4bb18f7fe298a47e1ce36c32ba55ce064e8/plasticine/c51_atari_plasticine.py#L356).

Requirements разнородны и местами старые: ALE/Procgen закрепляют torch 1.12.1 и Python <3.11; DMC — torch 2.4.1. `kron_torch` импортируется entrypoints безусловно, но поиск `kron` в requirements не нашёл зависимости. Поэтому установка всего стека для CPU-пилота потребует отдельной проверки, а не копирования quickstart. [Requirements](https://github.com/RLE-Foundation/Plasticine/tree/aa00b4bb18f7fe298a47e1ce36c32ba55ce064e8/requirements).

## 2. Новые близкие работы

**TeLAPA — Lillo & Cheney, arXiv 2604.15414v2, 9 июня 2026.** [Текст](https://arxiv.org/html/2604.15414v2). Из policy archives извлекают diverse candidates и проверяют short adaptation каждого с reset optimizer. Выбор использует final score, slope и AUC; возвращаются исходные archived weights, не post-probe weights. Это уже дискретный выбор по наблюдаемой обучаемости. Learned embedder служит геометрии архива; response predictor heterogeneous repairs здесь не предъявлен (§3.2, Appendix D.1.1). [Ссылка кода из paper](https://anonymous.4open.science/r/telapa-map_elites-54E8); исходники по ней не удалось прочитать. Appendix I отдельно учитывает заметную стоимость archives/probing.

**PAME — Zang et al., AAMAS 2025.** [Официальный PDF](https://ifaamas.csc.liv.ac.uk/Proceedings/aamas2025/pdfs/p2299.pdf). Module-level FAU относительно раннего anchor и новизна опыта задают пороги selective injection. Проверяется Recover Nothing/Mixer/Agent/All; восстановление неподходящих модулей в эксперименте вредит. Это prior для «диагностика → какой модуль и когда восстановить», но mapping эвристический, intervention family одна (§6–7). Подтверждённая ссылка кода не найдена.

**SBP / P3O — Zhou et al., ICML 2025.** [Публикация](https://proceedings.mlr.press/v267/zhou25am.html). Cycle reset совмещён с inner distillation для восстановления знания; после reset дообучение продолжается до threshold потери. Проверенный Algorithm 1 задаёт расписание, не learned выбор разнородных repairs. Практический смысл для нас — действие «reset+recovery» имеет дополнительную, переменную стоимость, которую нельзя исключать из бюджета. Полнотекстовый алгоритм проверен через индекс [первичного PDF](https://openreview.net/pdf?id=hTrSxX3kiV).

### Уже известные работы: уточнение границы claim

[Injection Appendix B](https://arxiv.org/html/2305.15555) содержит weight-norm threshold для adaptive timing. [OnPolicyStudy Table 3/4](https://arxiv.org/html/2405.19153) уже использует GLM diagnostic metrics→train/test reward. [OPEN](https://papers.nips.cc/paper/2024/file/09e1944b7f2372f9f81866470c59b663-Paper-Conference.pdf) учит continuous updates из diagnostics; [author code](https://github.com/AlexGoldie/rl-learned-optimization) даёт pretrained optimizers и MinAtar configs в JAX. Ни один из этих проверенных протоколов не является обученным дискретным predictor `E[Y(a)|D]` для weights/optimizer/data/targets repairs одного исходного checkpoint. Уже найденные Adaptive RR, automatic soft reset и NeuMoSync дополнительно исключают широкий claim «впервые adaptive repair из диагностики».

## 3. Как проводился поиск и что он не доказывает

Поиск был направлен на дополнительные работы, использующие Injection/OnPolicyStudy/OPEN или похожую терминологию. Это **не полная выгрузка citation graph**: Google Scholar/Semantic Scholar цитирования не выгружались, систематический screening всех citing papers не выполнен. Верхняя дата включения — 2026-10-05; API search не поддерживает точный верхний date filter, поэтому даты найденных кандидатов проверялись в первичных метаданных. Источники итоговых утверждений — paper/proceedings/авторский код; обзорные агрегаторы использовались только как зацепки.

Точные наиболее релевантные запросы; domain filter указан для воспроизводимости:

1. `"plasticity injection" "adaptive" "2025" reinforcement learning` — без domain filter; найден PAME.
2. `"plasticity" "repair" "selector" reinforcement learning` — без фильтра.
3. `"OPEN" "learned optimizer" plasticity` — без фильтра.
4. `"on-policy" "plasticity" "intervention" "prediction"` — без фильтра.
5. `"plasticity injection" "selector" reinforcement learning` — без фильтра.
6. `"plasticity" "treatment response" reinforcement learning` — без фильтра; преимущественно биомедицинский шум, исключён.
7. `"plasticity" "repair selection" neural reinforcement learning` — arxiv.org / openreview.net / proceedings.mlr.press.
8. `"plasticity" "intervention selection" reinforcement learning` — arxiv.org / openreview.net.
9. `"plasticity" "diagnostics" "predict" "reset" reinforcement learning` — arxiv.org / openreview.net.
10. `"plasticity" "algorithm selection" reinforcement learning` — arxiv.org / openreview.net / proceedings.mlr.press.
11. `"Deep Reinforcement Learning with Plasticity Injection" "selector"` — arxiv.org / openreview.net / proceedings.mlr.press / proceedings.neurips.cc.
12. `"Deep Reinforcement Learning with Plasticity Injection" "adaptive" "2026"` — arxiv.org / openreview.net / proceedings.mlr.press.
13. `"A Study of Plasticity Loss in On-Policy" "diagnostic"` — arxiv.org / openreview.net.
14. `"Can Learned Optimization Make Reinforcement Learning Less Difficult" "selection"` — arxiv.org / openreview.net / proceedings.mlr.press.
15. `"reinforcement learning" "plasticity" "bandit" reset` — arxiv.org / openreview.net / proceedings.mlr.press.
16. `"reinforcement learning" "plasticity" "selector" -synaptic` — arxiv.org / openreview.net.
17. `"reinforcement learning" "adaptive reset" "plasticity"` — arxiv.org / openreview.net / proceedings.mlr.press; найден SBP/P3O.
18. `"neural network" "plasticity" "predicting" "intervention"` — arxiv.org / openreview.net.
19. `"A Study of Plasticity Loss in On-Policy" "adaptive"` — arxiv.org / openreview.net.
20. `"Can Learned Optimization Make Reinforcement Learning Less Difficult" "plasticity" "2026"` — arxiv.org / openreview.net.
21. `"plasticity" "pre-intervention" neural` — arxiv.org / openreview.net.
22. `"plasticity" "meta" "repair" "reinforcement learning"` — arxiv.org / openreview.net.
23. `"plasticity" "counterfactual" "reset" "reinforcement learning"` — arxiv.org / openreview.net.
24. `"plasticity" "predict" "interventions" "reinforcement learning"` — arxiv.org / openreview.net.

TeLAPA пришёл отдельной зацепкой от root, после чего проверены v2 §3 и Appendix D/I. Негативный результат запроса — лишь отсутствие подходящего найденного результата; он не доказывает отсутствие paper. Некоторые hits совпадали только в references, некоторые PDF закрывались verification challenge. Полный code audit OPEN/OnPolicyStudy делегирован другим агентам; здесь не утверждается проверка их runtime.

## 4. Минимальное практическое заимствование — наша рекомендация

Для первого controlled CPU toy я бы оставил собственный небольшой learner и checkpoint fork runner, заимствовав **точные определения нескольких диагностик** и структуру menu из Plasticine. Полный перенос ALE/Procgen/DMC стека сейчас добавит работу, не отвечающую главному feasibility вопросу.

Минимальные adoption gates:

1. Сохранение/восстановление полного learner state и воспроизводимый fork без shared mutable tensors.
2. После repair проверка совпадения идентичностей текущих trainable parameters и optimizer param groups. In-place reset удобен, когда специально нужен сохранённый optimizer.
3. Отдельные действия weights reset / optimizer reset / оба вместе: очищение моментов не должно происходить скрыто внутри weights repair.
4. Для PI — проверка сохранения outputs при вмешательстве, frozen original/static branches и обучения нового head.
5. Явные названия диагностик: `sv_mass_rank99`, `effective_rank`, `activation_nonzero_fraction`, `rdu`, `grad_norm`, `weight_rms_delta`. Проверить нулевую матрицу, малый batch и изменение архитектуры.
6. Диагностики на общей reference distribution и на текущем occupancy; к каждому vector сохранять provenance batch, targets, architecture и момент вычисления.
7. До сложного selector построить response matrix checkpoint×repair×seed. Проверить существование устойчивого преимущества выбора над лучшим постоянным действием. Реальные probing baselines считать с расходом общего бюджета, поскольку TeLAPA показывает именно такой соседний подход.

Рабочее отличие проекта после этого поиска: **amortized prediction различий в эффекте heterogeneous interventions по информации до вмешательства**, проверяемое regret и gain на отложенных task families. «Диагностировать падение», «пробовать несколько методов», «измерять способность checkpoint дообучаться» или «выбирать адаптивный момент reset» отдельно такой новизны не дают.
