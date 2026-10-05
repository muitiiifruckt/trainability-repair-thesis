# Журнал литературного поиска

## 2026-10-05 — первый тематический проход

**Исследовательский вопрос:** can pre-intervention diagnostics predict which trainability repair benefits a deep RL checkpoint, and transfer that decision to unseen environments?

**Область:** преимущественно single-agent deep RL, loss of plasticity, optimizer nonstationarity, data/target feedback. Supervised continual learning включено только как источник методов/теории или близкий prior для learned control. Медицинская neuroplasticity и биологический repair исключены.

**Источники:** первичные papers через arXiv, PMLR, NeurIPS/ICLR proceedings, OpenReview, Nature; авторские code/project pages. Survey использован для discovery и taxonomy. Aggregators/search snippets не использованы как основание новых mechanistic claims; первичный текст проверялся отдельно.

**Метод:** тематические запросы, переход к cited foundational papers и точечный novelty search по сочетанию diagnosis, intervention, prediction, selection. Три параллельные ветки: parameters; optimizer/targets; occupancy/adaptive control. Затем общий синтез и независимая проверка formalization/protocol.

Примеры фактически выполненных root queries:

- `"loss of plasticity" reinforcement learning diagnostic 2025 2026`
- `"plasticity" "reinforcement learning" "optimizer" 2025`
- `"plasticity" "repair" "reinforcement learning" diagnostic`
- `"Plasticine" "plasticity" reinforcement learning 2504.17490`
- `"plasticity" reinforcement learning "diagnostic" "selection"`
- `"plasticity" reinforcement learning "adaptive" reset 2026`
- `"plasticity" reinforcement learning "intervention" "predict"`
- `"plasticity" "reinforcement learning" "selector"`
- `"plasticity" "reset" "meta" reinforcement learning 2026`
- `"plasticity" "intervention selection"`
- `"plasticity" "algorithm selection" reinforcement learning` с фильтром primary domains
- `"plasticity" "repair" "unseen" reinforcement learning` с фильтром primary domains
- `"plasticity loss" "predict" "interventions"` с фильтром primary domains
- `"plasticity" "diagnosis" "selector"` с фильтром primary domains
- `"Data, Auxiliary Losses, or Normalization Layers for Plasticity"`
- `"plasticity" "predicting" "reinforcement learning" site:arxiv.org`

**Inclusion:** directly informs operational plasticity, repair mechanisms/implementation, diagnostics-before-intervention, feedback controls, transfer/evaluation or close novelty. Published and preprint status separated in venue field. Abstract-only leads included but explicitly marked.

**Exclusion:** biological/clinical plasticity, generic evolutionary plasticity unrelated to adaptation of trained deep RL, third-party summaries without primary verification, papers whose relevance could not be supported.

## Важные найденные пересечения

Fixed-budget формализация, optimizer/weight ablation, clone-and-intervene diagnosis и adaptive intervention существовали до этого проекта. Новый кандидат — конкретная prospective selection/effect-prediction задача, а не общий лозунг diagnostic repair.

Также найдены библиографические version caveats: survey v3 меняет состав авторов; Plasticine v2 меняет число metrics; AdaLin и weights-vs-units имеют более поздний proceedings year; ReSiN изменил название; DreamerV3 manuscript/code variants различаются.

Поздний novelty search добавил Moalla2024 и Pyatko2025. Для Pyatko релевантные sections проверены через индексированные первичные passages; direct PDF показывает verification challenge. Код/ошибочная arXiv-ссылка со страницы автора не использованы как подтверждённые identifiers. Это ограничение сохранено в карточке.

## Что данный проход не покрывает

Нет полного database export, PRISMA counts, systematic forward-citation audit или formal quality scoring. Нет replication. Не прочитаны все приложения и proofs каждой работы. Поисковая система не гарантирует индексирование всех свежих publications.

Следовательно, фраза «точный аналог не найден» означает результат этого набора запросов и проверенных papers. Она не равна доказанной новизне или утверждению, что таких работ нет.

## Очередь дополнительных leads

- [InterpLayers / architectural design, ICML 2026](https://proceedings.mlr.press/v306/koeppe26a.html) — complementary architectural interventions; пока lead, не подробная evidence card.
- [Neuroplastic Expansion, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/e094d1e30e88949ed466067aef6be546-Abstract-Conference.html) — growth/pruning as adaptive control; полный текст/код ещё не аудирован.
- [On Predicting the Post-training Potential of Pre-trained LLMs, May 2026](https://arxiv.org/abs/2605.11978) — adjacent prospective trainability prediction; область LLM post-training, не наш checkpoint repair menu. Пока discovery lead.
- Полный текст SWD и первичная publisher metadata ASlib/Agarwal перед final thesis bibliography.
- [Continual Reinforcement Learning with Neuroevolution, 1 Oct 2026](https://arxiv.org/abs/2610.01583) — свежий lead о переносимости gradient-plasticity symptoms на другой learner; abstract проверен, в canonical bibliography пока не включён.

Число canonical sources и verification levels печатает `scripts/build_bibliography.py`; discovery leads из очереди не считаются изученными работами.

## 2026-10-05 — второй целевой проход

24 дополнительных novelty queries с filters и ограничениями записаны в [отдельном журнале](../notes/infrastructure_and_forward_search.md). Добавлены TeLAPA/PAME/SBP-P3O; это targeted search, не citation-graph export. Проверены приложения Injection, Asadi/AdamRel, assumptions SWD/AdamO и pinned author code Primacy/Juliani/OPEN/Plasticine/SWD. Код AdamRel expired, Injection exact code не найден. Полные paths, SHA и permalink evidence — в audit notes.

После этого пересмотрены menu semantics, probing baseline, costs и novelty wording. SWD full-text lead закрыт для релевантных theorem/proof sections; окончательные publisher BibTeX ASlib/Agarwal ещё в очереди. Собственный synthetic control выполнен отдельно; авторские экспериментальные результаты не реплицировались.
