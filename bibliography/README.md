# Источники и уровень проверки

`papers.json` — единый реестр, `references.bib` — рабочий экспорт. Источники разбиты по исследовательским веткам в `parameters.json`, `optimizer_targets.json`, `data_selection.json`, `additional.json`, `core.json` и `followup.json`; дубликаты объединяются по нормализованному названию.

Первый проход содержал 38 уникальных работ; второй добавил TeLAPA, PAME и SBP/P3O; forward-citation проход 2026-10-09 ([заметка](../notes/forward_search_20261009.md)) добавил 12 работ в `followup.json` (все `full_text` по разделам, перечисленным в карточках; приложения и доказательства, как правило, не читались; `verified_at` этих карточек — 2026-10-09). Текущий реестр: **53 работы, 48 full_text и 5 abstract**. Уровни проверки и последующее чтение теорем уточняются в карточках и audit notes. Authoritative число работ и verification levels печатает сборщик. `full_text` иногда означает доступ к релевантным индексированным passages первичного PDF; ограничение записано в карточке.

Пересборка: `python scripts/build_bibliography.py`.

## Как читать поля

- `verified_level: full_text` — проверены релевантные sections/appendices полного текста. Не означает проверки всех доказательств или репликации. Иногда детали проверены по авторскому препринту: смотреть `evidence_location` / `limitations`.
- `verified_level: abstract` — подтверждены metadata и abstract; detailed claims не использовать до чтения PDF.
- `year` — год cited publication/version. `first_release_year` или `first_publication_year`, если есть, фиксирует более ранний препринт; proceedings иногда выходят позже конференции.
- `mechanism`, `intervention`, `diagnostic` — краткая тематическая карточка, не независимое подтверждение claims.
- `limitations`, `relevance` — сочетание оговорок источника и нашей интерпретации для проекта. Подробное разделение — в notes.
- `source_files`, `aliases` — происхождение записи и альтернативные IDs из исследовательских веток.

Code URL — адрес авторской реализации, а не проверка её работы. Никакая карточка не означает, что результаты воспроизведены.

BibTeX использует `@misc` и `howpublished`, чтобы не выдумывать точные publisher fields. Перед включением в thesis заменить ключевые entries официальным BibTeX proceedings/journal и зафиксировать версии препринтов.

В документах claims снабжены прямыми ссылками. Обзорные и benchmark работы используются для ориентации/методологии; выводы о механизмах привязаны к первичным экспериментальным статьям.
