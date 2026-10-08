# Параллельные векторы развития (2026-10-08, во время работы кампании)

Работа не использует CPU/RAM кампании (0 optimizer updates) и не меняет её конфигурацию.

## 1. Поиск литературы (WebSearch, standard, 3 запроса)
Новых 2026 работ по (a) prospective выбору repair по pre-repair diagnostics и (b) взаимодействию
head reset × Adam reset не найдено; находятся только Plasticity Injection (Nikishin 2023),
Resetting the Optimizer (Asadi 2023), SNR (ICLR 2025), Plasticine. Поиск неполон (US-only, поверхностный),
это не доказательство отсутствия работ; `docs/novelty.md` границ не меняет.

## 2. Планирование мощности (`scripts/power_from_partial.py`, результат в `research/results/parallel/`)
По 4 завершённым парам (Breakout seed 0, 2 возраста × 2 повтора; **одна история**, единицы — не независимы):
средние парные эффекты относительно continue: optimizer −0.51, head −0.41, joint −1.49.
Оценка sd шумовая. Вывод для планирования: при 3 историях на game (минимум протокола) ловится лишь крупный
эффект (joint-вред); разрешение эффекта head reset (знак меняется между возрастами, sd≈1.1) потребует
порядка десятков историй. Следовательно gate `uncertain` для малых эффектов ожидаем, а расширение
до source seeds 3,4,5 (шаг 5 протокола) вероятно понадобится заранее. Пересчитать после >=3 историй.

## 3. Гипотезы для проверки после завершения screen (не запущены)
- H-int: взаимодействие head×optimizer отрицательно (joint хуже суммы). Проверка: contrast по всем историям; затем fixed-TD tape.
- H-split: вред joint reset — из-за сброса Adam-моментов head или body. Требует нового разделённого reset (development probe, согласно research_program).
- H-age: знак head reset зависит от возраста (50k +0.4/+0.3, 200k −0.5/−2.0 на seed 0). Проверка: age×repair в анализе.
- H-cost: учитывать immediate-drop head reset как стоимость в selector utility.
