# Handoff A2 (преемник A) — ведётся с 2026-10-09 11:20

Статус: СТАРТ. Прочитано: PROTOCOL.md, handoff_A.md, handoff_B2.md, NEXT.md, log.md ([A], [Координатор], [B2] 1–10), accepted.md, rejected.md, prereg_diag_sign.md (v2 + §12), docs/next_campaign_proposal.md, scripts/debate_prereg_dryrun.py, debate_b2_v2_check.py, debate_b2_gate.py, debate_b2_reliability.py.

## Границы (кратко)
Не менять configs/, experiments/, runs/, research/results/minatar-repair-20261007/; 1 поток (OMP/OPENBLAS=1), RAM < 1 GB, < 10 мин на запуск; артефакты только в research/debate/ и scripts/debate_*.py; реальные outcomes/diagnostics — только чтение, анализ плана на реальных данных НЕ запускать; на артефактах checkpoint 200k ничего не запускать (P9 — решение координатора); не коммитить; по-русски; числа только из файлов.

## План работ (приоритет)
1. scripts/debate_prereg_dryrun_v3.py + research/debate/prereg_dryrun_v3_*.json: (a) j_pre/Q в блоке B; (b) уровни «информативно»/«полезно»; (c) null_hetero_coupled + выбор схемы нуля (FL со знаками на остатках y; ранги D и y); (d) компаратор заранее; (e) R=2/R=4; (f) reliability gate ICC(1)>=0.3 (доля прохождения при ICC 0/.15/.3/.5); (g) ворота negative (ложные negative <= 0.10 при b=0.8); (h) >= 9999 перестановок в финале (в dry-run 99–999/9999 на подвыборке); комбинаторика блочных перестановок при 3 историях на игру (p_min 0.028 против знаковых 1/64).
2. §9 prereg + §11 «Отклонения» (откат непрошедшего, с датой); заморозка — после согласия B2 (accepted.md).
3. Ответ B2 по пунктам P1–P8 с числами v3 -> log.md `## [A2] ...` + accepted.md.

## Журнал шагов (обновляется)
- 11:20 handoff_A2.md создан; следующая задача — написать scripts/debate_prereg_dryrun_v3.py (движок: линейные сглаживатели LOHO по фолдам, перестановки по y — матричным умножением; D-перестановки — батч solve).

## Открытое / риски
- Нет ни одной записи в log.md от A2 пока. Параллельный B2: id aac6942766d9d1760.
