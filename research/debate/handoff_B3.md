# Handoff B3 (Skeptic) -> преемник. Финальная редакция 2026-10-09 ~11:35 (координатор: пользователь сворачивает работу; новых запусков нет)

Читать порядок: `research/debate/PROTOCOL.md` -> `handoff_B2.md` -> этот файл -> `log.md`: итоговая запись `[B3]` (последняя) и записи `[A2]` (если появились) -> `handoff_A2.md`.

## Границы и инциденты
- Границы PROTOCOL.md и handoff_B2.md соблюдены: `experiments/`, `configs/`, `runs/`, `research/results/minatar-repair-20261007/` не менялись; ничего не коммитилось; на артефактах checkpoint 200k ничего не запускалось; симуляции numpy/sklearn, 1 поток, RAM ~35 МБ на процесс.
- Скрипт `scripts/debate_b3_gate_sim.py` импортирует `experiments.rl_analysis` ТОЛЬКО из копии `research/debate/tmp/pkg0` (SHA rl_analysis 76315e19…, совпадает с `experiments/rl_analysis.py` и с `runs/.../dependency_manifest.json`; в скрипте стоит assert на путь).
- По просьбе координатора фоновая очередь сценариев остановлена (убит цикл bash; допущен до конца только уже запущенный `age_level`). Очередь была: uniform_best, uniform_failed, het_tau0.25/0.5/1.0, и варианты 3 истории/игру x 4 repeat — НЕ запущены.

## Что проверено (файл / число)
1. Аудит `experiments/rl_analysis.py` (прочитан целиком) и `rl_runner.py` (`source`, `branch`, `screen`, `report`, `adaptive`, `selection`, `frozen_transfer`, `auto`), `configs/research_program.json`, `tests/test_rl_analysis.py`.
2. Утечки SBS нет: `_crossfit` (`rl_analysis.py:217`), `naive` (`:439`), `_bootstrap_crossfit` (`:288`, исключены все копии origin), `fit_selectors` (`:621,:631`, GroupKFold по trajectory_id); тест `tests/test_rl_analysis.py:77-88`.
3. Моя функция `gate_for` (scripts/debate_b3_gate_sim.py) совпадает с `analyze()` (режим `equiv`: gate, advantage_vs_sbs, ci95, parameter status — equal=True).
4. Симуляции gate (копия кода, синтетика, 6 историй/игру, 6 repeat, B=200): `research/debate/b3_gate_null_h6_r6.json` (150 прогонов: gate uncertain 150/150; parameter_repair_signal positive 5/150), `b3_gate_game_level_h6_r6.json` (100 прогонов, различие только между играми, без неоднородности checkpoint: gate heterogeneous 100/100, parameter positive 98/100, mean advantage Asterix +0.379, LB +0.224), `b3_gate_age_level_h6_r6.json` (100 прогонов, только различие возрастов: heterogeneous 26/100, uncertain 74/100; parameter positive 0/100), `b3_selector_game_level_h6_r6.json` (ВСЕГО 2 набора, не оценка: оба `no_demonstrated_increment_over_age_and_sbs`).
5. Реальные данные: `scripts/debate_b3_outcomes_table.py` -> `research/debate/b3_outcomes_table.json` (outcomes.jsonl 48 строк, sha256 a3ec538452c390f8…; Breakout, 3 истории x 2 возраста x 2 repeat x 4 arms; Asterix 0 строк). Реальный cross-fit на Breakout (`scripts/debate_b3_gate_sim.py real 1000`, только чтение): advantage vs SBS -0.054, effect_vs_continue оценённого победителя -0.246, ci95 не определён (3 истории в одной игре).
6. Аудит заявлений: все числа README «Статус», docs/controlled_probe_results.md, docs/research_program_completion_audit.md сверены с файлами (список расхождений и правок — в итоговой записи log.md).

## Что НЕ проверено
- Задача 1 (воспроизведение dry-run v3 A2): итоговых результатов A2 на 11:31 нет; есть только калибровочные `prereg_dryrun_v3_cal1_*.json` (y_flip/y_perm; `cal1_yflip_game`: null_hetero_coupled L2 информативно 5.0% [3.8; 6.5], L1 1.6%; useful L2 2.8%) и `scripts/debate_prereg_dryrun_v3.py` (в работе). Моё независимое воспроизведение НЕ начато.
- Не запущены: uniform_best, uniform_failed, het_tau* (мощность gate), варианты 3 истории/игру; второй этап `fit_selectors` на >2 наборах; не сверялся `rl_scheduler._execute_waves` (completion audit, пункт про wall budget).
- Поведение `Campaign.adaptive()` в живом объекте не запускалось (разобрана логика кода и gate на синтетике).

## Точные следующие шаги
1. Когда A2 выложит итог (`research/debate/prereg_dryrun_v3_*.json` без префикса cal/tmp, запись `## [A2]` в log.md и §9 prereg): воспроизвести своим кодом `scripts/debate_b2_v2_check.py` (режимы через env: B2_PRESPEC, B2_RIDGE, B2_RIDGE_D, B2_STAT=F|W, B2_NULL=flip, B2_RANKD, B2_RANKY; DGP A или B2): FP на null_hetero_coupled (ожидание <=3–5% с рангами D и y), мощность «информативно»/«полезно» на n=12/24 при 2 и 4 repeat, `j_pre` в блоке B, ворота negative (ложные negative <=0.10 -> negative на n=12 запрещён). Сравнить с калибровкой A2 (`cal1_yflip_game`: 5.0% на null_hetero_coupled L2 — выше номинала 2.5%).
2. Докрутить очередь (каждый запуск < 270 с, 1 поток, из корня репозитория, `PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`): `python scripts/debate_b3_gate_sim.py run uniform_best 150 200 6 6`; `run uniform_failed 40 200 6 6`; `run het_tau0.25 100 200 6 6` (и 0.5, 1.0); `run het_tau1.0 60 200 3 4`; `selector age_level 50 100 6 6`; `selector game_level 50 100 6 6`. Ожидание (гипотеза, не результат): uniform_best — `single_repair` практически не достигается (`len(unique)==1`, rl_analysis.py:356); uniform_failed — gate `incomplete` в 100%.
3. Когда в outcomes.jsonl появятся строки Asterix: `python scripts/debate_b3_outcomes_table.py` и `python scripts/debate_b2_reliability.py asterix`; пересчитать таблицу «можно/нельзя».
4. Предложение к проверке (не реализовано): вторичный gate с SBS по игре (и по игре x возраст) — показать на симуляции game_level, что advantage над ним ~0.

## Окружение (ловушки)
- `sleep` в foreground блокируется; ждать только через фоновые команды/Monitor. Запись в log.md — только через файл (Write в scratchpad + `cat >> log.md`) или quoted heredoc.
- Список процессов: PowerShell `Get-CimInstance Win32_Process`; `ps` в Git Bash не видит Windows-python.
- Временные файлы очереди: `C:\Users\aayza\AppData\Local\Temp\b3_run_gate.sh`, `b3_gate_out.txt`.
- Heredoc с Python внутри bash: обратные слеши и `\n` в строках ломаются — править скрипты через Edit, не через `open(...).write`.
