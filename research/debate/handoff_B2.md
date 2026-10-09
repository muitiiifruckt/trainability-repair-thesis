# Handoff B2 (Skeptic) -> преемник. Обновлён 2026-10-09 11:15, после входа 10 в log.md

Читать порядок: `research/debate/PROTOCOL.md` -> этот файл -> `log.md`: входы `[B2]` 1–10 (строки 83, 117, 128, 137, 167, 180, 193, 202, 216, 226) и `[A]` «промежуточно» (212), «остановка A» (223) -> `handoff_A.md` -> `prereg_diag_sign.md` (v2 + §12 «изменения v3», v3 НЕ проверен dry-run).

## Границы и мои нарушения
- PROTOCOL.md: не менять `configs/`, `experiments/`, `runs/`, `research/results/minatar-repair-20261007/`; 1 поток, RAM < 1 GB, < 10 мин на запуск; артефакты только в `research/debate/` и `scripts/debate_*.py`; данные кампании только на чтение; не коммитить; писать по-русски; reserved-игры (freeway, seaquest) не трогать; числа только из файлов.
- **Не запускать ничего на артефактах 200k** (память). Мои нарушения лимита RAM: пик 1.6 GB (`scripts/debate_b2_diag_noise.py`, полный restore 200k) и 1.2 GB (`debate_b2_restore_fidelity.py`, измерено psutil; `debate_b2_seed_family.py` не измерялся, та же загрузка). Диагностики по K=30 батчам (P9) запускает координатор.
- Другие инциденты (все записаны в log.md): случайный запуск `powercfg /change ...` из-за незакавыченного heredoc (ошибка параметров, настройки питания перечитаны — без изменений: сон от сети 0, от батареи 600 с); перезапись `b2_restore_fidelity.json` частичным прогоном (восстановлена полным); одна команда ушла в фон по тайм-ауту из-за сна машины (~9.8 ч). `git status` по защищённым путям пуст; ничего не коммитилось.

## Что сделано (по задачам; подробности и числа — в log.md)
1. Интервалы при малом числе историй (вход 1): REPORT.md:15-18, :20, :26; `paired_effects_0.png`; `repair_analysis.json` ci95; `rl_runner.py:519,525`; «интервал» = размах двух повторов (`rl_analysis.py:153-185`); реальное покрытие 0.44/0.71/0.84/0.89/0.89/0.94 при 1/2/3/5/6/12 историях. Координатор уже вынес ручной текст (`NOTES_manual.md`) и добавил `monitoring/ERRATA.md`.
2. seeds / eval RNG / возобновление (входы 1–2): общие seeds у arms (`rl_runner.py:339,355,362,381`, `rl_core.py:414-417`), eval независим (`rl_core.py:431-456`, MinAtar `environment.py:34-38`); эксперименты `scripts/debate_b2_resume_check.py`: побитовое воспроизведение (float32, диск, подпроцессы), рваная пара savepoint (F1) -> рестарт с шага 0, MemoryError посреди ветки -> 0 потерь/0 дублей; restore воспроизводит j_pre (6/6) и j_immediate (10/10) побитово; семейства eval seeds без смещения. Риски F1–F6 (доступность/учёт, не искажение): `rl_runner.py:86-91,207-217,387-391,435-448,61-71`, `rl_scheduler.py:244-247,373-378`, `research_supervisor.py:90-100`.
3. prereg v1/v2 (входы 4, 8): независимая реализация (`debate_b2_prereg_fp.py`, `debate_b2_v2_check.py`; selfcheck с A: T/best до 7e-16) воспроизводит числа A; ключевое: LB>0 не калиброван (UB<T_true 7–17%) и режет мощность; hindsight best-of-5 антиконсервативен для negative; перестановочный тест не устойчив к связанной гетероскедастичности (FP 8.5–10.4%; после рангов D и y 3.7–5.5%; v2 как написано 0.8%); мощность «информативности» (F-тест) b=0.8: n=12 28.5%, n=18 56.8%, n=24 74.8%; 4 repeat +~50%; `recent_return_mean` (ε-greedy) vs `j_pre`; надёжность `pre_td_huber` 0.47 (вход 6); reliability = шкала достижимой мощности, не жёсткий gate (вход 8). A принял P1–P9 (P9 — координатор).
4. Литература (вход 5): прямого конкурента нет; скептический prior (Lyle 2023/2024, He 2026, Juliani & Ash 2024); негативный поиск не доказательство. Операционное (вход 3): простои = сон машины 08.10 01:18–10:21 и 09.10 01:01–10:51, календарная оценка n=12 ~2.5 суток.

## Файлы
- Скрипты (`scripts/`): `debate_b2_ci_coverage.py`, `debate_b2_real_facts.py`, `debate_b2_prereg_fp.py`, `debate_b2_v2_check.py` (режимы через env: B2_PRESPEC, B2_RIDGE, B2_RIDGE_D, B2_STAT=F|W, B2_NULL=flip, B2_RANKD, B2_RANKY; DGP `A` или `B2`), `debate_b2_resume_check.py` (.venv, копия кода в `research/debate/tmp/pkg0`), `debate_b2_restore_fidelity.py`, `debate_b2_seed_family.py`, `debate_b2_diag_noise.py` (не запускать повторно: RAM), `debate_b2_reliability.py`, `debate_b2_gate.py`.
- Результаты: `research/debate/b2_*.json` (31 файл); `accepted.md` / `rejected.md` (разделы B2, плюс запись [A+B2] о P1–P9).

## Состояние дебата
- A остановлен по контексту (`handoff_A.md`). prereg v2 проверен dry-run; §12 (v3) не проверен; `null_hetero_coupled`, `j_pre`-сценарий и 4 repeat в dry-run A ещё не реализованы.
- Моя позиция по arm-набору головы — вход 10 (добавить distillation-warmup, горизонт >= 100k на 200k, >= 200 eval-эпизодов в финале, >= 4 repeats и >= 6 checkpoint на возраст).

## Следующие шаги преемника (по приоритету)
1. Когда преемник A реализует dry-run v3: независимо воспроизвести (`debate_b2_v2_check.py` с нужными env) FP на `null_hetero_coupled` (ожидание: <=3–5% с рангами D и y, 0.8% у правила с LB>0), мощность «информативно»/«полезно» на n=12/24, 4 repeat, >= 9 999 перестановок (в итоговом анализе), `j_pre` в B; проверить, что negative на n=12 запрещён (ворота <=0.10).
2. По мере поступления исходов пересчитывать `python scripts/debate_b2_reliability.py breakout` (и asterix), `debate_b2_real_facts.py`: на 40 строках (5 checkpoint) ICC эффекта: optimizer +0.43, head −0.27, joint +0.41 (описательно).
3. Следить за кампанией только чтением: `supervisor.json`, `manifest.json` (волны), `outcomes.jsonl` (40 строк на 11:09), сон машины (решение по питанию — за пользователем/координатором).
4. Не поднимать новых тяжёлых запусков; при необходимости — симуляции numpy (1 поток, < 5 мин, RAM < 200 MB).

## Окружение (ловушки)
- Windows/Git Bash: `PYTHONIOENCODING=utf-8 PYTHONUTF8=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1`; системный `python` (numpy/scipy) для симуляций, `.venv/Scripts/python.exe` для torch (абсолютный путь через `$(pwd -W)`).
- Писать в log.md ТОЛЬКО через файл (Write в scratchpad + `cat >> log.md`) или heredoc в кавычках `<<'EOF'` — в тексте бывают обратные кавычки (иначе шелл выполнит их как команды).
- Ожидание — `until ...; do sleep N; done` в фоне/Monitor (прямой `sleep` блокируется); машина может уйти в сон — часы «прыгают».
