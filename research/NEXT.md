# NEXT — читать первым в каждой сессии

Обновлено: 2026-10-08. Практики: [docs/agent_research_practices.md](../docs/agent_research_practices.md).

## Состояние
- Кампания `minatar-repair-20261007`: online screen идёт (supervisor `scripts/research_supervisor.py --workers 2`, перезапущен 2026-10-08 после ~22 ч простоя).
- Живые данные: `runs/minatar-repair-20261007/{progress.json,outcomes.jsonl,supervisor.json}`. Конфигурацию не менять.
- Завершено 32 из 96 основных веток (2026-10-09: Breakout seeds 0,1 x возрасты 50k/200k); живой отчёт — `research/results/minatar-repair-20261007/REPORT.md`.

## Внимание: сон машины
Кампания приостанавливается, когда Windows уходит в сон (ночью 9–10 ч); это не сбой, возобновление корректно (E1–E4 в `research/debate/log.md`). Сон от сети отключён, значит усыпление ручное. Не менять питание без решения пользователя.

## Порядок действий при возвращении
1. Проверить `supervisor.json` (status, restarts) и что растёт `package_updates`. Если процессов Python нет — перезапустить supervisor.
2. Пересчитать `scripts/power_from_partial.py`; при новых полных историях обновить `notes/parallel_vectors_20261008.md`.
3. Только содержательные изменения — в REPORT и снимок monitoring; коммит.

## Реестр гипотез (статус)
| id | гипотеза | статус | чем проверяется |
|---|---|---|---|
| H-int | head×optimizer взаимодействие отрицательно | не реплицировано на 2-й истории (−0.05 против −0.56; n=2, неопределённо) | contrast по всем историям, затем fixed-TD |
| H-split | вред joint reset — моменты Adam head vs body | proposed, не запущено | development probe с раздельным reset |
| H-age | знак head reset зависит от возраста | proposed | age×repair в анализе |
| H-cost | immediate drop head reset учитывать в utility selector | proposed | immediate/recovery декомпозиция |

## Дебат A/B (research/debate/)
- Раунд 1 завершён (A и B); см. `research/debate/log.md`, `accepted.md`, `rejected.md`, решения — `research/decisions.md` (2026-10-09).
- Раунд 2: A пишет prereg диагностик + dry-run процедуры (`scripts/debate_prereg_dryrun.py`); B остановлен пользователем, не перезапущен.
- Патч whitelist (`research/debate/patch_whitelist.diff`) отложен до конца кампании (SHA в dependency_manifest).

- Проект следующей кампании (не запущен): [docs/next_campaign_proposal.md](../docs/next_campaign_proposal.md).
- Прогноз: после screen решить, нужна ли репликация (n>=24 историй, 4 repeats, >=100 eval-эпизодов); prereg v3 финализирует A.

## Контекст агентов дебата
Координатор измеряет контекст субагентов по полю `usage` в их транскрипте (без чтения транскрипта в основной контекст): input + cache_read + cache_creation последнего сообщения.
Замер 2026-10-09: A ~190k токенов (близко к пределу), у B2 транскрипт недоступен (0 байт). Правило: агенты ведут `research/debate/handoff_A.md` / `handoff_B2.md`;
при контексте ~150k+ агент дописывает handoff, возвращает отчёт и завершается, координатор запускает преемника с чистым контекстом по handoff-файлу.

## Правила
- Числа в выводах только из файлов с SHA; reserved-игры закрыты до freeze; повторы одной истории не независимы.
