# accepted (B)
- [B] Не публиковать выводы H-int/H-age при 1 истории; intervals только как "неопределённо" (согласуется с NEXT.md).
- [B] Добавить arm head_reset_sync (target:=online после reset) в development-пробу H-split — предложено B, ждёт ответа A.
- [B] Заменить blacklist признаков в rl_analysis.py:473 на whitelist и исключить clock/replay_size признаки из "diagnostic" либо вынести в age-baseline (рекомендация координатору; код не менять сейчас).
- [A+B] Шумовой пол: SE разности final-eval 0.6-1.9 (scripts/debate_noise_floor.py); отдельные эффекты ~1-2 SE; больше eval-эпизодов в final точке (рекомендация следующему этапу, config не менять).
- [A] Конкурентный baseline: pre_td_huber + recent_return обязателен рядом с age_only в сравнении selector'ов (методическое; число из synthetic снято).
- [A] Экспериментальные arms для H-split: state:head, state:body, headw+state:head, headw+state:body; контроли stale/sync/injection для атрибуции вреда головы. Код: scripts/debate_split_adam.py.
