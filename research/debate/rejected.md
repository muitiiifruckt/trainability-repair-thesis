# rejected (B)
- power_from_partial.py числа "histories_for_80pct_power" (2/7/62) как планировочные: единицы не независимы (1 история), нормальная вместо t, отношение шумных оценок. Причина: scripts/power_from_partial.py:14-24, проверка nct.
- diag_sign: AUC 0.972 pre_future_training_loss как "сигнал диагностики": признак содержит future labels (docs/controlled_probe_results.md, раздел Ограничения) — утечка.
# rejected (A)
- [A] Механизм A4 «свежий Adam => большие шаги head после reset» — опровергнут scripts/debate_mech_step.py (норма шага head при state-reset не больше).
- [A] diag_sign AUC 0.972 как сигнал — утечка (принято от B).
- [A] head_reset_sync как атрибуция вреда головы — sync делает target мусором; допустим только как контроль.
