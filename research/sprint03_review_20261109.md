# Ревью перед запуском Sprint 03, 2026-11-09

Ревьюер: субагент `senior_review`, `gpt-6-astra`, high reasoning, по требованию пользователя использовать старшую модель для всех ревью. Проверены протоколы, код, тесты и исправления до научных исходов.

- S3/S4: inverse action mapping, перенос Adam, тесты физического распределения действий/roundtrip/update-equivariance одобрены. Исправлены fail-stop очереди, проверка wrapper/attempt/manifest/artifacts, пересчёт исходов из curve. S3 имеет отдельный comparability gate, paired-noise/sign/family CI; slower не равен доказанной потере обучаемости.
- S4: малый gate допустим без lab v2, поскольку predictive validity метрики не заявляется. Обе стратегии выбираются на repeats0–1 и оцениваются на2–3; нет heldout-max comparator. Operational stop не является доказательством нулевого эффекта.
- S5: одинаковый горизонт, frozen behavior после repair и общий первый batch; heldout сравнение двух train-selected choices принято. Эффект включает вызванные данными изменения targets/IS, не pure occupancy при fixed targets.
- Probe v3: follow-up на просмотренных histories, новые labels не создают независимых histories. Поправка trained−frozen не чистый причинный эффект тела. Добавлены seals результатов/артефактов и проверки при resume/summary.
- Содержательная смена условий требует согласования с руководителем; не разрешена этим внутренним ревью.
- Научные запуски допускаются только после clean freeze и независимого последовательного S6 overhead gate (mean и median<=5%, bit equality); неуспешный gate блокирует соответствующую серию. Первые timing-пилоты пересеклись с другим benchmark и не используются как разрешение запуска.

Верификация: 14 тестов `test_sprint03_live`, `test_causal_probe_v3`, `test_causal_two_state_v2` прошли. Реальные исходы новых серий при этом не просматривались.

Финальное уточнение S6: v3 benchmark проверяет отдельно trained и frozen; запуск возможен только при обоих PASS. Контроллер сверяет revision.files отчётов с полным freeze; старый PASS другого кода не переиспользуется. Частоты до исходов: live16, S5 30, v3 400 (L3 в нескольких точках сохраняются).
