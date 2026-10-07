# Native import compatibility: наблюдение 7 октября 2026

В текущем Windows окружении standalone `tests.test_rl_core` проходил, но общий discovery завершался без обычного Python traceback на первом core test после шести synthetic tests. Это был native process failure, не неуспешный assertion.

Через дочерний процесс с `-X faulthandler` получен return code `3221226356`, то есть `0xC0000374`. Stack проходил через поздний import `minatar.environment → seaborn → pandas → pyarrow`; pandas/pyarrow/seaborn загружались из user site. Отдельный минимальный процесс с Torch matrix multiplication до import MinAtar также вызвал native access violations. Конкретная причина повреждения памяти внутри бинарных библиотек не установлена; нельзя приписывать её определённой версии BLAS по одному stack trace.

Наблюдавшийся порядок для воспроизведения:

```powershell
.venv/Scripts/python.exe -X faulthandler -c "import torch; torch.set_num_threads(1); m=torch.nn.Linear(8,32).double(); y=m(torch.randn(8,8,dtype=torch.float64)); y.sum().backward(); from minatar import Environment"
```

Изначально это также воспроизводилось последовательностью synthetic tests, затем `CoreTests.test_evaluation_and_diagnostics_do_not_mutate_training_or_global_rng`. Без faulthandler shell сообщал только exit 1.

## Применённый обход

В `experiments/rl_core.py` dependency `Environment` теперь импортируется при загрузке модуля, до learner work. Исследовательский core не использует rendering, но upstream MinAtar загружает optional visualization dependencies при import. Удалены поздние imports из `create` и `evaluate`.

Исходники upstream MinAtar и установленные версии не изменены. Это совместимость наблюдаемого runtime через порядок initialization, не доказательство устранения underlying binary bug. Entry points должны импортировать core до numerical learner work; поздний import core после чужого Torch вычисления не является проверенным режимом.

## Проверенная регрессия

```powershell
.venv/Scripts/python.exe -X faulthandler -m unittest discover -s tests -v
```

После обхода фактически выполнены **29 tests за 6.016 секунды, OK**: сначала шесть synthetic tests, затем 13 core tests, затем десять fixed-TD mechanism tests. Этот порядок включает прежний failure trigger. Отдельный core suite ранее прошёл 13 tests; общий запуск существенно важнее для данного дефекта.

Проверенное окружение: Python 3.13.12, Torch 2.12.1+cu126, NumPy 2.4.4, MinAtar 1.0.15, pandas 3.0.2, pyarrow 24.0.0, seaborn 0.13.2. Используется project venv с system site packages; learner работает на CPU.

