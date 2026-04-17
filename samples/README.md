# Приложения

## Краткое описание скриптов

- [`train_model.py`](#обучение-модели) — обучение собираемой модели. 
- [`run_experiment.py`](#эксперимент) — обучение модели с записью результатов
  обучения в файл в формате csv. 
- [`show_dataset.py`](#вывод-набора-данных) — вывод случайных 25 изображений
  набора данных с указанием классов.

## Обучение модели

> [!IMPORTANT]
> Все скрипты запускаются из виртуального окружения,
> поэтому здесь и далее префикс `(.venv)` перед коммандами
> будет опущен в угоду читаемости.

```bash
python train_model.py -trd <train_dataset_folder> \
                      -ted <test_dataset_folder> \
                      -c <config_file> \
                      -lf <log_folder> \
                      -ln <log_name> \
                      -s <size> <size>
```

**Аргументы:**

- `train_dataset_folder` — директория с тренировочными изображениями в формате <XX_NameOfASight>,
  например `01_NizhnyNovgorodKremlin`. 
- `test_dataset_folder` — директория с тестовыми изображениями в формате <XX_NameOfASight>,
  например `01_NizhnyNovgorodKremlin`.
- `config_file` — конфигурационный файл модели в формате toml.

**Пример:**

```toml
[macro_parameters]
batch_size = 64
epochs = 2

[model]
name = "AlexNet"
end = 2
classifier = "./classifiers/alexnet_classifier.json"

[optimizer]
name = "SGD"
learning_rate = 0.001

[loss_function]
name = "CrossEntropyLoss"
```

- `log_folder` — директория для сохранения логов.
- `log_name` — имя файла логов без расширения (stem).
- `size` — ширина и высота изображений, подаваемых
  на вход нейросети.

## Эксперимент

```bash
python train_model.py -trd <train_dataset_folder>
                      -ted <test_dataset_folder>
                      -c <config_file>
                      -lf <log_folder>
                      -ln <log_name>
                      -s <size> <size>
                      -o <output_file>
```

**Аргументы:**

- `output_file` — csv-файл, в который будет записан результат эксперимента.
- остальные аргументы аналогичны скрипту [`train_model.py`](#тренировка-модели).

# Вывод набора данных

```bash
python show_dataset.py -d <dataset_folder>
                       -s <size> <size>
```

**Аргументы:**

- `dataset_folder` — директория с изображениями в формате <XX_NameOfASight>,
  например `01_NizhnyNovgorodKremlin`.
- `size` — ширина и высота изображений, которые будут отображаться.
