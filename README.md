# Собственная CNN для обнаружения объектов

Проект содержит собственную свёрточную нейросеть для задачи object detection, отдельный скрипт обучения, отдельный скрипт тестирования, выгрузку метрик в CSV и графики TensorBoard. Для обучения модель инициализируется с нуля.

> **Архитектурное ограничение:** это самостоятельно реализованный учебный grid-детектор, а не готовая модель промышленного уровня. Он не использует YOLO, ResNet, Faster R-CNN, `torchvision.models`, `torchvision.ops`, `torchvision` или предобученные веса.

## 1. Структура проекта

```text
cnn_detector_project/
├── .gitignore
├── README.md
├── requirements.txt
├── detector_common.py
├── train_detector.py
├── evaluate_detector.py
├── download_plates_dataset.py
├── download_plates_dataset.ps1
├── Datasets/                      # загружаемые данные, не коммитятся в Git
│   ├── Cars/                      # данные для object detection
│   │   ├── images/
│   │   │   ├── train/
│   │   │   └── test/
│   │   └── labels/
│   │       ├── train/
│   │       └── test/
│   └── Plates/
│       ├── classes.names          # классы модели; создаётся/проверяется отдельно
│       └── raw/                   # файлы с Kaggle, исходная структура
├── outputs/                       # веса и CSV-метрики; не коммитятся
└── runs/                          # журналы TensorBoard; не коммитятся
```

Папки `Datasets`, `outputs`, `runs`, виртуальное окружение, `__pycache__`, веса модели и события TensorBoard исключены из Git через `.gitignore`.

## 2. Требования

- Python 3.11 рекомендуется, в частности для актуального Kaggle CLI.
- PyTorch.
- Pillow — чтение и изменение размера изображений.
- TensorBoard — графики обучения.
- Kaggle CLI — только для скачивания набора данных.

В коде модели нет зависимости от `torchvision`. Для GPU-обучения выберите подходящую сборку PyTorch на [официальной странице установки](https://pytorch.org/get-started/locally/). Если установщик предлагает отдельную команду для вашей версии CUDA, выполните её в созданном виртуальном окружении; остальные библиотеки ставятся из `requirements.txt`.

## 3. Создание `.venv` и установка библиотек

Все команды ниже выполняются из корня проекта — каталога, в котором находится этот `README.md`.

### Windows PowerShell

```powershell
# При наличии Python 3.11:
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Если PowerShell запрещает активацию скриптов, не обязательно менять Execution Policy: используйте интерпретатор окружения напрямую:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Linux / macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Проверка PyTorch и доступности GPU

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Device:', 'cuda' if torch.cuda.is_available() else 'cpu')"
```

Если GPU есть, но `CUDA available: False`, проверьте сборку PyTorch, драйвер NVIDIA и совместимость с CUDA. Следуйте инструкциям на официальном сайте PyTorch, а не устанавливайте `torchvision`: он этому проекту не нужен.

## 4. Скачивание датасета с Kaggle

Набор: [Russian License Plate Characters — 23 Classes](https://www.kaggle.com/datasets/dimahkiin/russian-license-plate-characters-23-classes).

Для скачивания через Kaggle API нужна авторизация:

1. Войдите в свою учётную запись на Kaggle и откройте настройки аккаунта, раздел API.
2. Создайте API token и получите `kaggle.json`.
3. Поместите файл вне репозитория, по стандартному пути Windows `%USERPROFILE%\.kaggle\kaggle.json` или Linux/macOS `~/.kaggle/kaggle.json`.
4. Не отправляйте этот файл другим людям и не добавляйте его в Git: в нём находятся учётные данные API.

Официальная справка: [Kaggle CLI — datasets](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets.md).

Установите зависимости (включая `kaggle`), затем выполните из корня проекта:

```powershell
python download_plates_dataset.py
```

Или в Windows можно использовать PowerShell-обёртку:

```powershell
.\download_plates_dataset.ps1
```

По умолчанию файлы будут скачаны и распакованы в `Datasets/Plates/raw`. Скрипт сначала скачивает данные во временную папку и переносит их в конечный каталог только после успешной распаковки. Если папка назначения уже содержит данные, скрипт остановится, не изменяя их. Для намеренной замены укажи `--force`:

```powershell
python download_plates_dataset.py --force
```

Можно задать другой каталог:

```powershell
python download_plates_dataset.py --output-dir Datasets/Plates/raw
```

Скрипт сохраняет опубликованную структуру датасета. Он **не переписывает** `classes.names`, не создаёт искусственные Bounding Box и не конвертирует разметку автоматически.

### Важно: датасет символов — не то же самое, что датасет обнаружения объектов

Текущие `train_detector.py` и `evaluate_detector.py` обучаются на изображениях целых сцен и размеченных рамках из `Datasets/Cars`. Данные ожидаются по путям:

- `Datasets/Cars/images/train` и `Datasets/Cars/labels/train` — обучение и последующее разделение на train/validation;
- `Datasets/Cars/images/test` и `Datasets/Cars/labels/test` — независимое итоговое тестирование;
- `Datasets/Plates/classes.names` — список классов в порядке идентификаторов `class_id`.

Каждая пара изображение/разметка должна совпадать по относительному пути и имени файла, например `images/train/frame001.jpg` и `labels/train/frame001.txt`. Разметка каждой непустой строки имеет формат:

```text
class_id x_center y_center width height
```

Координаты нормализованы относительно размеров исходного изображения в диапазоне 0–1. `class_id` начинается с нуля и соответствует строке в `classes.names`. Пустой `.txt` обозначает изображение без объектов.

Набор Kaggle с 23 классами символов может оказаться набором **обрезанных отдельных символов, разложенных по классам**, а не набором полноценных изображений с Bounding Box. Такой набор полезен для задачи классификации символов, но сам по себе не заменяет разметку рамок на изображениях целых номеров или автомобилей. Перед включением его в детектор проверьте структуру файлов и тип разметки; нельзя просто переименовать папку в `images/train` и считать задачу обнаружения размеченной.

Если при обучении используются классы из `Datasets/Plates/classes.names`, убедитесь, что этот файл точно соответствует идентификаторам классов в текстовой разметке `Datasets/Cars/labels`. Названия и порядок классов нельзя менять между обучением и тестированием.

## 5. Архитектура собственной CNN

В `detector_common.py` реализован `CustomCNNDetector`. Он построен на базовых модулях `torch.nn`, в том числе собственных свёрточных блоках с `Conv2d`, `BatchNorm2d` и `SiLU`. Веса инициализируются заново.

Сеть уменьшает пространственное разрешение входа в 8 раз. При `INPUT_SIZE = 320` её выходная сетка имеет размер 40×40. Для каждой ячейки предсказывается три слота; каждый слот содержит objectness, положение центра рамки, ширину и высоту, а также оценки классов. Потери для обнаружения, координат и классов, декодирование предсказаний, IoU, NMS и расчёт основных detection-метрик реализованы собственным кодом проекта на операциях PyTorch.

Это базовая архитектура для экспериментов. Ей может потребоваться настройка параметров и увеличение обучающего набора. В частности, несколько объектов с центрами внутри одной ячейки конкурируют за ограниченное число слотов.

## 6. Запуск обучения

После подготовки файлов изображений, разметки и `classes.names` запусти:

```powershell
python train_detector.py
```

Основные параметры расположены в начале `train_detector.py`: `EPOCHS`, `BATCH_SIZE`, `NUM_WORKERS`, `VALIDATION_RATIO`, `LEARNING_RATE`, `INPUT_SIZE` и `NUM_SLOTS`. По умолчанию часть данных из `images/train` выделяется под validation; каталог `images/test` во время обучения не используется.

Результаты сохраняются в `outputs/`:

- `best_detector.pth` — контрольная точка с лучшим validation mAP@0.5;
- `last_detector.pth` — контрольная точка последней завершённой эпохи;
- `training_history_<timestamp>.csv` — история потерь, метрик и времени по эпохам.

## 7. TensorBoard во время обучения

Запусти обучение в первом терминале. Во втором терминале, из корня проекта с активным `.venv`, запусти:

```powershell
tensorboard --logdir runs/custom_cnn_detector --host 127.0.0.1 --port 6006
```

Затем открой [http://localhost:6006](http://localhost:6006). Сервер TensorBoard и обучение работают независимо: остановка TensorBoard не останавливает тренировку. Графики записываются после завершения каждой эпохи, поэтому до первой эпохи панели могут быть пустыми.

Логируются train/validation loss, Precision, Recall, F1, mAP, IoU, скорость инференса, learning rate и время эпохи. Журналы каждого запуска находятся в отдельной подпапке `runs/custom_cnn_detector/`.

## 8. Независимое тестирование

Запускай тестирование после обучения, используя веса `best_detector.pth`:

```powershell
python evaluate_detector.py
```

Программа загрузит веса и проведёт оценку на `Datasets/Cars/images/test` с соответствующими файлами разметки из `Datasets/Cars/labels/test`. Веса при тестировании не обновляются. Результаты записываются в `outputs/test_metrics.csv`; повторные запуски добавляются строками, а при несовпадении схемы столбцов создаётся отдельный CSV.

В отчёт входят test loss, Precision, Recall, F1, mAP@0.5, mAP@0.5:0.95, средний IoU совпавших рамок, TP/FP/FN, AP по классам, количество параметров сети, время инференса модели, время постобработки и пропускная способность. Время инференса модели и время постобработки измеряются отдельно.

## 9. Типичный порядок работы

1. Создать `.venv` и установить `requirements.txt`.
2. Скачать Kaggle-датасет командой `python download_plates_dataset.py` и проверить его структуру.
3. Подготовить `Datasets/Cars` с изображениями и YOLO-совместимой разметкой рамок, а также правильный `Datasets/Plates/classes.names`.
4. Запустить `python train_detector.py`.
5. В другом терминале запустить TensorBoard и наблюдать графики.
6. После обучения запустить `python evaluate_detector.py` и изучить `outputs/test_metrics.csv`.

