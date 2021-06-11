# CHKG_proj

Кратко что было сделано:
* Сграблены данные сайта.
* Полученные данные отфильтрованы и приведены к удобному виду.
* Fine-tuning модели на сберовской gpt2
* Hyper-parameter tuning
* Генерация вопросов
Примеры вопросов можно посмотреть в “questions_genered_cosine_no_decay_1_word.txt” и “questions_genered_cosine_no_decay_5_words.txt”. Сразу бросается в глаза то, что важно не только обучить модель, но и грамотно выбрать метод генерации.

Текущий самый лучший результат по метрикам:
https://wandb.ai/falca/hse_dl_project/runs/ca4av3wz?workspace=user-falca

Проблемы, которые мы имеем на данный момент: 
1. Лосс на валидации растет вместе с метриками accuracy
2. Отсутствие логического окончания вопросов.
3. Грязный и бедный датасет.
4. Детерминированный способ генерации.

Как планируется решать проблемы:
1. Обсудить с Денисом
2. Есть идеи, но нужно изучить варианты, предлагаемые [hf] (https://huggingface.co/blog/how-to-generate).
3. Очистка датасета.
4. Реализация статистического способа генерации.

Также планируется задуматься над исследованием вопроса получения качественных контекстов для генерации. 


## Questions generation
In order to generate questions there is `generate.py` CLI script. From the project root run:
```
PYTHONPATH=. python src/generation/generate.py --model_dir %directory_with_model% --beam_size %beam_size% --max_len %max_seq_len_to_generate% --context %context_to_generate_from%
```
For now generation is pretty simple and has only 2 parameteres to tune -- `beam_size` and `max_len`.

## Trained models
* [Const lr model](https://hse-dl-models.s3.eu-central-1.amazonaws.com/model_const.tar.gz)
* [Cosine lr no decay model](https://hse-dl-models.s3.eu-central-1.amazonaws.com/cosine_no_decay.tar.gz)

