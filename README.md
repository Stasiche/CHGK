# CHKG_proj

## Data
Data is collected from (open database of CHGK questions)[https://db.chgk.info/]


Кратко что было сделано:
Примеры вопросов можно посмотреть в “questions_genered_cosine_no_decay_1_word.txt” и “questions_genered_cosine_no_decay_5_words.txt”. Сразу бросается в глаза то, что важно не только обучить модель, но и грамотно выбрать метод генерации.



## Telegram bot
* Natasha for domain recognition
* Finetuned ru-gpt3 for question generation
* RuBert + KMeans for theme detection

## Questions generation
In order to generate questions there is `generate.py` CLI script. From the project root run:
```
PYTHONPATH=. python src/generation/generate.py --model_dir %directory_with_model% --beam_size %beam_size% --max_len %max_seq_len_to_generate% --context %context_to_generate_from%
```
For now generation has only 2 parameters: 
    1. `beam_size` --- number of generated hypothesis
    2. `max_len` --- length of a generated question

## Trained models
### HTML data
* [Const lr model](https://hse-dl-models.s3.eu-central-1.amazonaws.com/model_const.tar.gz)
* [Cosine lr no decay model](https://hse-dl-models.s3.eu-central-1.amazonaws.com/cosine_no_decay.tar.gz)

### XML data
* [Cosine lr no decay model](https://hse-dl-models.s3.eu-central-1.amazonaws.com/new_data.tar.gz)
* [Blitz model](https://hse-dl-models.s3.eu-central-1.amazonaws.com/blitz_model.tar.gz)
* [Answers in context model](https://hse-dl-models.s3.eu-central-1.amazonaws.com/answers_100_20_model.tar.gz)

### Validation accuracy-top-5
Лучший [результат](https://wandb.ai/falca/hse_dl_project/runs/ca4av3wz?workspace=user-falca) по метрикам:

![Validation accuracy-top-5](pics/val_scores.png)
