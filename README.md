# CHKG_proj
## Questions generation
In order to generate questions there is `generate.py` CLI script. From the project root run:
```
PYTHONPATH=. python src/generation/generate.py --model_dir %directory_with_model% --beam_size %beam_size% --max_len %max_seq_len_to_generate% --context %context_to_generate_from%
```
For now generation is pretty simple and has only 2 parameteres to tune -- `beam_size` and `max_len`.

## Trained models
[Const lr model](https://hse-dl-models.s3.eu-central-1.amazonaws.com/model_const.tar.gz)
