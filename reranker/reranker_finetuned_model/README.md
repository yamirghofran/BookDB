---
tags:
- sentence-transformers
- cross-encoder
- generated_from_trainer
- dataset_size:222360
- loss:BinaryCrossEntropyLoss
base_model: cross-encoder/ms-marco-MiniLM-L6-v2
pipeline_tag: text-ranking
library_name: sentence-transformers
metrics:
- map
- mrr@10
- ndcg@10
model-index:
- name: CrossEncoder based on cross-encoder/ms-marco-MiniLM-L6-v2
  results:
  - task:
      type: cross-encoder-reranking
      name: Cross Encoder Reranking
    dataset:
      name: val rerank
      type: val_rerank
    metrics:
    - type: map
      value: 0.8095938412000587
      name: Map
    - type: mrr@10
      value: 0.936482235381199
      name: Mrr@10
    - type: ndcg@10
      value: 0.8829935625266973
      name: Ndcg@10
---

# CrossEncoder based on cross-encoder/ms-marco-MiniLM-L6-v2

This is a [Cross Encoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html) model finetuned from [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) using the [sentence-transformers](https://www.SBERT.net) library. It computes scores for pairs of texts, which can be used for text reranking and semantic search.

## Model Details

### Model Description
- **Model Type:** Cross Encoder
- **Base model:** [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) <!-- at revision ce0834f22110de6d9222af7a7a03628121708969 -->
- **Maximum Sequence Length:** 384 tokens
- **Number of Output Labels:** 1 label
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Documentation:** [Cross Encoder Documentation](https://www.sbert.net/docs/cross_encoder/usage/usage.html)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Cross Encoders on Hugging Face](https://huggingface.co/models?library=sentence-transformers&other=cross-encoder)

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import CrossEncoder

# Download from the 🤗 Hub
model = CrossEncoder("cross_encoder_model_id")
# Get scores for pairs of texts
pairs = [
    ['Favorite books: To Kill a Mockingbird by Harper Lee, A Common Life: The Wedding Story (Mitford Years, #6) by Jan Karon and The Hiding Place: The Triumphant True Story of Corrie Ten Boom by Corrie ten Boom and John Sherrill and Elizabeth Sherrill. Favorite genres: contemporary and drama.', "Title: A Tale of Two Cities | Genres: adventure, drama, historical, history, literature, romance, war | Description: 'Liberty, equality, fraternity, or death; -- the last, much the easiest to bestow, O Guillotine!'\nAfter eighteen years as a political prisoner in the Bastille, the ageing Doctor Manette is finally released and reunited with his daughter in England. There the lives of two very different men, Charles Darnay, an exiled French aristocrat, and Sydney Carton, a disreputable but brilliant English lawyer, become enmeshed through their love for Lucie Manette. From the tranquil roads of London, they are drawn against their will to the vengeful, bloodstained streets of Paris at the height of the Reign of Terror, and they soon fall under the lethal shadow of La Guillotine.\nThis edition uses the text as it appeared in its serial publication in 1859 to convey the full scope of Dickens's vision, and includes the original illustrations by H. K. Browne ('Phiz'). Richard Maxwell's introduction discusses the intricate interweaving of epic drama with personal tragedy.\n--back cover | Authors: Charles Dickens, Richard Maxwell, Hablot Knight Browne"],
    ['Favorite books: Alice in Zombieland (White Rabbit Chronicles, #1) by Gena Showalter, Animal Magnetism (Animal Magnetism, #1) by Jill Shalvis and Chitty Chitty Bang Bang (Chitty Chitty Bang Bang, #1) by Ian Fleming and Brian Selznick. Favorite genres: action and adventure.', 'Title: Resurrection Men (Inspector Rebus, #13) | Genres: art, contemporary, crime, horror, literature, mystery, noir, suspense, thriller, war | Description: Inspector John Rebus has messed up badly this time, so badly that he\'s been sent to a kind of reform school for damaged cops. While there among the last-chancers known as "resurrection men," he joins a covert mission to gain evidence of a drug heist orchestrated by three of his classmates. But the group has been assigned an unsolved murder that may have resulted from Rebus\'s own mistake. Now Rebus can\'t determine if he\'s been set up for a fall or if his disgraced classmates are as ruthless as he suspects. When Detective Sergeant Siobhan Clarke discovers that her investigation of an art dealer\'s murder is tied to Rebus\'s inquiry, the two--protege and mentor--join forces. Soon they find themselves in the midst of an even bigger scandal than they had imagined-a plot with conspirators in every corner of Scotland and deadly implications about their colleagues. With the brilliant eye for character and place that earned him the name "the Dickens of Edinburgh," Ian Rankin delivers a page-turning novel of intricate suspense. | Authors: Ian Rankin'],
    ['Favorite books: Point of Impact (Bob Lee Swagger, #1) by Stephen Hunter, Stalking the Angel (Elvis Cole, #2) by Robert Crais and The Brass Verdict (Harry Bosch, #14; Mickey Haller, #2; Harry Bosch Universe, #17) by Michael Connelly. Favorite genres: action and crime.', "Title: The Deep Blue Good-By (Travis McGee, #1) | Genres: action, adventure, crime, hardboiled, literature, mystery, noir, suspense, thriller | Description: TRAVIS McGEE\nHe's a self-described beach bum who won his houseboat in a card game. He's also a knight errant who's wary of credit cards, retirement benefits, political parties, mortgages, and television. He only works when his cash runs out and his rule is simple: he'll help you find whatever was taken from you, as long as he can keep half. | Authors: John D. MacDonald, Carl Hiaasen"],
    ['Favorite books: Red Prophet (Tales of Alvin Maker, #2) by Orson Scott Card, Through Gates of Splendor by Elisabeth Elliot and Sybil: The Classic True Story of a Woman Possessed by Sixteen Personalities by Flora Rheta Schreiber. Favorite genres: history and adventure.', "Title: The Judas Strain (Sigma Force, #4) | Genres: action, adventure, contemporary, crime, fantasy, historical, history, mystery, science, scifi, suspense, techno-thriller, thriller | Description: New York Times bestselling author James Rollins returns with a terrifying story of an ancient menace reborn to plague the modern world . . . and of an impossible hope that lies hidden in the most shocking place imaginable: within the language of angels.ju*das strain, n. A scientific term for an organism that drives an entire species to extinction.From the depths of the Indian Ocean, a horrific plague has arisen to devastate humankind--a disease that's unknown, unstoppable . . . and deadly. But it is merely a harbinger of the doom that is to follow. Aboard a cruise liner transformed into a makeshift hospital, Dr. Lisa Cummings and Monk Kokkalis--operatives of SIGMA Force--search for answers to the bizarre affliction. But there are others with far less altruistic intentions. In a savage and sudden coup, terrorists hijack the vessel, turning a mercy ship into a floating bio-weapons lab.\nA world away, SIGMA's Commander Gray Pierce thwarts the murderous schemes of a beautiful would-be killer who holds the first clue to the discovery of a possible cure. Pierce joins forces with the woman who wanted him dead, and together they embark upon an astonishing quest following the trail of the most fabled explorer in history: Marco Polo. But time is an enemy as a worldwide pandemic grows rapidly out of control. As a relentless madman dogs their every step, Gray and his unlikely ally are being pulled into an astonishing mystery buried deep in antiquity and in humanity's genetic code. And as the seconds tick closer to doomsday, Gray Pierce will realize he can truly trust no one, for any one of them could be . . . a Judas. | Authors: James Rollins"],
    ['Favorite books: Scarlet (Scarlet, #1) by A.C. Gaughen, Frostbite (Vampire Academy, #2) by Richelle Mead and The City of Ember (Book of Ember, #1) by Jeanne DuPrau. Favorite genres: action and adventure.', "Title: Flowers for Algernon | Genres: children, contemporary, drama, fantasy, health, literature, romance, science, scifi, war, ya | Description: The story of a mentally disabled man whose experimental quest for intelligence mirrors that of Algernon, an extraordinary lab mouse. In diary entries, Charlie tells how a brain operation increases his IQ and changes his life. As the experimental procedure takes effect, Charlie's intelligence expands until it surpasses that of the doctors who engineered his metamorphosis. The experiment seems to be a scientific breakthrough of paramount importance--until Algernon begins his sudden, unexpected deterioration. Will the same happen to Charlie? | Authors: Daniel Keyes"],
]
scores = model.predict(pairs)
print(scores.shape)
# (5,)

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    'Favorite books: To Kill a Mockingbird by Harper Lee, A Common Life: The Wedding Story (Mitford Years, #6) by Jan Karon and The Hiding Place: The Triumphant True Story of Corrie Ten Boom by Corrie ten Boom and John Sherrill and Elizabeth Sherrill. Favorite genres: contemporary and drama.',
    [
        "Title: A Tale of Two Cities | Genres: adventure, drama, historical, history, literature, romance, war | Description: 'Liberty, equality, fraternity, or death; -- the last, much the easiest to bestow, O Guillotine!'\nAfter eighteen years as a political prisoner in the Bastille, the ageing Doctor Manette is finally released and reunited with his daughter in England. There the lives of two very different men, Charles Darnay, an exiled French aristocrat, and Sydney Carton, a disreputable but brilliant English lawyer, become enmeshed through their love for Lucie Manette. From the tranquil roads of London, they are drawn against their will to the vengeful, bloodstained streets of Paris at the height of the Reign of Terror, and they soon fall under the lethal shadow of La Guillotine.\nThis edition uses the text as it appeared in its serial publication in 1859 to convey the full scope of Dickens's vision, and includes the original illustrations by H. K. Browne ('Phiz'). Richard Maxwell's introduction discusses the intricate interweaving of epic drama with personal tragedy.\n--back cover | Authors: Charles Dickens, Richard Maxwell, Hablot Knight Browne",
        'Title: Resurrection Men (Inspector Rebus, #13) | Genres: art, contemporary, crime, horror, literature, mystery, noir, suspense, thriller, war | Description: Inspector John Rebus has messed up badly this time, so badly that he\'s been sent to a kind of reform school for damaged cops. While there among the last-chancers known as "resurrection men," he joins a covert mission to gain evidence of a drug heist orchestrated by three of his classmates. But the group has been assigned an unsolved murder that may have resulted from Rebus\'s own mistake. Now Rebus can\'t determine if he\'s been set up for a fall or if his disgraced classmates are as ruthless as he suspects. When Detective Sergeant Siobhan Clarke discovers that her investigation of an art dealer\'s murder is tied to Rebus\'s inquiry, the two--protege and mentor--join forces. Soon they find themselves in the midst of an even bigger scandal than they had imagined-a plot with conspirators in every corner of Scotland and deadly implications about their colleagues. With the brilliant eye for character and place that earned him the name "the Dickens of Edinburgh," Ian Rankin delivers a page-turning novel of intricate suspense. | Authors: Ian Rankin',
        "Title: The Deep Blue Good-By (Travis McGee, #1) | Genres: action, adventure, crime, hardboiled, literature, mystery, noir, suspense, thriller | Description: TRAVIS McGEE\nHe's a self-described beach bum who won his houseboat in a card game. He's also a knight errant who's wary of credit cards, retirement benefits, political parties, mortgages, and television. He only works when his cash runs out and his rule is simple: he'll help you find whatever was taken from you, as long as he can keep half. | Authors: John D. MacDonald, Carl Hiaasen",
        "Title: The Judas Strain (Sigma Force, #4) | Genres: action, adventure, contemporary, crime, fantasy, historical, history, mystery, science, scifi, suspense, techno-thriller, thriller | Description: New York Times bestselling author James Rollins returns with a terrifying story of an ancient menace reborn to plague the modern world . . . and of an impossible hope that lies hidden in the most shocking place imaginable: within the language of angels.ju*das strain, n. A scientific term for an organism that drives an entire species to extinction.From the depths of the Indian Ocean, a horrific plague has arisen to devastate humankind--a disease that's unknown, unstoppable . . . and deadly. But it is merely a harbinger of the doom that is to follow. Aboard a cruise liner transformed into a makeshift hospital, Dr. Lisa Cummings and Monk Kokkalis--operatives of SIGMA Force--search for answers to the bizarre affliction. But there are others with far less altruistic intentions. In a savage and sudden coup, terrorists hijack the vessel, turning a mercy ship into a floating bio-weapons lab.\nA world away, SIGMA's Commander Gray Pierce thwarts the murderous schemes of a beautiful would-be killer who holds the first clue to the discovery of a possible cure. Pierce joins forces with the woman who wanted him dead, and together they embark upon an astonishing quest following the trail of the most fabled explorer in history: Marco Polo. But time is an enemy as a worldwide pandemic grows rapidly out of control. As a relentless madman dogs their every step, Gray and his unlikely ally are being pulled into an astonishing mystery buried deep in antiquity and in humanity's genetic code. And as the seconds tick closer to doomsday, Gray Pierce will realize he can truly trust no one, for any one of them could be . . . a Judas. | Authors: James Rollins",
        "Title: Flowers for Algernon | Genres: children, contemporary, drama, fantasy, health, literature, romance, science, scifi, war, ya | Description: The story of a mentally disabled man whose experimental quest for intelligence mirrors that of Algernon, an extraordinary lab mouse. In diary entries, Charlie tells how a brain operation increases his IQ and changes his life. As the experimental procedure takes effect, Charlie's intelligence expands until it surpasses that of the doctors who engineered his metamorphosis. The experiment seems to be a scientific breakthrough of paramount importance--until Algernon begins his sudden, unexpected deterioration. Will the same happen to Charlie? | Authors: Daniel Keyes",
    ]
)
# [{'corpus_id': ..., 'score': ...}, {'corpus_id': ..., 'score': ...}, ...]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Cross Encoder Reranking

* Dataset: `val_rerank`
* Evaluated with [<code>CrossEncoderRerankingEvaluator</code>](https://sbert.net/docs/package_reference/cross_encoder/evaluation.html#sentence_transformers.cross_encoder.evaluation.CrossEncoderRerankingEvaluator) with these parameters:
  ```json
  {
      "at_k": 10
  }
  ```

| Metric      | Value     |
|:------------|:----------|
| map         | 0.8096    |
| mrr@10      | 0.9365    |
| **ndcg@10** | **0.883** |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 222,360 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                                        | sentence_1                                                                                          | label                                                          |
  |:--------|:--------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                                            | string                                                                                              | float                                                          |
  | details | <ul><li>min: 145 characters</li><li>mean: 231.35 characters</li><li>max: 576 characters</li></ul> | <ul><li>min: 140 characters</li><li>mean: 1047.24 characters</li><li>max: 3691 characters</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.26</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                   | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | label            |
  |:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>Favorite books: To Kill a Mockingbird by Harper Lee, A Common Life: The Wedding Story (Mitford Years, #6) by Jan Karon and The Hiding Place: The Triumphant True Story of Corrie Ten Boom by Corrie ten Boom and John Sherrill and Elizabeth Sherrill. Favorite genres: contemporary and drama.</code> | <code>Title: A Tale of Two Cities | Genres: adventure, drama, historical, history, literature, romance, war | Description: 'Liberty, equality, fraternity, or death; -- the last, much the easiest to bestow, O Guillotine!'<br>After eighteen years as a political prisoner in the Bastille, the ageing Doctor Manette is finally released and reunited with his daughter in England. There the lives of two very different men, Charles Darnay, an exiled French aristocrat, and Sydney Carton, a disreputable but brilliant English lawyer, become enmeshed through their love for Lucie Manette. From the tranquil roads of London, they are drawn against their will to the vengeful, bloodstained streets of Paris at the height of the Reign of Terror, and they soon fall under the lethal shadow of La Guillotine.<br>This edition uses the text as it appeared in its serial publication in 1859 to convey the full scope of Dickens's vision, and includes the original illustrations by H. K. Browne ('Phiz'). Richard Maxwell's introdu...</code> | <code>1.0</code> |
  | <code>Favorite books: Alice in Zombieland (White Rabbit Chronicles, #1) by Gena Showalter, Animal Magnetism (Animal Magnetism, #1) by Jill Shalvis and Chitty Chitty Bang Bang (Chitty Chitty Bang Bang, #1) by Ian Fleming and Brian Selznick. Favorite genres: action and adventure.</code>                | <code>Title: Resurrection Men (Inspector Rebus, #13) | Genres: art, contemporary, crime, horror, literature, mystery, noir, suspense, thriller, war | Description: Inspector John Rebus has messed up badly this time, so badly that he's been sent to a kind of reform school for damaged cops. While there among the last-chancers known as "resurrection men," he joins a covert mission to gain evidence of a drug heist orchestrated by three of his classmates. But the group has been assigned an unsolved murder that may have resulted from Rebus's own mistake. Now Rebus can't determine if he's been set up for a fall or if his disgraced classmates are as ruthless as he suspects. When Detective Sergeant Siobhan Clarke discovers that her investigation of an art dealer's murder is tied to Rebus's inquiry, the two--protege and mentor--join forces. Soon they find themselves in the midst of an even bigger scandal than they had imagined-a plot with conspirators in every corner of Scotland and deadly implications...</code>       | <code>0.0</code> |
  | <code>Favorite books: Point of Impact (Bob Lee Swagger, #1) by Stephen Hunter, Stalking the Angel (Elvis Cole, #2) by Robert Crais and The Brass Verdict (Harry Bosch, #14; Mickey Haller, #2; Harry Bosch Universe, #17) by Michael Connelly. Favorite genres: action and crime.</code>                     | <code>Title: The Deep Blue Good-By (Travis McGee, #1) | Genres: action, adventure, crime, hardboiled, literature, mystery, noir, suspense, thriller | Description: TRAVIS McGEE<br>He's a self-described beach bum who won his houseboat in a card game. He's also a knight errant who's wary of credit cards, retirement benefits, political parties, mortgages, and television. He only works when his cash runs out and his rule is simple: he'll help you find whatever was taken from you, as long as he can keep half. | Authors: John D. MacDonald, Carl Hiaasen</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | <code>1.0</code> |
* Loss: [<code>BinaryCrossEntropyLoss</code>](https://sbert.net/docs/package_reference/cross_encoder/losses.html#binarycrossentropyloss) with these parameters:
  ```json
  {
      "activation_fn": "torch.nn.modules.linear.Identity",
      "pos_weight": null
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 48
- `per_device_eval_batch_size`: 48
- `fp16`: True

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 48
- `per_device_eval_batch_size`: 48
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 3
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: True
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `tp_size`: 0
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional

</details>

### Training Logs
| Epoch  | Step | Training Loss | val_rerank_ndcg@10 |
|:------:|:----:|:-------------:|:------------------:|
| 0.1079 | 500  | 0.6475        | -                  |
| 0.2158 | 1000 | 0.4707        | -                  |
| 0.3238 | 1500 | 0.4473        | -                  |
| 0.4317 | 2000 | 0.4354        | -                  |
| 0.5396 | 2500 | 0.4261        | -                  |
| 0.6475 | 3000 | 0.4183        | -                  |
| 0.7555 | 3500 | 0.4109        | -                  |
| 0.8634 | 4000 | 0.4107        | -                  |
| 0.9713 | 4500 | 0.405         | -                  |
| 1.0    | 4633 | -             | 0.8735             |
| 1.0792 | 5000 | 0.381         | -                  |
| 1.1871 | 5500 | 0.3717        | -                  |
| 1.2951 | 6000 | 0.3732        | -                  |
| 1.4030 | 6500 | 0.3756        | -                  |
| 1.5109 | 7000 | 0.3703        | -                  |
| 1.6188 | 7500 | 0.3615        | -                  |
| 1.7267 | 8000 | 0.3604        | -                  |
| 1.8347 | 8500 | 0.3603        | -                  |
| 1.9426 | 9000 | 0.3529        | -                  |
| 2.0    | 9266 | -             | 0.8830             |


### Framework Versions
- Python: 3.11.12
- Sentence Transformers: 4.1.0
- Transformers: 4.51.3
- PyTorch: 2.6.0+cu124
- Accelerate: 1.6.0
- Datasets: 2.14.4
- Tokenizers: 0.21.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->