## Smart Tab Grouping Topic Model

A flow to fine tune a generic LLM to a generative Topic Model for tab group topics

Metaflow or Outerbounds libraries need to be


**To run training jobs**
*local*
Remove 'kubernetes' tag from TuneGenTopicModel.py
python TuneGenTopicModel.py run

*remote cluster*
python TuneGenTopicModel.py argo-workflows create
python TuneGenTopicModel.py argo-workflows trigger


To quantize, go to W&B and download the artifacts of the preferred run.
(Note that tokenizer information is not saved with each run, so you may need to copy it 
from a previous instance on huggingface at https://huggingface.co/Mozilla/smart-tab-topic)

```
python -m convert --model_id <model_dir> --quantize --modes q8 --task text2text-generation-with-past
cd models/<model_dir>
<commit changes to repo>
git tag v0.<x>.<y>
git push origin --tags
```

