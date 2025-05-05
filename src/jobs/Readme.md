## Smart Tab Grouping Topic Model

A flow to fine tune a generic LLM to a generative Topic Model for tab group topics

Metaflow or Outerbounds libraries need to be


**To run training jobs**

*Nvidia Dedicated GPUs*
python TuneGenTopicModel.py --no-pylint --environment=conda run

*remote cluster*
python TuneGenTopicModel.py argo-workflows create
python TuneGenTopicModel.py argo-workflows trigger
To use remote cluster, the header of this step should have
@conda and @nvida tags removed and this added
```
    @kubernetes(image="us-docker.pkg.dev/moz-fx-mozsoc-ml-nonprod/metaflow-dockers/metaflow_gpu:onnx2",
                gpu_vendor="nvidia",
                gpu=1,
                memory=10240,
                disk=20240,
                cpu=2,
                )
```


Note that there are to different configs when running TuneGenTopicModel.py
1) Fine tuning a flan-t5-base-model
2) Distillation tuning that artifact into a t5-efficient-mini (optinally removing additional layers)

As a final step to quantize, go to W&B and download the artifacts of the preferred run from GCS.
(path is in the run logs)

```
python -m convert --model_id <model_dir> --quantize --modes q8 --task text2text-generation-with-past
cd models/<model_dir> 
<copy files to huggingface repo dir>
<commit changes to repo>
git tag v0.<x>.<y>
git push origin --tags
```

