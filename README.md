# X-Boundary: Establishing Exact Safety Boundary to Shield LLMs from Multi-Turn Jailbreaks without Compromising Usability

[Paper](https://arxiv.org/abs/2502.09990) | [Code](https://github.com/AI45Lab/X-Boundary) | [Models](https://huggingface.co/collections/Ursulalala/x-boundary-67af0133e09495b151f7ab07)

In this paper, we comprehensively compare existing defense methods in multi-turn attack scenarios and reveal their shortcomings in balancing the robustness of defense and LLM usability. We analyze this issue from the perspective of LLMs' feature space, and conclude that previous methods fail to learn a precise boundary that distinguishes safe and harmful representations without an explicit formulation. To address this issue, we propose the X-Boundary to push harmful representations away from safe representations through explicit loss functions and obtain a clear distinction boundary. Such distinction boundary enables the consequential removal of harmful representations without disrupting safe ones, thereby achieving a balance between robustness against multi-turn jailbreaks and LLM usability.

![alt text](asset/motivation.png)

## Snapshot of Results
![alt text](asset/experiment.png)

## Installation
``` shell
conda create -n xboun python=3.10
conda activate xboun
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Training

```shell
sh scripts/lorra_x_boundary_llama3_8b.sh

sh scripts/lorra_x_boundary_qwen2_7b.sh
```

## Evaluation
Evaluate defense against single-turn attack in HarmBench
```shell
sh scripts/eval/eval_cb.sh $model_path
```

Evaluate defense against ActorAttack
```shell
sh scripts/eval/multi_round_eval.sh $model_path
```

Evaluate defense against RedQueen attack
```shell
sh scripts/eval/red_queen_eval.sh $model_path

sh scripts/eval/red_queen_eval_llama.sh $model_path # for llama-3
```

Evaluate over-refusal rate
```shell
sh scripts/eval/overrefusal_eval.sh $model_path data/test/OKTest.json

sh scripts/eval/overrefusal_eval.sh $model_path data/test/PHtest.json

sh scripts/eval/overrefusal_eval.sh $model_path data/test/ORbench_test300.json

sh scripts/eval/overrefusal_eval.sh $model_path data/test/xstest_v2_prompts.json
```

If you want to speed up inference, especially when using a reasoning model like deepseek-R1, you can set "--vlm_acc true" in your evaluation scripts to use vllm as the backend of inference.

``` shell
pip install vllm==0.7.3
```

For the R1 distilled models, we set the max_new_tokens to 8192 for evaluating single-turn safety and over-refusal, and 2048 for evaluation under multi-turn attacks.

## Inference with X-Boundary adapter and gibberish filter

After the harmful representations are erased, the LLM has a certain probability of generating gibberish due to its inability to produce harmful content. We can use a rule-based detector to identify gibberish and replace it with a refusal response. This post-processing generally does not affect normal outputs. We provide a demo in `R1_X_Boundary_demo.py`.

![alt text](asset/reasoning_model_case.png)

## Acknowledge
Leveraged the part of code framework of [Circuit Breaker](https://github.com/GraySwanAI/circuit-breakers).

## Citation
```
@misc{lu2025xboundarye,
      title={X-Boundary: Establishing Exact Safety Boundary to Shield LLMs from Multi-Turn Jailbreaks without Compromising Usability}, 
      author={Xiaoya Lu and Dongrui Liu and Yi Yu and Luxin Xu and Jing Shao},
      year={2025},
      eprint={2502.09990},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2502.09990}, 
}
```
