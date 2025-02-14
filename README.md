# X-Boundary: Establishing Exact Safety Boundary to Shield LLMs from Multi-Turn Jailbreaks without Compromising Usability

[Paper]() | [Code](https://github.com/AI45Lab/X-Boundary) | [Models](https://huggingface.co/collections/Ursulalala/x-boundary-67af0133e09495b151f7ab07)

In this paper, we comprehensively compare existing defense methods in multi-turn attack scenarios and reveal their shortcomings in balancing the robustness of defense and LLM usability. We analyze this issue from the perspective of LLMs' feature space, and conclude that previous methods fail to learn a precise boundary that distinguishes safe and harmful representations without an explicit formulation. To address this issue, we propose the X-Boundary to push harmful representations away from safe representations through explicit loss functions and obtain a clear distinction boundary. Such distinction boundary enables the consequential removal of harmful representations without disrupting safe ones, thereby achieving a balance between robustness against multi-turn jailbreaks and LLM usability.

![alt text](asset/motivation.png)

## Snapshot of Results
![alt text](asset/experiment.png)

## Installation

conda create -n xboun python=3.10
conda activate xboun

pip install -r requirements.txt

## Training

```shell
sh scripts/lorra_x_boundary_llama3_8b.sh

sh scripts/lorra_x_boundary_qwen2_7b.sh
```

## Evaluation
evaluate defense against single-turn attack in HarmBench
```shell
sh scripts/eval/eval_cb.sh $model_path &
```

evaluate defense against ActorAttack
```shell
sh scripts/eval/multi_round_eval.sh $model_path &
```

evaluate defense against RedQueen attack
```shell
sh scripts/eval/red_queen_eval.sh $model_path &

sh scripts/eval/red_queen_eval_llama.sh $model_path & # for llama-3
```

evaluate over-refusal rate
```shell
sh scripts/eval/overrefusal_eval.sh $model_path data/test/OKTest.json &

sh scripts/eval/overrefusal_eval.sh $model_path data/test/PHtest.json &

sh scripts/eval/overrefusal_eval.sh $model_path data/test/ORbench_test300.json &

sh scripts/eval/overrefusal_eval.sh $model_path data/test/xstest_v2_prompts.json &
```

## Acknowledge
Leveraged the part of code framework of [Circuit Breaker](https://github.com/GraySwanAI/circuit-breakers).

## Citation
```

```