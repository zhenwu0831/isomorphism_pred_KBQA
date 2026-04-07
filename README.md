# isomorphism_pred_KBQA

## Setting up vLLM server

```bash
vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 65536
```

```bash
vllm serve Qwen/Qwen3-14B \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.8
```

## Temperature (Agentic Setting)

We use the following models and temperature settings:

| Model                       | Temperature |
| --------------------------- | ----------- |
| GPT-5-mini                  | 1.0         |
| Claude Sonnet 4.6           | 1.0         |
| Kimi 2.5                    | 0.6         |
| Qwen3-30B-A3B-Instruct-2507 | 1.0         |
| Qwen3-14B                   | 0.7         |
