export VLLM_WORKER_MULTIPROC_METHOD=spawn
python3 run_src/do_raise.py \
    --dataset_name STG \
    --test_json_filename test \
    --model_ckpt Meta-Llama-3.1-8B-Instruct \
    --note default \
    --num_rollouts 4 \
    --mode run \
    --disable_a5 \
    --disable_a8 \
    --retrieval_corpus "wikipedia" \
    --retrieval_threshold 0.5 \
    --enable_chat_template true \
