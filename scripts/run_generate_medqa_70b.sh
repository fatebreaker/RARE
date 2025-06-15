export VLLM_WORKER_MULTIPROC_METHOD=spawn
python3 run_src/do_raise.py \
    --dataset_name SIQA \
    --test_json_filename dev \
    --model_ckpt Meta-Llama-3.1-70B-Instruct \
    --note default \
    --num_rollouts 1 \
    --mode run \
    --disable_a5 \
    --disable_a8 \
    --tensor_parallel_size 2 \
    --mcts_num_last_votes 32 \
    --enable_chat_template true \
    --disable_a7 \
    --disable_a4 \
    --disable_a1 \
    --disable_a6 \
