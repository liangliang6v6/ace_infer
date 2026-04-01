export CUDA_VISIBLE_DEVICES=1,2,3,4
echo "Run decomposer"
python inference/decompose_vllm.py \
    --model_path "AceSearcher/AceSearcher-14B" \
    --tokenizer "Qwen/Qwen2.5-14B-Instruct" \
    --datasets "hotpotqa" \
    --expname "acesearcher" \
    --tensor_parallel_size 4 \
    --temperature 0.0

echo "Run solver"
python inference/main_qa.py \
    --llm_model_path "AceSearcher/AceSearcher-14B" \
    --llm_tokenizer "Qwen/Qwen2.5-14B-Instruct" \
    --dataset "hotpotqa" \
    --expname "acesearcher" \
    --save_dir "eval_datasets/test" \
    --sentence_embedding_model "intfloat/e5-large-v2" \
    --sentence_embedding_model_save_name "e5-large-v2" \
    --k 10 \
    --add_passage 1 \
    --tensor_parallel_size 4