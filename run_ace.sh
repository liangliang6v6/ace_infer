# AceSearcher/AceSearcher-14B Qwen/Qwen2.5-14B-Instruct
# 2wikimultihopqa musique hotpotqa

# echo "Run decomposer"
# export CUDA_VISIBLE_DEVICES=0,1
# python inference/decompose_vllm.py \
#     --model_path "AceSearcher/AceSearcher-14B" \
#     --tokenizer "AceSearcher/AceSearcher-14B" \
#     --datasets "2wiki_origin" \
#     --expname "ace" \
#     --tensor_parallel_size 2 \
#     --temperature 0.0


echo "Run solver"
export CUDA_VISIBLE_DEVICES=2,3
python inference/main_qa.py \
    --llm_model_path "AceSearcher/AceSearcher-14B" \
    --llm_tokenizer "AceSearcher/AceSearcher-14B" \
    --dataset "2wiki_origin" \
    --expname "ace" \
    --save_dir "eval_datasets/test" \
    --sentence_embedding_model "intfloat/e5-large-v2" \
    --sentence_embedding_model_save_name "e5-large-v2" \
    --temperature 0.0\
    --k 10 \
    --add_passage 0 \
    --tensor_parallel_size 2