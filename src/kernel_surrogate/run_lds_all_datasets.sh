conda run -n vlm python -m src.kernel_surrogate.lds_real_train \
    --datasets imdb humor sarcasm mosi mosei avmnist \
    --seeds 0 \
    --device cuda:0 \
    --feature-device cuda:0 \
    --num-groups 50 \
    --num-subset-samples 100 \
    --data-keep-rate 0.5 \
    --modality-keep-rate 0.5 \
    --train-repeat-seeds 0 \
    --max-test-loss-samples 1000 \
    --run-name grouped50_subset100_data50_mod50

conda run -n vlm python -m src.kernel_surrogate.lds_real_train \
    --datasets imdb humor sarcasm mosi mosei avmnist \
    --seeds 0 \
    --device cuda:0 \
    --feature-device cuda:0 \
    --num-groups 50 \
    --num-subset-samples 1 \
    --data-keep-rate 1.0 \
    --modality-keep-rate 1.0 \
    --train-repeat-seeds 0 \
    --max-test-loss-samples 1000 \
    --run-name grouped50_full_data_train