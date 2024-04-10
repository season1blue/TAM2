
# python3 utils/TrainInputProcess.py

#! /bin/bash
python3 ./Train.py \
--epochs 100 \
--save_steps 300 \
--dataset_type 2015 \
--batch_size 16 \
--lr 2e-5 \
--text_model_name "deberta" \
--image_model_name "vit" \
--output_dir /data/results \
--output_result_file /data/result.txt \
--log_dir ./data/log.log \
--device_id "cuda:1" \
--enable_log \
--only_text_loss \
--add_gan \
# --add_gan_loss
# --alpha 0 \
# --beta 0 \
# --add_llm \
