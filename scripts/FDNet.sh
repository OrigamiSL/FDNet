# ETT
python -u main.py --model FDNet --data ETTh1 --features S  --label_len 672  --pred_len 96 --pyramid 4 --attn_nums 5 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --d_model 8 --timebed None --batch_size 16 --itr 10 --train

python -u main.py --model FDNet --data ETTh1 --features S  --label_len 672  --pred_len 192 --pyramid 4 --attn_nums 5 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --d_model 8 --timebed None --batch_size 16 --itr 10 --train

python -u main.py --model FDNet --data ETTh1 --features S  --label_len 672  --pred_len 336 --pyramid 4 --attn_nums 5 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --d_model 8 --timebed None --batch_size 16 --itr 10 --train

python -u main.py --model FDNet --data ETTh1 --features S  --label_len 672  --pred_len 720 --pyramid 4 --attn_nums 5 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --d_model 8 --timebed None --batch_size 16 --itr 10 --train

python -u main.py --model FDNet --data ETTh1 --features M  --label_len 672  --pred_len 96 --pyramid 4 --attn_nums 5 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --d_model 8 --timebed None --batch_size 16 --itr 10 --train

python -u main.py --model FDNet --data ETTh1 --features M  --label_len 672  --pred_len 192 --pyramid 4 --attn_nums 5 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --d_model 8 --timebed None --batch_size 16 --itr 10 --train

python -u main.py --model FDNet --data ETTh1 --features M  --label_len 672  --pred_len 336 --pyramid 4 --attn_nums 5 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --d_model 8 --timebed None --batch_size 16 --itr 10 --train

python -u main.py --model FDNet --data ETTh1 --features M  --label_len 672  --pred_len 720 --pyramid 4 --attn_nums 5 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --d_model 8 --timebed None --batch_size 16 --itr 10 --train

python -u main.py --model FDNet --data ETTm2 --features S  --label_len 672  --pred_len 96 --pyramid 4 --attn_nums 5 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --d_model 8 --timebed None --batch_size 16 --itr 10 --train

python -u main.py --model FDNet --data ETTm2 --features S  --label_len 672  --pred_len 192 --pyramid 4 --attn_nums 5 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --d_model 8 --timebed None --batch_size 16 --itr 10 --train

python -u main.py --model FDNet --data ETTm2 --features S  --label_len 672  --pred_len 336 --pyramid 4 --attn_nums 5 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --d_model 8 --timebed None --batch_size 16 --itr 10 --train

python -u main.py --model FDNet --data ETTm2 --features S  --label_len 672  --pred_len 720 --pyramid 4 --attn_nums 5 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --d_model 8 --timebed None --batch_size 16 --itr 10 --train

python -u main.py --model FDNet --data ETTm2 --features M  --label_len 672  --pred_len 96 --pyramid 4 --attn_nums 5 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --d_model 8 --timebed None --batch_size 16 --itr 10 --train

python -u main.py --model FDNet --data ETTm2 --features M  --label_len 672  --pred_len 192 --pyramid 4 --attn_nums 5 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --d_model 8 --timebed None --batch_size 16 --itr 10 --train

python -u main.py --model FDNet --data ETTm2 --features M  --label_len 672  --pred_len 336 --pyramid 4 --attn_nums 5 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --d_model 8 --timebed None --batch_size 16 --itr 10 --train

python -u main.py --model FDNet --data ETTm2 --features M  --label_len 672  --pred_len 720 --pyramid 4 --attn_nums 5 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --d_model 8 --timebed None --batch_size 16 --itr 10 --train

# ECL
python -u main.py --model FDNet --data ECL --root_path ./data/ECL/ --features S  --label_len 672  --pred_len 96 --pyramid 4 --attn_nums 5 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --d_model 8 --timebed None --batch_size 16 --target MT_321 --itr 10 --train

python -u main.py --model FDNet --data ECL --root_path ./data/ECL/ --features S  --label_len 672  --pred_len 192 --pyramid 4 --attn_nums 5 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --d_model 8 --timebed None --batch_size 16 --target MT_321 --itr 10 --train

python -u main.py --model FDNet --data ECL --root_path ./data/ECL/ --features S  --label_len 672  --pred_len 336 --pyramid 4 --attn_nums 5 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --d_model 8 --timebed None --batch_size 16 --target MT_321 --itr 10 --train

python -u main.py --model FDNet --data ECL --root_path ./data/ECL/ --features S  --label_len 672  --pred_len 720 --pyramid 4 --attn_nums 5 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --d_model 8 --timebed None --batch_size 16 --target MT_321 --itr 10 --train

python -u main.py --model FDNet --data ECL --root_path ./data/ECL/ --features M  --label_len 672  --pred_len 96 --pyramid 4 --attn_nums 5 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --d_model 8 --timebed None --batch_size 16 --target MT_321 --itr 10 --train

python -u main.py --model FDNet --data ECL --root_path ./data/ECL/ --features M  --label_len 672  --pred_len 192 --pyramid 4 --attn_nums 5 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --d_model 8 --timebed None --batch_size 16 --target MT_321 --itr 10 --train

python -u main.py --model FDNet --data ECL --root_path ./data/ECL/ --features M  --label_len 672  --pred_len 336 --pyramid 4 --attn_nums 5 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --d_model 8 --timebed None --batch_size 16 --target MT_321 --itr 10 --train

python -u main.py --model FDNet --data ECL --root_path ./data/ECL/ --features M  --label_len 672  --pred_len 720 --pyramid 4 --attn_nums 5 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --d_model 8 --timebed None --batch_size 16 --target MT_321 --itr 10 --train

# Traffic
python -u main.py --model FDNet --data Traffic --root_path ./data/Traffic/ --features S  --label_len 672  --pred_len 96 --pyramid 4 --attn_nums 5 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --d_model 8 --timebed None --batch_size 16 --target Sensor_861 --itr 10 --train

python -u main.py --model FDNet --data Traffic --root_path ./data/Traffic/ --features S  --label_len 672  --pred_len 192 --pyramid 4 --attn_nums 5 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --d_model 8 --timebed None --batch_size 16 --target Sensor_861 --itr 10 --train

python -u main.py --model FDNet --data Traffic --root_path ./data/Traffic/ --features S  --label_len 672  --pred_len 336 --pyramid 4 --attn_nums 5 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --d_model 8 --timebed None --batch_size 16 --target Sensor_861 --itr 10 --train

python -u main.py --model FDNet --data Traffic --root_path ./data/Traffic/ --features S  --label_len 672  --pred_len 720 --pyramid 4 --attn_nums 5 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --d_model 8 --timebed None --batch_size 16 --target Sensor_861 --itr 10 --train

python -u main.py --model FDNet --data Traffic --root_path ./data/Traffic/ --features M  --label_len 672  --pred_len 96 --pyramid 4 --attn_nums 5 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --d_model 8 --timebed None --batch_size 16 --target Sensor_861 --itr 10 --train

python -u main.py --model FDNet --data Traffic --root_path ./data/Traffic/ --features M  --label_len 672  --pred_len 192 --pyramid 4 --attn_nums 5 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --d_model 8 --timebed None --batch_size 16 --target Sensor_861 --itr 10 --train

python -u main.py --model FDNet --data Traffic --root_path ./data/Traffic/ --features M  --label_len 672  --pred_len 336 --pyramid 4 --attn_nums 5 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --d_model 8 --timebed None --batch_size 16 --target Sensor_861 --itr 10 --train

python -u main.py --model FDNet --data Traffic --root_path ./data/Traffic/ --features M  --label_len 672  --pred_len 720 --pyramid 4 --attn_nums 5 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --d_model 8 --timebed None --batch_size 16 --target Sensor_861 --itr 10 --train

# weather
python -u main.py --model FDNet --data weather --root_path ./data/weather/ --features S  --label_len 672  --pred_len 96 --pyramid 4 --attn_nums 5 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --d_model 8 --timebed None --batch_size 16 --itr 10 --train

python -u main.py --model FDNet --data weather --root_path ./data/weather/ --features S  --label_len 672  --pred_len 192 --pyramid 4 --attn_nums 5 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --d_model 8 --timebed None --batch_size 16 --itr 10 --train

python -u main.py --model FDNet --data weather --root_path ./data/weather/ --features S  --label_len 672  --pred_len 336 --pyramid 4 --attn_nums 5 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --d_model 8 --timebed None --batch_size 16 --itr 10 --train

python -u main.py --model FDNet --data weather --root_path ./data/weather/ --features S  --label_len 672  --pred_len 720 --pyramid 4 --attn_nums 5 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --d_model 8 --timebed None --batch_size 16 --itr 10 --train

python -u main.py --model FDNet --data weather --root_path ./data/weather/ --features M  --label_len 672  --pred_len 96 --pyramid 4 --attn_nums 5 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --d_model 8 --timebed None --batch_size 16 --itr 10 --train

python -u main.py --model FDNet --data weather --root_path ./data/weather/ --features M  --label_len 672  --pred_len 192 --pyramid 4 --attn_nums 5 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --d_model 8 --timebed None --batch_size 16 --itr 10 --train

python -u main.py --model FDNet --data weather --root_path ./data/weather/ --features M  --label_len 672  --pred_len 336 --pyramid 4 --attn_nums 5 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --d_model 8 --timebed None --batch_size 16 --itr 10 --train

python -u main.py --model FDNet --data weather --root_path ./data/weather/ --features M  --label_len 672  --pred_len 720 --pyramid 4 --attn_nums 5 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --d_model 8 --timebed None --batch_size 16 --itr 10 --train

# Exchange
python -u main.py --model FDNet --data Exchange --root_path ./data/Exchange/ --features S  --label_len 96  --pred_len 96 --pyramid 0 --attn_nums 1 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --d_model 8 --timebed None --target Singapore --batch_size 16 --itr 10 --train

python -u main.py --model FDNet --data Exchange --root_path ./data/Exchange/ --features S  --label_len 96  --pred_len 192 --pyramid 0 --attn_nums 1 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --d_model 8 --timebed None --target Singapore --batch_size 16 --itr 10 --train

python -u main.py --model FDNet --data Exchange --root_path ./data/Exchange/ --features S  --label_len 96  --pred_len 336 --pyramid 0 --attn_nums 1 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --d_model 8 --timebed None --target Singapore --batch_size 16 --itr 10 --train

python -u main.py --model FDNet --data Exchange --root_path ./data/Exchange/ --features S  --label_len 96  --pred_len 720 --pyramid 0 --attn_nums 1 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --d_model 8 --timebed None --target Singapore --batch_size 16 --itr 10 --train

python -u main.py --model FDNet --data Exchange --root_path ./data/Exchange/ --features M  --label_len 96  --pred_len 96 --pyramid 0 --attn_nums 1 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --d_model 8 --timebed None --target Singapore --batch_size 16 --itr 10 --train

python -u main.py --model FDNet --data Exchange --root_path ./data/Exchange/ --features M  --label_len 96  --pred_len 192 --pyramid 0 --attn_nums 1 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --d_model 8 --timebed None --target Singapore --batch_size 16 --itr 10 --train

python -u main.py --model FDNet --data Exchange --root_path ./data/Exchange/ --features M  --label_len 96  --pred_len 336 --pyramid 0 --attn_nums 1 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --d_model 8 --timebed None --target Singapore --batch_size 16 --itr 10 --train

python -u main.py --model FDNet --data Exchange --root_path ./data/Exchange/ --features M  --label_len 96  --pred_len 720 --pyramid 0 --attn_nums 1 --kernel 3 --learning_rate 0.0001 --dropout 0.1 --criterion Standard --d_model 8 --timebed None --target Singapore --batch_size 16 --itr 10 --train

