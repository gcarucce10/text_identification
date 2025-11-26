from utils import calculate_cer_wer

ground_truth_list, predicted_list = [], []
for i in range(1,29):
    with open(f'tests/metrics/inputs/gt_line{i}.txt','r') as f:
        ground_truth_list.append(f.read())
    with open(f'tests/metrics/inputs/pr_line{i}.txt','r') as f:
        predicted_list.append(f.read())

cer, wer = calculate_cer_wer(ground_truth_list, predicted_list)
with open(f'tests/metrics/outputs/results.txt','w') as f:
    f.write(f'CER: {cer}\nWER: {wer}')