from utils import calculate_cer_wer

ground_truth_list, htr_predicted_list, pos_predicted_list = [], [], []
for i in range(0,11):
    with open(f'tests/metrics/inputs/gt_linha{i}.txt','r') as f:
        ground_truth_list.append(f.read())
    with open(f'tests/metrics/inputs/htr_pr_linha{i}.txt','r') as f:
        htr_predicted_list.append(f.read())
    with open(f'tests/metrics/inputs/pos_pr_linha{i}.txt','r') as f:
        pos_predicted_list.append(f.read())

cer, wer = calculate_cer_wer(ground_truth_list, htr_predicted_list)
with open(f'tests/metrics/outputs/results.txt','w') as f:
    f.write(f'HTR METRICS\n  CER: {cer}\n  WER: {wer}\n')

cer, wer = calculate_cer_wer(ground_truth_list, pos_predicted_list)
with open(f'tests/metrics/outputs/results.txt','a') as f:
    f.write(f'POSTPROCESS METRICS\n  CER: {cer}\n  WER: {wer}')