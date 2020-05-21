import csv
import os
import json

experiments = ['gs://oto-experiments/byidh13as828-27', 'gs://oto-experiments/byidh13as828-28', 'gs://oto-experiments/byidh13as828-29', 'gs://oto-experiments/byidh13as828-30', 'gs://oto-experiments/byidh13as828-31', 'gs://oto-experiments/byidh13as828-32', 'gs://oto-experiments/byidh13as828-33', 'gs://oto-experiments/byidh13as828-34', 'gs://oto-experiments/byidh13as828-35']
relations = ["per:spouse"]*len(experiments)

RESULTS_FILE = 'temp_results_file.json'
DEV_RESULTS_FILE = 'temp_dev_results.json'
DEV_SCORES_FILE = 'temp_full_dev_eval_scores.json'
TEST_RESULTS_FILE = 'temp_test_results.json'
TEST_SCORES_FILE = 'temp_full_test_eval_scores.json'
ALL_OUTPUTS_FILE = 'all_outputs_file.tsv'

def main():
    with open(ALL_OUTPUTS_FILE, 'w') as all_outputs_file:
        tsv_writer = csv.writer(all_outputs_file, delimiter='\t')

        for experiment, relation in zip(experiments, relations):
            cmd = f"gsutil cp {experiment}/full_results.json temp_results_file.json"
            os.system(cmd)
            results = json.load(open(RESULTS_FILE, 'r'))
            json.dump(results['full_dev_eval_results'][0], open(DEV_RESULTS_FILE, 'w'))
            json.dump(results['full_test_eval_results'][0], open(TEST_RESULTS_FILE, 'w'))

            cmd = f"python -m classification.evaluation.tacred_evaluation --gold_dir data/tacred --gold_file dev.json --relation_name {relation} --pred_file {DEV_RESULTS_FILE} --output_file {DEV_SCORES_FILE}"
            # cmd = f"python -m classification.evaluation.docred_evaluation --gold_dir data/DocRED --gold_file eval_split_from_annotated.json --relation_name {relation} --pred_file {DEV_RESULTS_FILE} --output_file {DEV_SCORES_FILE}"
            os.system(cmd)

            confidence_threshold_on_dev_eval = json.load(open(DEV_SCORES_FILE, 'r'))['best_confidence']
            
            cmd = f"python -m classification.evaluation.tacred_evaluation --gold_dir data/tacred --gold_file test.json --relation_name {relation} --pred_file {TEST_RESULTS_FILE} --confidence_threshold {confidence_threshold_on_dev_eval} --output_file {TEST_SCORES_FILE}"
            # cmd = f"python -m classification.evaluation.docred_evaluation --gold_dir data/DocRED --gold_file dev.json --relation_name {relation} --pred_file {TEST_RESULTS_FILE} --confidence_threshold {confidence_threshold_on_dev_eval} --output_file {TEST_SCORES_FILE}"
            os.system(cmd)

            test_scores = json.load(open(TEST_SCORES_FILE, 'r'))
            test_f1, test_precision, test_recall = test_scores['F1'], test_scores['precision'], test_scores['recall']

            tsv_writer.writerow([test_f1, test_precision, test_recall, confidence_threshold_on_dev_eval])

            # delete files
            # os.remove(RESULTS_FILE)
            # os.remove(DEV_RESULTS_FILE)
            # os.remove(DEV_SCORES_FILE)
            # os.remove(TEST_RESULTS_FILE)
            # os.remove(TEST_SCORES_FILE)


if __name__ == "__main__":
    main()