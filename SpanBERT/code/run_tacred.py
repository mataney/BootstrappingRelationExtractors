# Copyright (c) 2019, Facebook, Inc. and its affiliates. All Rights Reserved
"""
Run BERT on several relation extraction benchmarks.
Adding some special tokens instead of doing span pair prediction in this version.
"""

import argparse
import json
import logging
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from torch.nn import CrossEntropyLoss

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from tacred_data_processor import DataProcessor, convert_examples_to_features

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def compute_f1(preds, labels):
    n_gold = n_pred = n_correct = 0
    for pred, label in zip(preds, labels):
        if pred != 0:
            n_pred += 1
        if label != 0:
            n_gold += 1
        if (pred != 0) and (label != 0) and (pred == label):
            n_correct += 1
    if n_correct == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    else:
        prec = n_correct * 1.0 / n_pred
        recall = n_correct * 1.0 / n_gold
        if prec + recall > 0:
            f1 = 2.0 * prec * recall / (prec + recall)
        else:
            f1 = 0.0
        return {'precision': prec, 'recall': recall, 'f1': f1}


def evaluate(model, device, eval_dataloader, eval_label_ids, num_labels, verbose=True):
    model.eval()
    eval_loss = 0
    nb_eval_steps = 0
    preds = []
    for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, labels=None)
        loss_fct = CrossEntropyLoss()
        tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    confidences, preds = torch.max(torch.from_numpy(preds[0]).float().softmax(1), axis=1)
    preds = preds.detach().cpu().numpy()
    confidences = confidences.detach().cpu().numpy()
    result = compute_f1(preds, eval_label_ids.numpy())
    result['accuracy'] = simple_accuracy(preds, eval_label_ids.numpy())
    result['eval_loss'] = eval_loss
    if verbose:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))

    return preds, result, confidences


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.do_train:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "train.log"), 'w'))
    else:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "eval.log"), 'w'))
    logger.info(args)
    logger.info("device: {}, n_gpu: {}, 16-bits training: {}".format(
        device, n_gpu, args.fp16))

    processor = DataProcessor(args.relation_name, args.num_positive_examples, args.ratio_negative_examples)
    label_list = processor.get_labels(args.data_dir, args.negative_label)
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}
    num_labels = len(label_list)
    tokenizer = BertTokenizer.from_pretrained(args.model, do_lower_case=args.do_lower_case)

    special_tokens = {}
    if args.do_eval:
        eval_examples = processor.get_dev_examples(args.data_dir)
        eval_features = convert_examples_to_features(
            eval_examples, label2id, args.max_seq_length, tokenizer, special_tokens, args.feature_mode)
        logger.info("***** Dev *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size)
        eval_label_ids = all_label_ids

    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        train_features = convert_examples_to_features(
                train_examples, label2id, args.max_seq_length, tokenizer, special_tokens, args.feature_mode)

        if args.train_mode == 'sorted' or args.train_mode == 'random_sorted':
            train_features = sorted(train_features, key=lambda f: np.sum(f.input_mask))
        else:
            random.shuffle(train_features)

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        train_dataloader = DataLoader(train_data, batch_size=args.train_batch_size)
        train_batches = [batch for batch in train_dataloader]

        num_train_optimization_steps = \
            len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

        logger.info("***** Training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        best_result = None
        eval_step = max(1, len(train_batches) // args.eval_per_epoch)
        lrs = [args.learning_rate] if args.learning_rate else \
            [1e-6, 2e-6, 3e-6, 5e-6, 1e-5, 2e-5, 3e-5, 5e-5]
        for lr in lrs:
            model = BertForSequenceClassification.from_pretrained(
                args.model, cache_dir=str(PYTORCH_PRETRAINED_BERT_CACHE), num_labels=num_labels)
            if args.fp16:
                model.half()
            model.to(device)
            if n_gpu > 1:
                model = torch.nn.DataParallel(model)

            param_optimizer = list(model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer
                            if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in param_optimizer
                            if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            if args.fp16:
                try:
                    from apex.optimizers import FP16_Optimizer
                    from apex.optimizers import FusedAdam
                except ImportError:
                    raise ImportError("Please install apex from https://www.github.com/nvidia/apex"
                                      "to use distributed and fp16 training.")

                optimizer = FusedAdam(optimizer_grouped_parameters,
                                      lr=lr,
                                      bias_correction=False,
                                      max_grad_norm=1.0)
                if args.loss_scale == 0:
                    optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
                else:
                    optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

            else:
                optimizer = BertAdam(optimizer_grouped_parameters,
                                     lr=lr,
                                     warmup=args.warmup_proportion,
                                     t_total=num_train_optimization_steps)

            start_time = time.time()
            global_step = 0
            tr_loss = 0
            nb_tr_examples = 0
            nb_tr_steps = 0
            n_no_improve = 0
            for epoch in range(int(args.num_train_epochs)):
                model.train()
                logger.info("Start epoch #{} (lr = {})...".format(epoch, lr))
                if args.train_mode == 'random' or args.train_mode == 'random_sorted':
                    random.shuffle(train_batches)
                for step, batch in enumerate(train_batches):
                    batch = tuple(t.to(device) for t in batch)
                    input_ids, input_mask, segment_ids, label_ids = batch
                    loss = model(input_ids, segment_ids, input_mask, label_ids)
                    if n_gpu > 1:
                        loss = loss.mean()
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps

                    if args.fp16:
                        optimizer.backward(loss)
                    else:
                        loss.backward()

                    tr_loss += loss.item()
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1

                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        if args.fp16:
                            lr_this_step = lr * \
                                warmup_linear(global_step/num_train_optimization_steps, args.warmup_proportion)
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = lr_this_step
                        optimizer.step()
                        optimizer.zero_grad()
                        global_step += 1

                    if (global_step + 1) % eval_step == 0:
                        logger.info('Epoch: {}, Step: {} / {}, used_time = {:.2f}s, loss = {:.6f}'.format(
                                     epoch, step + 1, len(train_batches),
                                     time.time() - start_time, tr_loss / nb_tr_steps))
                        save_model = False
                        if args.do_eval:
                            preds, result, _ = evaluate(model, device, eval_dataloader, eval_label_ids, num_labels)
                            model.train()
                            result['global_step'] = global_step
                            result['epoch'] = epoch
                            result['learning_rate'] = lr
                            result['batch_size'] = args.train_batch_size
                            logger.info("First 20 predictions:")
                            for pred, label in zip(preds[:20], eval_label_ids.numpy()[:20]):
                                sign = u'\u2713' if pred == label else u'\u2718'
                                logger.info("pred = %s, label = %s %s" % (id2label[pred], id2label[label], sign))
                            if (best_result is None) or (result[args.eval_metric] > best_result[args.eval_metric]):
                                best_result = result
                                save_model = True
                                logger.info("!!! Best dev %s (lr=%s, epoch=%d): %.2f" %
                                            (args.eval_metric, str(lr), epoch, result[args.eval_metric] * 100.0))
                                n_no_improve = 0
                            else:
                                n_no_improve += 1
                        else:
                            save_model = True

                        if save_model:
                            model_to_save = model.module if hasattr(model, 'module') else model
                            output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
                            output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
                            torch.save(model_to_save.state_dict(), output_model_file)
                            model_to_save.config.to_json_file(output_config_file)
                            tokenizer.save_vocabulary(args.output_dir)
                            if best_result:
                                output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
                                with open(output_eval_file, "w") as writer:
                                    for key in sorted(result.keys()):
                                        writer.write("%s = %s\n" % (key, str(result[key])))

                if n_no_improve >= args.patience:
                    logger.info("Patience Ended.")
                    break

    if args.do_eval:
        model = BertForSequenceClassification.from_pretrained(args.output_dir, num_labels=num_labels)
        if args.fp16:
            model.half()
        model.to(device)
        preds, result, confidences = evaluate(model, device, eval_dataloader, eval_label_ids, num_labels)
        with open(os.path.join(args.output_dir, "dev_predictions.json"), "w") as f:
            eval_results = []
            for ex, pred, confidence in zip(eval_examples, preds, confidences):
                eval_results.append({'guid': ex.guid, 'r': id2label[pred], 'c': float(confidence)})
            json.dump(eval_results, f)
        with open(os.path.join(args.output_dir, "dev_results.txt"), "w") as f:
            for key in sorted(result.keys()):
                f.write("%s = %s\n" % (key, str(result[key])))
        if args.eval_test:
            eval_examples = processor.get_test_examples(args.data_dir)
            eval_features = convert_examples_to_features(
                eval_examples, label2id, args.max_seq_length, tokenizer, special_tokens, args.feature_mode)
            logger.info("***** Test *****")
            logger.info("  Num examples = %d", len(eval_examples))
            logger.info("  Batch size = %d", args.eval_batch_size)
            all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
            all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
            eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
            eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size)
            eval_label_ids = all_label_ids

            preds, result, confidences = evaluate(model, device, eval_dataloader, eval_label_ids, num_labels)
            with open(os.path.join(args.output_dir, "test_predictions.json"), "w") as f:
                eval_results = []
                for ex, pred, confidence in zip(eval_examples, preds, confidences):
                    eval_results.append({'guid': ex.guid, 'r': id2label[pred], 'c': float(confidence)})
                json.dump(eval_results, f)
            with open(os.path.join(args.output_dir, "test_results.txt"), "w") as f:
                for key in sorted(result.keys()):
                    f.write("%s = %s\n" % (key, str(result[key])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, type=str, required=True)
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--relation_name", default=None, type=str, required=True, help="The relation name on which you want to do binary classification")
    parser.add_argument("--num_positive_examples", default=None, type=int, required=True, help="The number of positive examples allowed for classification")
    parser.add_argument("--ratio_negative_examples", default=None, type=int, required=True, help="The ratio of negative examples allowed for classification comparing to positive examples")
    parser.add_argument("--patience", default=-1, type=int, help="Patience for Early Stopping.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--eval_per_epoch", default=10, type=float,
                        help="How many times it evaluates on dev set per epoch")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--negative_label", default="no_relation", type=str)
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--train_mode", type=str, default='random_sorted', choices=['random', 'sorted', 'random_sorted'])
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--eval_test", action="store_true", help="Whether to evaluate on final test set.")
    parser.add_argument("--feature_mode", type=str, default="ner", choices=["text", "ner", "text_ner", "ner_text"])
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_metric", default="f1", type=str)
    parser.add_argument("--learning_rate", default=None, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    args = parser.parse_args()
    main(args)
