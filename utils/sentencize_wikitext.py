#!/usr/bin/env python3

import os
import argparse


def split_list(lst, delimiter):
    """Split a list into sublist by a delimiter."""
    idx = [i + 1 for i, e in enumerate(lst) if e == delimiter]
    idx = idx if not lst and idx[-1] != len(lst) else idx[:-1]
    return [lst[i:j] for i, j in zip([0] + idx, idx + [None])]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert wikitext into sentences.")
    parser.add_argument('--data', type=str, required=True,
                        help="Path to a directory containing wikitext.")
    parser.add_argument('--path', type=str, required=True,
                        help="Path to store the sentencized wikitext.")
    parser.add_argument('--min-length', type=int, default=3)
    parser.add_argument('--max-length', type=int, default=50)
    args = parser.parse_args()

    if not os.path.isdir(args.path):
        os.makedirs(args.path)

    for filename in ('wiki.train.tokens', 'wiki.valid.tokens', 'wiki.test.tokens'):
        num_line = 0
        num_line_keep = 0
        num_sent = 0
        num_token = 0
        num_sent_token = 0
        wikitext_path = os.path.join(args.data, filename)
        sent_path = os.path.join(args.path, filename)
        with open(wikitext_path, 'r', encoding='utf-8') as wikitext, open(sent_path, 'w+', encoding='utf-8') as sent:
            for line in wikitext:
                num_line += 1
                line = line.strip().split()  # Tokenize
                num_token += len(line)
                # Skip empty lines or lines not end with '.'
                # to avoid non-sentences, e.g., titles.
                if not line or line[-1] != '.':
                    continue
                num_line_keep += 1
                sent_lst = split_list(line, '.')  # Split list on '.' token
                for lst in sent_lst:
                    if args.min_length <= len(lst) <= args.max_length:
                        num_sent += 1
                        num_sent_token += len(lst)
                        sent_line = ' '.join(lst)
                        sent.write(sent_line + '\n')

        print(f"== File: {filename} ==\n"
              f"|  line count: {num_line} -> {num_line_keep}\n"
              f"|  sentence count: {num_sent}\n"
              f"|  token count: {num_token} -> {num_sent_token}")
