import sys, os, re
import json
import numpy as np
import math

with open("hans_heuristics_evaluation_set_relabeled.jsonl") as f:
  records = [json.loads(x) for x in f]

count = len(records)
sample_ratio = 0.02
sample_count = math.ceil(count * sample_ratio)

sample_indexes = np.random.choice(
  count,
  sample_count
)

sample_records = []
for sample_index in sample_indexes:
  sample_record = records[sample_index]
  sample_records.append(sample_record)

assert len(sample_records) == sample_count

with open("hans_rand_sample.jsonl", "w") as f:
  for record in sample_records:
    f.write(json.dumps(record) + "\n")

print("Sampled {} records from {} original records.".format(
  sample_count,
  count
))