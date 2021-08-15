#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .base_extractor import BaseExtractor


class PointerSequenceExtractor(BaseExtractor):
    def __init__(self):
        super(PointerSequenceExtractor, self).__init__()

    def decode(self, **kwarg):
        start_logits = kwarg['start_logits']
        end_logits = kwarg['end_logits']
        lens = kwarg['lens']
        if confidence in kwarg:
            confidence = kwarg['confidence']
        if overlap in kwarg:
            overlap = kwarg['overlap']
        else:
            overlap = False

        num_tokens = lens[0]
        S = []

        starts = torch.argmax(start_logits, -1).cpu().numpy()[0]  #[1:-1]
        ends = torch.argmax(end_logits, -1).cpu().numpy()[0]  #[1:-1]

        # start_logits.shape: (1, sentence_length, num_labels)
        #  logger.debug(f"start_logits: {start_logits.shape}")
        #  start_max_probs = [
        #      f"{start_logits[0][i][x]:.4f}" for i, x in enumerate(starts)
        #  ]
        #  logger.debug(f"start_max_probs: {start_max_probs}")
        #  logger.debug(f"starts: {starts.shape} {starts}")
        #  starts = [
        #      x if start_logits[0][i][x] >= confidence else 0
        #      for i, x in enumerate(starts)
        #  ]
        #  ends = [
        #      x if end_logits[0][i][x] >= confidence else 0
        #      for i, x in enumerate(ends)
        #  ]

        #  starts = [starts[0]] + [ x if x != starts[i] else 0 for i, x in enumerate(starts[1:])]
        #  ends = [ends[0]] + [ x if x != ends[i] else 0 for i, x in enumerate(ends[1:])]
        def filter_process(starts):
            new_starts = []
            for i, x in enumerate(starts):
                is_dup = False
                if i < len(starts) - 1 and x == starts[i + 1]:
                    is_dup = True
                elif i > 0 and starts[i - 1] == x:
                    is_dup = True
                if is_dup:
                    new_starts.append(0)
                else:
                    new_starts.append(x)
            return new_starts

        #  starts = filter_process(starts)
        #  ends = filter_process(ends)

        #  starts = np.array([x for x in starts if x >= 0 and x < num_tokens])
        #  ends = np.array([x for x in ends if x >= 0 and x < num_tokens])
        starts = starts[:num_tokens]
        ends = ends[:num_tokens]

        #  logger.info(f"start_pred: {starts}")
        #  logger.info(f"end_pred: {ends}")
        #  for i, s_l in enumerate(starts):
        #      if s_l == 0:
        #          continue
        #      for j, e_l in enumerate(ends[i:]):
        #          if s_l == e_l:
        #              S.append((s_l, i, i + j))
        #              break
        #          if i + j < len(starts) - 1 and starts[i + j + 1] != 0:
        #              break
        #  for i in range(len(starts) - 1):
        last_j = -1
        for i in range(len(starts)):
            if i <= last_j:
                continue
            s_l = starts[i]
            if s_l == 0:
                continue
            for j, e_l in enumerate(ends[i:]):
                if s_l == e_l:
                    if not overlap:
                        #  if sum(starts[i + 1:i + j + 1]) == 0:
                        S.append((int(s_l), i, i + j))
                        last_j = j
                        i = j + 1
                        break
                    else:
                        if sum(starts[i + 1:i + j + 1]) != 0:
                            break
                        S.append((int(s_l), i, i + j))
                if i + j < len(starts) - 1 and starts[i + j + 1] != 0:
                    break
        #  S = [x for x in S if x[1] <= x[2]]

        #  for x in S:
        #      assert x[1] >= 0 and x[2] >= 0 and x[1] <= x[2], f"S: {S}"

        return S
