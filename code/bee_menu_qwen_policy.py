"""Batched Qwen bee policy that chooses an item from a local feasible-action menu.

The model sees the same frozen world observation as its batch peers. The menu
is computed from that observation only; the researcher-only hive vector is
never included. Generation is token-constrained to one menu item.
"""
from __future__ import annotations

import json
from typing import Any

import torch

from bee_feasibility import feasible_actions
from bee_local_qwen_policy import BeeLocalQwenPolicy
from beehive_sim import SYSTEM_PROMPT


MENU_SYSTEM_PROMPT = SYSTEM_PROMPT + (
    "\nYour observation includes feasible_actions, a local list of executable actions. "
    "Choose exactly one item from that list and copy its six JSON fields exactly. "
    "The list does not contain the colony-wide hive vector or scores. "
    "All bees in a decision batch see one frozen state.\n"
)


class BeeMenuQwenPolicy(BeeLocalQwenPolicy):
    label = "local_qwen35_feasible_menu_constrained"

    def __init__(self, model, tokenizer, max_new_tokens: int = 160,
                 batch_size: int = 16):
        super().__init__(model, tokenizer, max_new_tokens, batch_size)
        self.last_menu_sizes: list[int] = []
        self.last_selected_indices: list[int] = []

    def _generate_chunk(self, observations: list[dict[str, Any]]):
        menus = [feasible_actions(obs) for obs in observations]
        enriched = [dict(obs, feasible_actions=menu) for obs, menu in zip(observations, menus)]
        conversations = [[
            {"role": "system", "content": MENU_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(obs, ensure_ascii=False,
                                                       separators=(",", ":"), allow_nan=False)},
        ] for obs in enriched]
        encoded = self.tokenizer.apply_chat_template(
            conversations, tokenize=True, add_generation_prompt=True,
            padding=True, return_tensors="pt", return_dict=True)
        inputs = {k: v.to(self.model.device) for k, v in encoded.items()
                  if isinstance(v, torch.Tensor)}
        prompt_width = inputs["input_ids"].shape[-1]
        input_counts = inputs["attention_mask"].sum(dim=-1).tolist()
        tokenizer = getattr(self.tokenizer, "tokenizer", self.tokenizer)
        eos_id = self.tokenizer.eos_token_id

        # One token trie per bee. A terminal node permits EOS only after an
        # entire JSON action; every valid leaf corresponds to a menu item.
        tries = []
        for menu in menus:
            root = {}
            for action in menu:
                text = json.dumps(action, ensure_ascii=False, separators=(",", ":"),
                                  allow_nan=False)
                ids = tokenizer.encode(text, add_special_tokens=False)
                node = root
                for token_id in ids:
                    node = node.setdefault(int(token_id), {})
                node[int(eos_id)] = {}
            tries.append(root)

        def allowed(batch_id, input_ids):
            generated = input_ids[prompt_width:].tolist()
            # Generation may keep querying a finished row while longer rows
            # continue. Keep its logits mask nonempty after its first EOS.
            if int(eos_id) in generated:
                return [int(eos_id)]
            node = tries[int(batch_id)]
            for token_id in generated:
                node = node.get(int(token_id))
                if node is None:
                    raise RuntimeError("Constrained generation left the local action menu")
            return list(node) or [int(eos_id)]

        with torch.inference_mode():
            generated = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=eos_id, use_cache=True,
                prefix_allowed_tokens_fn=allowed,
            )
        suffixes = generated[:, prompt_width:]
        texts = self.tokenizer.batch_decode(suffixes, skip_special_tokens=True)
        output_counts = []
        for row in suffixes:
            ids = row.tolist()
            output_counts.append(ids.index(eos_id) + 1 if eos_id in ids else len(ids))
        for text, menu in zip(texts, menus):
            action = json.loads(text)
            if action not in menu:
                raise RuntimeError("Generated action was not in its frozen local menu")
        self.last_menu_sizes.extend(len(menu) for menu in menus)
        return texts, input_counts, output_counts

    def decide_batch(self, observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self.last_menu_sizes = []
        return super().decide_batch(observations)
