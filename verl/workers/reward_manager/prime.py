# Copyright 2024 PRIME team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import inspect
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Any, Callable, Optional

import psutil
import torch
from transformers import PreTrainedTokenizer

from verl import DataProto
from verl.utils.ray_utils import get_event_loop
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


def _maybe_debug_log_line(debug_log: dict) -> None:
    p = os.environ.get("ATTACK_OR_DEBUG_LOG")
    if not p:
        return
    try:
        with open(p, "a") as f:
            f.write(json.dumps(debug_log) + "\n")
    except Exception:
        pass


async def single_compute_score(evaluation_func, completion, reference, task, task_extra_info, executor, timeout=300.0, reward_kwargs=None):
    # Debug logging: Check function type
    debug_log = {
        "runId": "debug1",
        "hypothesisId": "A",
        "location": "prime.py:single_compute_score",
        "timestamp": int(time.time() * 1000)
    }
    
    # Check if evaluation_func is async (including when wrapped by partial)
    is_async = False
    try:
        func_to_check = evaluation_func.func if isinstance(evaluation_func, partial) else evaluation_func
        is_async = inspect.iscoroutinefunction(func_to_check)
        debug_log["message"] = "function_type_check"
        debug_log["data"] = {
            "is_async": is_async,
            "is_partial": isinstance(evaluation_func, partial),
            "func_name": getattr(func_to_check, "__name__", "unknown"),
            "completion_len": len(str(completion)[:200]),
            "has_extra_info": task_extra_info is not None
        }
        _maybe_debug_log_line(debug_log)
    except Exception:
        pass
    
    loop = get_event_loop()
    try:
        # If async function, await directly (don't use executor)
        if is_async:
            debug_log["message"] = "async_call"
            debug_log["data"] = {"calling_async": True}
            _maybe_debug_log_line(debug_log)
            
            # Pass reward_kwargs if provided
            if reward_kwargs:
                result = await asyncio.wait_for(
                    evaluation_func(task, completion, reference, task_extra_info, **reward_kwargs),
                    timeout=timeout
                )
            else:
                result = await asyncio.wait_for(
                    evaluation_func(task, completion, reference, task_extra_info),
                    timeout=timeout
                )
        else:
            # Sync function uses executor
            if reward_kwargs:
                # Use lambda to properly pass kwargs
                future = loop.run_in_executor(
                    executor, 
                    lambda: evaluation_func(task, completion, reference, task_extra_info, **reward_kwargs)
                )
            else:
                future = loop.run_in_executor(
                    executor, 
                    partial(evaluation_func, task, completion, reference, task_extra_info)
                )
            result = await asyncio.wait_for(future, timeout=timeout)
        
        # Debug logging: Check return value type
        debug_log["message"] = "result_check"
        debug_log["data"] = {
            "result_type": type(result).__name__,
            "is_coroutine": inspect.iscoroutine(result) if hasattr(inspect, 'iscoroutine') else False,
            "result_value": float(result) if isinstance(result, (int, float)) else str(result)[:100]
        }
        _maybe_debug_log_line(debug_log)
        
        return result
    except asyncio.TimeoutError:
        print(f"[Timeout] Task timeout: {completion}")
        return None  # Default value for timed-out rows
    except Exception as e:
        print(f"[Error] Task failed: {e}, completion: {completion[:80]}")
        debug_log["message"] = "exception"
        debug_log["data"] = {
            "error": str(e)[:200],
            "error_type": type(e).__name__
        }
        _maybe_debug_log_line(debug_log)
        return None  # Default value for failed rows


async def parallel_compute_score_async(
    evaluation_func, completions, references, tasks, extra_info=None, num_processes=64, reward_kwargs=None
):
    if extra_info is None:
        extra_info = [None] * len(tasks)
    scores = []
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # to prevent very occasional starvation caused by some anomalous programs ( like infinite loop ), the
        # exceptions in async programs will instantly halt the evaluation, and all summoned processes will be killed.
        try:
            # Create tasks for all rows
            tasks_async = [
                single_compute_score(evaluation_func, c, r, t, ei, executor, timeout=300.0, reward_kwargs=reward_kwargs)
                for c, r, t, ei in zip(completions, references, tasks, extra_info, strict=True)
            ]
            results = await asyncio.gather(*tasks_async, return_exceptions=False)
        except Exception as e:
            print(f"[Exception] async gather failed: {e}")
            raise
        finally:
            terminated_count = 0
            for pid, proc in executor._processes.items():
                try:
                    p = psutil.Process(pid)
                    p.terminate()
                    try:
                        p.wait(timeout=5)
                    except psutil.TimeoutExpired:
                        p.kill()
                    terminated_count += 1
                except Exception:
                    pass
            print(f"[Shutdown] {terminated_count} subprocess(es) terminated.")

    # Process results
    for result, completion, reference, task in zip(results, completions, references, tasks, strict=True):
        # Debug logging: Check result type
        debug_log = {
            "runId": "debug1",
            "hypothesisId": "A",
            "location": "prime.py:parallel_compute_score_async",
            "message": "processing_result",
            "timestamp": int(time.time() * 1000)
        }
        try:
            debug_log["data"] = {
                "result_type": type(result).__name__,
                "is_none": result is None,
                "is_exception": isinstance(result, Exception),
                "result_value": float(result) if isinstance(result, (int, float, bool)) else str(result)[:100]
            }
            _maybe_debug_log_line(debug_log)
        except Exception:
            pass
        
        if isinstance(result, Exception) or result is None:
            # Handle failed or timed-out tasks
            scores.append(0.0)
        elif isinstance(result, int | float | bool):
            scores.append(float(result))
        else:
            scores.append(float(result[0]))
    return scores


def run_reward_scoring(evaluation_func, completions, references, tasks, extra_info=None, num_processes=64, reward_kwargs=None):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(
            parallel_compute_score_async(evaluation_func, completions, references, tasks, extra_info, num_processes, reward_kwargs=reward_kwargs)
        )
    finally:
        loop.close()


@register("prime")
class PrimeRewardManager(AbstractRewardManager):
    """
    The Reward Manager used in https://github.com/PRIME-RL/PRIME
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        num_examine: int,
        compute_score: Optional[Callable] = None,
        reward_fn_key: str = "data_source",
        **reward_kwargs,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.reward_kwargs = reward_kwargs

    def verify(self, data):
        """
        verify the batch and save as ``acc`` tensor
        """
        # batched scoring
        prompt_ids = data.batch["prompts"]

        response_ids = data.batch["responses"]
        sequences_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        ground_truth = [data_item.non_tensor_batch["reward_model"]["ground_truth"] for data_item in data]
        data_sources = data.non_tensor_batch[self.reward_fn_key]
        extra_info = data.non_tensor_batch.get("extra_info", None)

        assert len(sequences_str) == len(ground_truth) == len(data_sources)
        try:
            scores = run_reward_scoring(
                self.compute_score,
                completions=sequences_str,
                references=ground_truth,
                tasks=data_sources,
                extra_info=extra_info,
                num_processes=64,
                reward_kwargs=self.reward_kwargs,
            )
        except asyncio.TimeoutError:
            print("[Timeout] Global reward scoring timed out. Setting all as 0.")
            scores = [0.0 for _ in range(len(sequences_str))]
        except Exception as e:
            print(f"[Error] Unexpected error during scoring. Setting all as 0. {e}")
            scores = [0.0 for _ in range(len(sequences_str))]
        data.batch["acc"] = torch.tensor(scores, dtype=torch.float32, device=prompt_ids.device)
        return scores

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        reward_from_rm_scores = self._extract_reward_from_rm_scores(data, return_dict)
        if reward_from_rm_scores is not None:
            return reward_from_rm_scores

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        already_print_data_sources = {}

        # batched scoring
        prompt_ids = data.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]

        response_ids = data.batch["responses"]
        valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(dim=-1)
        sequences_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        data_sources = data.non_tensor_batch["data_source"]

        scores = self.verify(data)

        for i in range(len(data)):
            data_source = data_sources[i]
            reward_tensor[i, valid_response_length[i].item() - 1] = scores[i]

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

        if return_dict:
            return {"reward_tensor": reward_tensor}
        else:
            return reward_tensor
