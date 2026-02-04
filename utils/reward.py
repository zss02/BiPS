from verl.utils.reward_score import geo3k


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
):
    return geo3k.compute_score(solution_str, ground_truth, format_score=0.0)
