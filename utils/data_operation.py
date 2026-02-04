def pop_multimodal_data(batch, data_key="abl_multi_modal_data", inputs_key="abl_multi_modal_inputs"):
    """Extract edited data from batch (always returns both keys)."""
    return {
        "images": batch.non_tensor_batch.pop(data_key, None),
        "multi_modal_inputs": batch.non_tensor_batch.pop(inputs_key, None),
    }

def preprocess_aug_batch(batch, aug_data):
    """
    Swap in edited data while saving originals (to be restored later).
    Returns a dict with the originals.
    """
    images_aug = aug_data["images"]
    inputs_aug = aug_data["multi_modal_inputs"]

    # Save originals (may be None if absent)
    images_orig = batch.non_tensor_batch.pop("multi_modal_data", None)
    inputs_orig = batch.non_tensor_batch.pop("multi_modal_inputs", None)

    # Only replace if the original field existed (is not None)
    if images_orig is not None:
        batch.non_tensor_batch["multi_modal_data"] = images_aug
    if inputs_orig is not None:
        batch.non_tensor_batch["multi_modal_inputs"] = inputs_aug

    return {"images": images_orig, "multi_modal_inputs": inputs_orig}


def postprocess_aug_batch(batch, original_data):
    """
    Restore originals saved by preprocess_aug_batch.
    Uses `is not None` to avoid ndarray/Tensor truthiness errors.
    """
    if original_data["images"] is not None:
        _ = batch.non_tensor_batch.pop("multi_modal_data", None)  # discard edited
        batch.non_tensor_batch["multi_modal_data"] = original_data["images"]

    if original_data["multi_modal_inputs"] is not None:
        _ = batch.non_tensor_batch.pop("multi_modal_inputs", None)  # discard edited
        batch.non_tensor_batch["multi_modal_inputs"] = original_data["multi_modal_inputs"]
