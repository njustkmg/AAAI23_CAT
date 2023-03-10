from misc.utils import *

class GatherLayer(ms.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        output = [ms.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = ms.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def gather_from_all(tensor: ms.Tensor) -> ms.Tensor:
    """
    Similar to classy_vision.generic.distributed_util.gather_from_all
    except that it does not cut the gradients
    """
    if tensor.ndim == 0:
        # 0 dim tensors cannot be gathered. so unsqueeze
        tensor = tensor.unsqueeze(0)

    if is_distributed_training_run():
        tensor, orig_device = convert_to_distributed_tensor(tensor)
        gathered_tensors = GatherLayer.apply(tensor)
        gathered_tensors = [
            convert_to_normal_tensor(_tensor, orig_device)
            for _tensor in gathered_tensors
        ]
    else:
        gathered_tensors = [tensor]
    gathered_tensor = ms.cat(gathered_tensors, 0)
    return gathered_tensor
