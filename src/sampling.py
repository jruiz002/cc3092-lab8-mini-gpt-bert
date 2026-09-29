import torch
import torch.nn.functional as F

def sample_greedy(logits):
    """
    Selecciona siempre el token con mayor logit, sin aleatoriedad.
    """
    return torch.argmax(logits, dim=-1).item()

def sample_top_k(logits, k, tau=1.0):
    """
    Divide logits por tau, conserva solo los k mayores, aplica softmax y muestrea.
    """
    logits = logits / tau
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[..., [-1]]] = float('-inf')
    probs = F.softmax(out, dim=-1)
    return torch.multinomial(probs, 1).item()

def sample_top_p(logits, p, tau=1.0):
    """
    Divide por tau, aplica softmax, ordena de mayor a menor, incluye tokens 
    hasta que la probabilidad acumulada supere p, renormaliza y muestrea.
    """
    logits = logits / tau
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    # Tokens to remove (those whose cumulative probability is strictly > p)
    # But we shift right to keep the exact token that crosses the threshold p
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    # Scatter back to the original ordering
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )
    
    out = logits.clone()
    out[indices_to_remove] = float('-inf')
    probs = F.softmax(out, dim=-1)
    return torch.multinomial(probs, 1).item()
