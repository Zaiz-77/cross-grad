from math import sqrt


def compute_gradient_cosine(grad1, grad2):
    dot_product = sum((g1 * g2).sum() for g1, g2 in zip(grad1, grad2))
    norm1 = sqrt(sum((g * g).sum() for g in grad1))
    norm2 = sqrt(sum((g * g).sum() for g in grad2))
    return dot_product / (norm1 * norm2)


def decompose_gradient(grad_mnist, grad_usps):
    dot_product = sum((g1 * g2).sum() for g1, g2 in zip(grad_mnist, grad_usps))
    norm_usps_square = sum((g * g).sum() for g in grad_usps)

    scale = dot_product / norm_usps_square
    h = [g * scale for g in grad_usps]
    v = [g1 - g2 for g1, g2 in zip(grad_mnist, h)]

    return h, v