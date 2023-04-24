from DTAN.smoothness_prior import smoothness_norm

# def alignment_loss(X_trasformed, labels, thetas, n_channels, model, smoothness_prior=True):
def alignment_loss(X_trasformed, labels, n_channels):
    '''
    Torch data format is  [N, C, W] W=timesteps
    Args:
        X_trasformed:
        labels:
        thetas:
        DTANargs:

    Returns:

    '''
    loss = 0
    # T = model.get_basis()
    prior_loss = 0
    n_classes = labels.unique()
    for i in n_classes:
        X_within_class = X_trasformed[labels==i]
        if n_channels == 1:
            # Single channel variance across samples
            loss = loss + X_within_class.var(dim=0, unbiased=False).mean()
        else:
            # variance between signalls in each channel (dim=0)
            # mean over each channel (dim=1)
            per_channel_loss = X_within_class.var(dim=0, unbiased=False).mean(dim=1)
            per_channel_loss = per_channel_loss.mean()
            loss = loss + per_channel_loss

    loss = loss / len(n_classes)

    return loss
