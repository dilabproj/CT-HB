import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from sklearn.cluster import KMeans


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, input_value, target, reduction='mean'):  # pylint: disable=arguments-differ
        if input_value.dim() > 2:
            # N,C,H,W => N,C,H*W
            input_value = input_value.view(input_value.size(0), input_value.size(1), -1)
            # N,C,H*W => N,H*W,C
            input_value = input_value.transpose(1, 2)
            # N,H*W,C => N*H*W,C
            input_value = input_value.contiguous().view(-1, input_value.size(2))
        target = target.view(-1, 1)

        logpt = F.log_softmax(input_value)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        loss = -1 * (1 - pt) ** self.gamma * logpt

        return loss.mean() if reduction == 'mean' else loss.sum()


class GWLoss(nn.Module):
    def __init__(self):  # pylint: disable=useless-super-delegation
        super(GWLoss, self).__init__()

    @staticmethod
    def gaussian(x, mean=0.5, variance=0.25):
        for i, v in enumerate(x.data):
            x[i] = math.exp(-(v - mean)**2 / (2.0 * variance**2))
        return x

    def forward(self, input_value, target, reduction='mean'):  # pylint: disable=arguments-differ

        if input_value.dim() > 2:
            input_value = input_value.view(input_value.size(0), input_value.size(1), -1)
            input_value = input_value.transpose(1, 2)
            input_value = input_value.contiguous().view(-1, input_value.size(2))
        target = target.view(-1, 1)

        logpt = F.log_softmax(input_value)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())
        loss = -1 * (self.gaussian(pt, variance=0.1 * math.exp(1), mean=0.5) - 0.1 * pt) * logpt
        return loss.mean() if reduction == 'mean' else loss.sum()


class MSESparseLoss(nn.Module):
    def __init__(self):  # pylint: disable=useless-super-delegation
        super(MSESparseLoss, self).__init__()

    def __sparse_loss(self, input_value, autoencoder):  # pylint: disable=no-self-use
        loss = 0
        values = input_value
        for i in range(len(list(autoencoder.encoder.children())) // 2):
            fc_layer = list(autoencoder.encoder.children())[2 * i]
            relu = list(autoencoder.encoder.children())[2 * i + 1]
            values = relu(fc_layer(values))
            loss += torch.mean(torch.abs(values))
        for i in range(len(list(autoencoder.decoder.children())) // 2):
            fc_layer = list(autoencoder.decoder.children())[2 * i]
            relu = list(autoencoder.decoder.children())[2 * i + 1]
            values = relu(fc_layer(values))
            loss += torch.mean(torch.abs(values))
        return loss

    def forward(self, input_value, target, autoencoder, sparse_reg=0.03):  # pylint: disable=arguments-differ
        return F.mse_loss(input_value, target) + sparse_reg * self.__sparse_loss(input_value, autoencoder)


class NCELoss(nn.Module):
    def __init__(self):  # pylint: disable=useless-super-delegation
        super(NCELoss, self).__init__()

    def forward(self, logit_anchor, logit_positive, logit_negative, reduction='mean'):  # pylint: disable=arguments-differ
        self.negative_output = logit_negative  # pylint: disable=attribute-defined-outside-init
        second_term = torch.sum(torch.log10(
            1 - self._logitprob_cal(logit_positive, self.negative_output, second_term=True)), dim=1)
        first_term = torch.log10(self._logitprob_cal(logit_anchor, logit_positive))
        loss = - first_term - second_term
        return loss.mean() if reduction == 'mean' else loss.sum()

    def _logitprob_cal(self, x, y, second_term=False, temperature=1):
        if second_term:
            a_norm = x / x.norm(dim=1)[:, None]
            b_norm = y / y.norm(dim=1)[:, None]
            pos_logits = torch.exp(torch.mm(a_norm, b_norm.transpose(0, 1)) / temperature)
            mask = np.ones((x.size(0), x.size(0)))
            ind = np.diag_indices(mask.shape[0])  # pylint: disable=unsubscriptable-object
            mask[ind[0], ind[1]] = torch.zeros(mask.shape[0])  # pylint: disable=unsubscriptable-object
            neg_cos_similarity = torch.mm(b_norm, b_norm.transpose(0, 1)) * torch.Tensor(mask).cuda(2)
            neg_logits = torch.sum(torch.exp(neg_cos_similarity) / temperature, dim=0)
            neg_logits = torch.transpose(neg_logits.unsqueeze(1).expand(*neg_logits.size(), neg_logits.size(0)), 1, 0)
            logit_prob = pos_logits / (pos_logits + neg_logits)
        else:
            pos_logits = torch.exp(F.cosine_similarity(x, y) / temperature)
            a_norm = y / y.norm(dim=1)[:, None]
            b_norm = self.negative_output / self.negative_output.norm(dim=1)[:, None]
            neg_logits = torch.exp(torch.mm(a_norm, b_norm.transpose(0, 1)) / temperature)
            logit_prob = pos_logits / (pos_logits + torch.sum(neg_logits, dim=1))

        return logit_prob


class MultiSimilarityLoss(nn.Module):
    def __init__(self, k, n_clusters=None):
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = 0.8
        self.margin = 0.05
        self.k = k
        if n_clusters is not None:
            self.kmeans = KMeans(n_clusters=n_clusters, n_init=100, n_jobs=24)

        self.scale_pos = 4
        self.scale_neg = 50

    def _deep_clustering(self, logit_anchor, logit_negative):
        X = np.concatenate((logit_anchor, logit_negative), axis=0)
        self.kmeans = self.kmeans.fit(X)
        return self.kmeans.labels_[:logit_anchor.shape[0]], self.kmeans.labels_[logit_anchor.shape[0]:]

    def forward(self, logit_anchor, logit_positive, logit_negative):  # pylint: disable=arguments-differ
        batch_size = logit_positive.size(0) // self.k
        logit_positive = logit_positive.view(batch_size, self.k, -1)

        # Compute Kmeans
        # anchor_labels, negative_labels = self._deep_clustering(
        #     logit_anchor.cpu().detach().numpy(), logit_negative.cpu().detach().numpy())

        epsilon = 1e-5
        loss = list()

        for i in range(batch_size):
            pos_pair_ = torch.matmul(logit_anchor[i], torch.t(logit_positive[i]))
            pos_pair_ = pos_pair_[pos_pair_ < 1 - epsilon]
            neg_pair_ = torch.matmul(logit_anchor[i], torch.t(logit_negative))
            # neg_pair_ = neg_pair_[negative_labels != anchor_labels[i]]

            neg_pair = neg_pair_[neg_pair_ + self.margin > min(pos_pair_)]
            pos_pair = pos_pair_[pos_pair_ - self.margin < max(neg_pair_)]

            if len(neg_pair) < 1 or len(pos_pair) < 1:
                continue

            # weighting step
            pos_loss = 1.0 / self.scale_pos * torch.log(
                1 + torch.sum(torch.exp(-self.scale_pos * (pos_pair - self.thresh))))
            neg_loss = 1.0 / self.scale_neg * torch.log(
                1 + torch.sum(torch.exp(self.scale_neg * (neg_pair - self.thresh))))
            loss.append(pos_loss + neg_loss)

        if len(loss) == 0:
            return torch.zeros([], requires_grad=True)

        loss = sum(loss) / batch_size
        return loss


class CosineSimilarityLoss(nn.Module):
    def __init__(self, k, device="0"):
        super(CosineSimilarityLoss, self).__init__()
        self.k = k
        self.device = int(device)

    def forward(self, logit_anchor, logit_strong_positive, logit_positive, reduction='mean'):  # pylint: disable=arguments-differ
        batch_size, size_representation = logit_anchor.size()
        first_term = - F.logsigmoid(torch.bmm(
            logit_anchor.view(batch_size, 1, size_representation),
            logit_strong_positive.view(batch_size, size_representation, 1))).squeeze(-1)
        second_term = - torch.mean(F.logsigmoid(torch.bmm(
            logit_anchor.view(batch_size, 1, size_representation),
            logit_positive.view(batch_size, size_representation, self.k))), dim=2)
        loss = first_term + second_term
        return loss.mean() if reduction == 'mean' else loss.sum()


class TripletLoss(nn.Module):
    def __init__(self, k=1):
        super(TripletLoss, self).__init__()
        self.k = k

    def forward(self, logit_anchor, logit_positive, logit_negative, reduction='mean'):  # pylint: disable=arguments-differ
        batch_size, size_representation = logit_anchor.size()
        first_term = - F.logsigmoid(torch.bmm(logit_anchor.view(batch_size, 1, size_representation),
                                              logit_positive.view(batch_size, size_representation, self.k))).squeeze()
        second_term = - torch.mean(F.logsigmoid(-torch.mm(logit_anchor, logit_negative.transpose(1, 0))), dim=1)
        loss = torch.mean(first_term, dim=1) + second_term
        return loss.mean() if reduction == 'mean' else loss.sum()


class TripletLoss_unsup(torch.nn.modules.loss._Loss):  # pylint: disable=protected-access
    """
    Triplet loss for representations of time series. Optimized for training
    sets where all time series have the same length.
    Takes as input a tensor as the chosen batch to compute the loss,
    a PyTorch module as the encoder, a 3D tensor (`B`, `C`, `L`) containing
    the training set, where `B` is the batch size, `C` is the number of
    channels and `L` is the length of the time series, as well as a boolean
    which, if True, enables to save GPU memory by propagating gradients after
    each loss term, instead of doing it after computing the whole loss.
    The triplets are chosen in the following manner. First the size of the
    positive and negative samples are randomly chosen in the range of lengths
    of time series in the dataset. The size of the anchor time series is
    randomly chosen with the same length upper bound but the the length of the
    positive samples as lower bound. An anchor of this length is then chosen
    randomly in the given time series of the train set, and positive samples
    are randomly chosen among subseries of the anchor. Finally, negative
    samples of the chosen length are randomly chosen in random time series of
    the train set.
    @param compared_length Maximum length of randomly chosen time series. If
           None, this parameter is ignored.
    @param nb_random_samples Number of negative samples per batch example.
    @param negative_penalty Multiplicative coefficient for the negative sample
           loss.
    """
    def __init__(self, compared_length, nb_random_samples, negative_penalty):
        super(TripletLoss_unsup, self).__init__()
        self.compared_length = compared_length
        if self.compared_length is None:
            self.compared_length = np.inf
        self.nb_random_samples = nb_random_samples
        self.negative_penalty = negative_penalty

    def forward(self, batch, encoder, negative, save_memory=False):  # pylint: disable=arguments-differ
        batch_size = batch.size(0)
        # train_size = train.size(0)
        length = min(self.compared_length, batch.size(2))

        # For each batch element, we pick nb_random_samples possible random
        # time series in the training set (choice of batches from where the
        # negative examples will be sampled)
        # samples = np.random.choice(
        #     train_size, size=(self.nb_random_samples, batch_size)
        # )
        # samples = torch.LongTensor(samples)

        # Choice of length of positive and negative samples
        length_pos_neg = np.random.randint(1, high=length + 1)

        # We choose for each batch example a random interval in the time
        # series, which is the 'anchor'
        random_length = np.random.randint(
            length_pos_neg, high=length + 1
        )  # Length of anchors
        beginning_batches = np.random.randint(
            0, high=length - random_length + 1, size=batch_size
        )  # Start of anchors

        # The positive samples are chosen at random in the chosen anchors
        beginning_samples_pos = np.random.randint(
            0, high=random_length - length_pos_neg + 1, size=batch_size
        )  # Start of positive samples in the anchors
        # Start of positive samples in the batch examples
        beginning_positive = beginning_batches + beginning_samples_pos
        # End of positive samples in the batch examples
        end_positive = beginning_positive + length_pos_neg

        # We randomly choose nb_random_samples potential negative samples for
        # each batch example
        beginning_samples_neg = np.random.randint(
            0, high=length - length_pos_neg + 1,
            size=(self.nb_random_samples, batch_size)
        )

        representation = encoder(torch.cat(
            [batch[
                j: j + 1, :,
                beginning_batches[j]: beginning_batches[j] + random_length
            ] for j in range(batch_size)]
        ))  # Anchors representations

        positive_representation = encoder(torch.cat(
            [batch[
                j: j + 1, :, end_positive[j] - length_pos_neg: end_positive[j]
            ] for j in range(batch_size)]
        ))  # Positive samples representations

        size_representation = representation.size(1)
        # Positive loss: -logsigmoid of dot product between anchor and positive
        # representations
        loss = -torch.mean(torch.nn.functional.logsigmoid(torch.bmm(
            representation.view(batch_size, 1, size_representation),
            positive_representation.view(batch_size, size_representation, 1)
        )))

        # If required, backward through the first computed term of the loss and
        # free from the graph everything related to the positive sample
        if save_memory:
            loss.backward(retain_graph=True)
            loss = 0
            del positive_representation
            torch.cuda.empty_cache()

        multiplicative_ratio = self.negative_penalty / self.nb_random_samples
        for i in range(self.nb_random_samples):
            # Negative loss: -logsigmoid of minus the dot product between
            # anchor and negative representations
            negative_representation = encoder(
                torch.cat([negative[:, i, :, :][
                    j: j + 1, :,
                    beginning_samples_neg[i, j]:
                    beginning_samples_neg[i, j] + length_pos_neg
                ] for j in range(batch_size)])
            )
            loss += multiplicative_ratio * -torch.mean(
                torch.nn.functional.logsigmoid(-torch.bmm(
                    representation.view(batch_size, 1, size_representation),
                    negative_representation.view(
                        batch_size, size_representation, 1
                    )
                ))
            )
            # If required, backward through the first computed term of the loss
            # and free from the graph everything related to the negative sample
            # Leaves the last backward pass to the training procedure
            if save_memory and i != self.nb_random_samples - 1:
                loss.backward(retain_graph=True)
                loss = 0
                del negative_representation
                torch.cuda.empty_cache()

        return loss


class TripletLossVaryingLength(torch.nn.modules.loss._Loss):  # pylint: disable=protected-access
    """
    Triplet loss for representations of time series where the training set
    features time series with unequal lengths.
    Takes as input a tensor as the chosen batch to compute the loss,
    a PyTorch module as the encoder, a 3D tensor (`B`, `C`, `L`) containing the
    training set, where `B` is the batch size, `C` is the number of channels
    and `L` is the maximum length of the time series (NaN values representing
    the end of a shorter time series), as well as a boolean which, if True,
    enables to save GPU memory by propagating gradients after each loss term,
    instead of doing it after computing the whole loss.
    The triplets are chosen in the following manner. First the sizes of
    positive and negative samples are randomly chosen in the range of lengths
    of time series in the dataset. The size of the anchor time series is
    randomly chosen with the same length upper bound but the the length of the
    positive samples as lower bound. An anchor of this length is then chosen
    randomly in the given time series of the train set, and positive samples
    are randomly chosen among subseries of the anchor. Finally, negative
    samples of the chosen length are randomly chosen in random time series of
    the train set.
    @param compared_length Maximum length of randomly chosen time series. If
           None, this parameter is ignored.
    @param nb_random_samples Number of negative samples per batch example.
    @param negative_penalty Multiplicative coefficient for the negative sample
           loss.
    """
    def __init__(self, compared_length, nb_random_samples, negative_penalty):
        super(TripletLossVaryingLength, self).__init__()
        self.compared_length = compared_length
        if self.compared_length is None:
            self.compared_length = np.inf
        self.nb_random_samples = nb_random_samples
        self.negative_penalty = negative_penalty

    def forward(self, batch, encoder, negative, device, save_memory=False):  # pylint: disable=arguments-differ
        batch_size = batch.size(0)
        max_length = batch.size(2)

        # For each batch element, we pick nb_random_samples possible random
        # time series in the training set (choice of batches from where the
        # negative examples will be sampled)
        # samples = np.random.choice(
        #     train_size, size=(self.nb_random_samples, batch_size)
        # )
        # samples = torch.LongTensor(samples)

        # Computation of the lengths of the relevant time series
        with torch.no_grad():
            lengths_batch = max_length - torch.sum(
                torch.isnan(batch[:, 0]), 1
            ).data.cpu().numpy()
            lengths_samples = np.empty(
                (self.nb_random_samples, batch_size), dtype=int
            )
            for i in range(self.nb_random_samples):
                lengths_samples[i] = max_length - torch.sum(
                    torch.isnan(negative[:, i, 0]), 1
                ).data.cpu().numpy()

        # Choice of lengths of positive and negative samples
        lengths_pos = np.empty(batch_size, dtype=int)
        lengths_neg = np.empty(
            (self.nb_random_samples, batch_size), dtype=int
        )
        for j in range(batch_size):
            lengths_pos[j] = np.random.randint(
                1, high=min(self.compared_length, lengths_batch[j]) + 1
            )
            for i in range(self.nb_random_samples):
                lengths_neg[i, j] = np.random.randint(
                    1,
                    high=min(self.compared_length, lengths_samples[i, j]) + 1
                )

        # We choose for each batch example a random interval in the time
        # series, which is the 'anchor'
        random_length = np.array([np.random.randint(
            lengths_pos[j],
            high=min(self.compared_length, lengths_batch[j]) + 1
        ) for j in range(batch_size)])  # Length of anchors
        beginning_batches = np.array([np.random.randint(
            0, high=lengths_batch[j] - random_length[j] + 1
        ) for j in range(batch_size)])  # Start of anchors

        # The positive samples are chosen at random in the chosen anchors
        # Start of positive samples in the anchors
        beginning_samples_pos = np.array([np.random.randint(
            0, high=random_length[j] - lengths_pos[j] + 1
        ) for j in range(batch_size)])
        # Start of positive samples in the batch examples
        beginning_positive = beginning_batches + beginning_samples_pos
        # End of positive samples in the batch examples
        end_positive = beginning_positive + lengths_pos

        # We randomly choose nb_random_samples potential negative samples for
        # each batch example
        beginning_samples_neg = np.array([[np.random.randint(
            0, high=lengths_samples[i, j] - lengths_neg[i, j] + 1
        ) for j in range(batch_size)] for i in range(self.nb_random_samples)])

        representation = torch.cat([encoder(
            batch[
                j: j + 1, :,
                beginning_batches[j]: beginning_batches[j] + random_length[j]
            ].to(device)
        ) for j in range(batch_size)])  # Anchors representations

        positive_representation = torch.cat([encoder(
            batch[
                j: j + 1, :,
                end_positive[j] - lengths_pos[j]: end_positive[j]
            ].to(device)
        ) for j in range(batch_size)])  # Positive samples representations

        size_representation = representation.size(1)
        # Positive loss: -logsigmoid of dot product between anchor and positive
        # representations
        loss = -torch.mean(torch.nn.functional.logsigmoid(torch.bmm(
            representation.view(batch_size, 1, size_representation),
            positive_representation.view(batch_size, size_representation, 1)
        )))

        # If required, backward through the first computed term of the loss and
        # free from the graph everything related to the positive sample
        if save_memory:
            loss.backward(retain_graph=True)
            loss = 0
            del positive_representation
            torch.cuda.empty_cache()

        multiplicative_ratio = self.negative_penalty / self.nb_random_samples
        for i in range(self.nb_random_samples):
            # Negative loss: -logsigmoid of minus the dot product between
            # anchor and negative representations
            negative_representation = torch.cat([encoder(
                negative[:, i, :, :][
                    j: j + 1, :,
                    beginning_samples_neg[i, j]:
                    beginning_samples_neg[i, j] + lengths_neg[i, j]
                ].to(device)
            ) for j in range(batch_size)])
            loss += multiplicative_ratio * -torch.mean(
                torch.nn.functional.logsigmoid(-torch.bmm(
                    representation.view(batch_size, 1, size_representation),
                    negative_representation.view(
                        batch_size, size_representation, 1
                    )
                ))
            )
            # If required, backward through the first computed term of the loss
            # and free from the graph everything related to the negative sample
            # Leaves the last backward pass to the training procedure
            if save_memory and i != self.nb_random_samples - 1:
                loss.backward(retain_graph=True)
                loss = 0
                del negative_representation
                torch.cuda.empty_cache()

        return loss
