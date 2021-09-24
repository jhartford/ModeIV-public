import numpy as np
from scipy.stats import norm
import torch
from torch import dtype
import torch.utils.data


def eval_slopes(g_hat, datafn, training_treatment, ntest=50_000, seed=None):
    rng = np.random.RandomState(seed)

    x, z, t, y, g_true = datafn(ntest, rng.randint(100000), test=True)
    _, beta = g_true(x, z, t, return_beta=True)
    trange = np.percentile(training_treatment.flatten(), [25, 75])
    perf = []
    for b in np.unique(beta):
        xb = x[beta == b, :]
        x_ = np.repeat(xb, 2, axis=0)
        t_ = np.tile(trange, xb.shape[0])[:, None]
        z_ = np.repeat(
            z[np.random.choice(z.shape[0], xb.shape[0]), :], 2, axis=0)
        y_ = g_hat(t_, x_, z_).reshape((-1, 2))
        g_ave = ((y_[:, 0] - y_[:, 1]) / (trange[0] - trange[1])).mean()
        perf.append([(beta == b).mean(), b, g_ave, np.abs(b-g_ave)])
    return np.array(perf)


def monte_carlo_error(g_hat, data_fn, ntest=5000, has_latent=False, debug=False, redraw=True, trange=None):
    seed = np.random.randint(1e9)
    x, z, t, y, g_true = data_fn(ntest, seed, test=True)
    # re-draw to get new independent treatment and implied response
    if redraw:
        # redraw t from the intervention distribution
        if trange is None:
            t = np.linspace(np.percentile(t, 2.5), np.percentile(
                t, 97.5), ntest).reshape(-1, 1)
        else:
            t = np.linspace(trange[0], trange[1], ntest).reshape(-1, 1)
        y = g_true(x, z, t)

    y_true = y.flatten()
    y_hat = g_hat(x, z, t).flatten()  # TODO: change arg ordering to (t, x, z)
    return ((y_hat - y_true)**2).mean()


def one_hot(col, n_values):
    y = np.zeros((col.shape[0], n_values))
    y[np.arange(col.shape[0]), col] = 1
    return y


def sensf(x):
    return 2.0*((x - 5)**4 / 600 + np.exp(-((x - 5)/0.5)**2) + x/10. - 2)


def emocoef(emo):
    emoc = (emo * np.array([1., 2., 3., 4., 5., 6., 7.])[None, :]).sum(axis=1)
    return emoc


# demand sim constants
psd = 3.7
pmu = 17.779
ysd = 158.  # 292.
ymu = -292.1


class IVDataset:
    def __init__(self, x, z, t, y, validation=None, seed=None):
        rng = np.random.RandomState(seed)
        self.n = y.shape[0]
        if isinstance(validation, float):
            idx = rng.permutation(np.arange(self.n))
            train_idx = idx[0:int(n * (1-validation))]
            valid_idx = idx[int(n * (1-validation)):]
            validation = [i[valid_idx, ...]
                          for i in [x, z, t, y] if i is not None]
            train = [i[train_idx, ...] for i in [x, z, t, y] if i is not None]
        else:
            train = [x, z, t, y]

        self.has_features = x is not None
        self.instrument_dim = z.shape[1]
        self.feature_dim = x.shape[1] if x is not None else None
        self.treatment_dim = t.shape[1]
        self.response_dim = y.shape[1]

        self.dataset = torch.utils.data.TensorDataset(*[torch.from_numpy(i).float()
                                                        for i in train if i is not None])
        if validation is not None:
            self.validation = torch.utils.data.TensorDataset(*[torch.from_numpy(i).float()
                                                               for i in validation if i is not None])
        else:
            self.validation = None

    def get_dataloader(self, batch_size, shuffle=True):
        return torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle)

    def get_validation(self, batch_size=None, shuffle=False):
        if self.validation is None:
            raise ValueError("No validation set supplied")
        if batch_size is None:
            batch_size = self.validation.tensors[-1].shape[0]
        return torch.utils.data.DataLoader(self.validation, batch_size=batch_size, shuffle=shuffle)


class MultiInstrument(IVDataset):
    def __init__(self, n, datafunction, seed=None, validation=0.1, ntest=50_000):
        rng = np.random.RandomState(seed)
        self.n = n
        self.ntest = ntest
        self._validation_prop = validation
        self.datafunction = datafunction
        x, z, t, y, g_true = datafunction(n, rng.randint(int(1e6)), test=False)
        self.has_features = x is not None
        idx = rng.permutation(np.arange(n))
        self.train_idx = idx[0:int(n * (1-validation))]
        self.valid_idx = idx[int(n * (1-validation)):]

        self.training_data = {k: torch.from_numpy(i[self.train_idx, ...]).float() for k, i in zip(["feat", "inst", "treat", "response"],
                                                                                                  [x, z, t, y]) if i is not None}
        self._means = {'feat': 0.}
        self._std = {'feat': 1.}
        self._means.update(
            {k: t.mean() if k != 'treat' else 0. for k, t in self.training_data.items()})
        self._std.update(
            {k: t.std() if k != 'treat' else 1. for k, t in self.training_data.items()})
        self._means['inst'] = 0.
        self._std['inst'] = 1.
        # standardize training data
        self.training_data = {k: self._norm(
            i, self._means[k], self._std[k]) for k, i in self.training_data.items()}
        self.dataset = None

        self.validation_data = {k: torch.from_numpy(i[self.valid_idx, ...]).float() for k, i in zip(["feat", "inst", "treat", "response"],
                                                                                                    [x, z, t, y]) if i is not None}
        # standardize validation data
        self.validation_data = {k: self._norm(
            i, self._means[k], self._std[k]) for k, i in self.validation_data.items()}
        self.validation = torch.utils.data.TensorDataset(
            *self.validation_data.values())
        self.g_true = g_true
        self.instrument_dim = z.shape[1]
        self.feature_dim = x.shape[1] if x is not None else None
        self.treatment_dim = t.shape[1]
        self.response_dim = y.shape[1]

    def _norm(self, x, mean, sd):
        return x

    def evaluate(self, g_hat):
        return monte_carlo_error(g_hat, self.datafunction, ntest=self.ntest)

    def _make_features(self, G):
        G = np.array(G, dtype='int')
        x = np.zeros((G.shape[0], 0))
        for i in range(G.shape[1]):
            n = np.unique(G[:, i]).shape[0]
            col = np.zeros((G.shape[0], n))
            col[np.arange(G.shape[0]), G[:, i]] = 1
            x = np.concatenate([x, col], axis=1)
        return torch.from_numpy(x).float()

    def _prep_data(self, dataset, instrument_idx, cat_invalid, one_hot_z=True):
        y = dataset['response']
        t = dataset['treat']
        z = dataset['inst'][:, instrument_idx]
        if one_hot_z:
            z = self._make_features(z)
        not_inst = list(set(range(self.instrument_dim)
                            ).difference(instrument_idx))
        if len(not_inst) > 0 or 'feat' in dataset:
            feat = [dataset['feat']] if 'feat' in dataset else []
            if cat_invalid:
                invalid = dataset['inst'][:, not_inst]
                if one_hot_z:
                    invalid = self._make_features(invalid)
                feat += [invalid]
            if len(feat) > 0:
                x = torch.cat(feat, dim=1)
                self.has_features = True
            else:
                x = None
                self.has_features = True
        else:
            x = None
            self.has_features = False
        return [i for i in [x, z, t, y] if i is not None]

    def _bootstrap(self, data, i=None):
        n = data['response'].shape[0]
        idx = np.random.RandomState(i).choice(n, n)
        return {k: v[idx, ...] for k, v in data.items()}

    def get_dataloader(self, batch_size, shuffle=True, instrument_idx=None, cat_invalid=True, boot_index=None):
        if boot_index is None:
            training_data = self.training_data
        else:
            training_data = self._bootstrap(self.training_data, boot_index)
        if instrument_idx is None:
            dataset = torch.utils.data.TensorDataset(*training_data.values())
        else:
            ds = self._prep_data(training_data, instrument_idx,
                                 cat_invalid, one_hot_z=False)
            dataset = torch.utils.data.TensorDataset(*ds)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def get_validation(self, batch_size=None, shuffle=False, instrument_idx=None, cat_invalid=True):
        if self.validation is None:
            raise ValueError("No validation set supplied")
        if batch_size is None:
            batch_size = self.validation.tensors[-1].shape[0]
        if instrument_idx is None:
            dataset = self.validation
        else:
            dataset = torch.utils.data.TensorDataset(
                *self._prep_data(self.validation_data, instrument_idx, cat_invalid, one_hot_z=False))
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def relu(x):
    return np.maximum(0., x)


def get_beta(feat_weights, features, rng=None):
    if rng is None:
        rng = np.random.RandomState()
    beta = np.round(np.dot(features, feat_weights).flatten(), 1)
    beta = np.minimum(beta, 0.3)
    beta = np.maximum(beta, -0.3)
    return beta


def modal_paper(n=100000, beta=0.1, L=30, rho=15, sim=1,
                sig_u=0.78, sig_x=0.845, sig_y=0.905, seed=None,
                hetrogenous=False, feat_weights=None, valid=None,
                p=None, delta_y=None, delta_x=None, delta_u=None):
    rng = np.random.RandomState(seed)
    # which instruments are valid?
    valid = rng.permutation(
        np.array([1] * rho + [0] * (L-rho))) if valid is None else valid
    # generate G as per the paper
    p = rng.uniform(0.1, 0.9, size=(L)) if p is None else p
    #p = 0.5
    G = rng.binomial(2, p=p, size=(n, L))

    # constants from the paper
    theta_x = np.sqrt(0.3)
    theta_y = np.sqrt(0.3)
    gamma_x = np.sqrt(0.1)
    gamma_u = rho * np.sqrt(0.1) / L
    gamma_y = rho * np.sqrt(0.1) / L

    # simulation-dependent deltas
    delta_y = rng.uniform(0.01, 0.2, size=(L)) if delta_y is None else delta_y
    # what is the correct delta x? I can't find it in the paper
    delta_x = rng.uniform(0.01, 0.2, size=(L)) if delta_x is None else delta_x
    delta_u = rng.uniform(0.01, 0.2, size=(L)) if delta_u is None else delta_u
    if sim == 1:
        delta_y *= (1-valid)
        delta_u *= 0
    if sim == 2:
        delta_y *= 0
        delta_u *= (1-valid)
    z_u = np.dot(G, delta_u)
    # the "+ 1e-16" below is so you don't get div by zero errors when delta_u = 0 (and hence std = 0.).
    zu_cnst = (z_u.std() + 1e-16)
    z_u /= zu_cnst
    z_x = np.dot(G, delta_x)
    zx_cnst = (z_x.std() + 1e-16)
    z_x /= zx_cnst
    z_y = np.dot(G, delta_y)
    zy_cnst = (z_y.std() + 1e-16)
    z_y /= zy_cnst
    if hetrogenous:
        # hetrogenous treatment effect
        # Assume there are a sparse set of features
        # that effect the true beta.
        # For each true beta
        if feat_weights is None:
            print('feat weights none')
            n_feat = 20
            n_real = 3
            nonzeros = rng.permutation(
                np.array([0] * (n_feat - n_real) + [1] * n_real))
            feat_weights = rng.uniform(0.2, 1., (n_feat,)) * nonzeros
        n_groups = len(feat_weights)
        features = rng.uniform(-1., 1., (n, n_groups)) * 0.5
        beta = get_beta(feat_weights, features)
    else:
        features = None

    u = gamma_u * z_u + rng.randn(n) * sig_u
    x = gamma_x * z_x + theta_x * u + rng.randn(n) * sig_x
    y = gamma_y * z_y + beta * x.flatten() + theta_y * u + rng.randn(n) * sig_y
    if hetrogenous:
        def g(c, z, x, average=False, return_beta=False):
            '''
            return true structural function. Average = true
            averages over the direct effect the instruments.
            '''
            z_y_new = np.dot(z, delta_y)
            z_y_new /= zy_cnst
            beta = get_beta(feat_weights, c).flatten()
            y_ = beta * x.flatten()
            if average:
                return y_ + z_y.mean() * gamma_y
            else:
                if return_beta:
                    return y_ + gamma_y * z_y_new, beta
                else:
                    return y_ + gamma_y * z_y_new
    else:
        def g(c, z, x, average=False):
            z_y_new = np.dot(z, delta_y)
            z_y_new /= zy_cnst  # (z_y_new.std() + 1e-16)
            y_ = beta * x.flatten()
            if average:
                return y_ + z_y.mean() * gamma_y
            else:
                return y_ + gamma_y * z_y_new

    extra_info = {"valid": valid, "beta": beta,
                  "target": g,
                  "gamma_x": gamma_x, "gamma_y": gamma_y, "gamma_u": gamma_u,
                  "delta_u": delta_u, "delta_y": delta_y, "delta_x": delta_x}
    return features, x, y, G, extra_info


class Mendel(MultiInstrument):
    def __init__(self, n, n_inst=30, n_valid=30, seed=None, hetrogenous=False, beta=0.1, ntest=200_000, var_scale=1.,
                 use_one_hot_inst=False):
        rng = np.random.RandomState(seed+10)
        if hetrogenous:
            n_feat = 10
            n_real = 3
            nonzeros = rng.permutation(
                np.array([0] * (n_feat - n_real) + [1] * n_real))
            self.feat_weights = rng.uniform(0.2, 0.5, (n_feat,)) * nonzeros
        else:
            feat_weights = None
        self.valid = rng.permutation(
            np.array([1] * n_valid + [0] * (n_inst-n_valid)))
        self.p = rng.uniform(0.1, 0.9, size=(n_inst))
        self.delta_y = rng.uniform(0.01, 0.2, size=(n_inst))
        self.delta_x = rng.uniform(0.01, 0.2, size=(n_inst))
        self.delta_u = rng.uniform(0.01, 0.2, size=(n_inst))
        self.beta = beta

        def datafunction(n, s, test=False):
            x, p, y, G, extra = modal_paper(n=n, beta=self.beta, L=n_inst, seed=s, rho=n_valid,
                                            hetrogenous=hetrogenous, sig_y=var_scale, sig_x=var_scale, sig_u=var_scale,
                                            feat_weights=self.feat_weights, valid=self.valid, p=self.p, delta_x=self.delta_x,
                                            delta_y=self.delta_y, delta_u=self.delta_u)
            self.valid = extra['valid']
            if use_one_hot_inst:
                G = self._make_features(G)
            return x, G, p.reshape(-1, 1), y.reshape(-1, 1), extra['target']
        super(Mendel, self).__init__(n, datafunction, seed, ntest=ntest)

    def evaluate(self, g_hat):
        t_min, t_max = np.percentile(
            self.training_data['treat'].cpu().detach().numpy().flatten(), [2.5, 97.5])
        return monte_carlo_error(g_hat, self.datafunction, ntest=self.ntest, trange=[t_min, t_max])

    def evaluate_slopes(self, g_hat):
        return eval_slopes(g_hat, self.datafunction, self.dataset.tensors[-2].numpy())


class MultiDemand(MultiInstrument):
    def storeg(self, x, price, z=0, z_coeff=None, scale_exclusion=60.):
        emoc = emocoef(x[:, 1:])
        time = x[:, 0]
        g = sensf(time)*emoc*10. + (emoc*sensf(time)-2.0) * \
            (psd*price.flatten() + pmu)
        if z_coeff is not None:
            g += np.sin(2 * np.dot(z, z_coeff)) * scale_exclusion
        y = (g - ymu)/ysd
        return y.reshape(-1, 1)

    def demand(self, n, k, w_y, seed=1, ynoise=1., pnoise=1.,
               ypcor=0.8, scale_exclusion=60., use_images=False, test=False):
        rng = np.random.RandomState(seed)

        # covariates: time and emotion
        time = rng.rand(n) * 10
        emotion_id = rng.randint(0, 7, size=n)
        emotion = one_hot(emotion_id, n_values=7)
        if use_images:
            idx = np.argsort(emotion_id)
            emotion_feature = np.zeros((0, 28*28))
            for i in range(7):
                img = get_images(i, np.sum(emotion_id == i), seed, test)
                emotion_feature = np.vstack([emotion_feature, img])
            reorder = np.argsort(idx)
            emotion_feature = emotion_feature[reorder, :]
        else:
            emotion_feature = emotion

        # random instrument
        z = rng.randn(n, k)
        w = rng.uniform(0.5, 1.5, size=(k))

        # z -> price
        v = rng.randn(n)*pnoise
        price = sensf(time)*(np.dot(z, w) + 3) + 25.
        price = price + v
        price = (price - pmu)/psd

        # true observable demand function
        x = np.concatenate([time.reshape((-1, 1)), emotion_feature], axis=1)
        x_latent = np.concatenate([time.reshape((-1, 1)), emotion], axis=1)

        def g(x, z, p): return self.storeg(
            x, p, z, w_y, scale_exclusion=scale_exclusion)

        # errors
        e = (ypcor*ynoise/pnoise)*v + rng.randn(n)*ynoise*np.sqrt(1-ypcor**2)
        e = e.reshape(-1, 1)

        # response
        y = g(x_latent, z, price) + e

        return (x,
                z,  # * (w_v == 0)[None,:],
                price.reshape((-1, 1)),
                y.reshape((-1, 1)),
                g)

    def __init__(self, n, k, seed=None, ypcor=0.5, images=False, n_valid=2, scale_exclusion=240., scale_noise=1.):
        rng = np.random.RandomState(seed)
        self.w_y = rng.uniform(0.5, 1.5, size=(k))
        w_v = rng.permutation([1] * (k-n_valid) + [0] * (n_valid))
        self.w_y *= w_v
        self.valid = w_v == 0
        x, z, t, y, _ = self.demand(
            n=10000, k=k, w_y=self.w_y, seed=654321, ypcor=ypcor, use_images=False, test=False)
        self._means = {i: 0. for i, j in zip(
            ["feat", "inst", "treat", "response"], [x, z, t, y])}
        self._std = {i: 1. for i, j in zip(
            ["feat", "inst", "treat", "response"], [x, z, t, y])}

        def datafunction(n, s, test=False):
            return self.demand(n=n, k=k, w_y=self.w_y, seed=s,
                               ypcor=ypcor, use_images=images, test=test,
                               ynoise=scale_noise, pnoise=scale_noise, scale_exclusion=scale_exclusion)
        super(MultiDemand, self).__init__(n, datafunction, seed)
