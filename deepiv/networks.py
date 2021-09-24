import torch
from deepiv.layers import MixtureDensityNetwork, BernoulliNetwork, MultinomialNetwork
from deepiv.losses import MixtureGaussianLoss, IV_MSELoss
import numpy as np
from tqdm import trange
import os
import time

def build_mlp(input_dim, hiddens, activation=torch.nn.ReLU, dropout_rate=0.):
    hiddens = [input_dim] + hiddens
    layers = []
    #layers += [torch.nn.BatchNorm1d(input_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)]
    for i,j in zip(hiddens[:-1], hiddens[1:]):
        layers += [torch.nn.Linear(i,j)]
        layers += [activation()]
        if dropout_rate > 0:
            layers += [torch.nn.Dropout(dropout_rate)]
    return torch.nn.Sequential(*layers)

class ResponseNetwork(torch.nn.Module):
    def __init__(self, representation, representation_dim, output_dim=1):
        super(ResponseNetwork, self).__init__()
        self.net = torch.nn.Sequential(representation, torch.nn.Linear(representation_dim, output_dim))
    
    def _prep_input(self, treatment, feature):
        if feature is not None:
            input = torch.cat([treatment, feature], dim=1)
        else:
            input = treatment
        return input

    def forward(self, treatment, feature=None):
        return self.net(self._prep_input(treatment, feature))

    def predict(self, treatment, feature=None):
        if isinstance(treatment, np.ndarray):
            treatment = torch.from_numpy(treatment).float()
        if feature is not None and isinstance(feature, np.ndarray):
            feature = torch.from_numpy(feature).float()
        output = self.forward(treatment, feature)
        return output

class LinearResponseNetwork(torch.nn.Module):
    def __init__(self, representation, representation_dim):
        super(LinearResponseNetwork, self).__init__()
        self.representation = representation
        self.treatment_coeff = torch.nn.Linear(representation_dim, 1)
        self.const = torch.nn.Linear(representation_dim, 1)

    def forward(self, treatment, feature=None):
        representation = self.representation(feature)
        return treatment * self.treatment_coeff(representation) + self.const(representation)

    def predict(self, treatment, feature=None):
        if isinstance(treatment, np.ndarray):
            treatment = torch.from_numpy(treatment).float()
        if feature is not None and isinstance(feature, np.ndarray):
            feature = torch.from_numpy(feature).float()
        output = self.forward(treatment, feature)
        return output

    def cate(self, treatment, feature=None):
        representation = self.representation(feature)
        return self.treatment_coeff(representation)

class TreatmentNetwork(torch.nn.Module):
    def __init__(self, instrument_dim, feature_dim=None, out_dim=1):
        super(TreatmentNetwork, self).__init__()
        self.instrument_dim = instrument_dim
        self.feature_dim = feature_dim if (isinstance(feature_dim, int) and feature_dim > 0) else None
        self._dim = instrument_dim + feature_dim if feature_dim is not None else instrument_dim
        self.net = None
    
    def _prep_input(self, instrument, feature):
        if self.feature_dim is not None and feature is None:
            raise RuntimeError("Features expected but not supplied (feature_dim is not None)")
        if feature is not None:
            input = torch.cat([instrument, feature], dim=1)
        else:
            input = instrument
        return input

    def forward(self, instrument, feature=None):
        return self.net(self._prep_input(instrument, feature))

    def sample(self, instrument, feature=None, n_samples=1):
        return self.net.sample(self._prep_input(instrument, feature), n_samples=n_samples)

    def expected_value(self, instrument, feature=None):
        return self.net.expected_value(self._prep_input(instrument, feature))

    def predicted_parameters(self, instrument, feature=None):
        return self.net.predicted_parameters(self._prep_input(instrument, feature))

    def get_loss_function(self):
        raise NotImplementedError()

class MixtureDensityTreatmentNetwork(TreatmentNetwork):
    def __init__(self, instrument_dim, feature_dim=None, out_dim=1, n_mixtures=10, hiddens=None, 
                 activation=torch.nn.ReLU, dropout_rate=0., representation=None, **kwargs):
        super(MixtureDensityTreatmentNetwork, self).__init__(instrument_dim, feature_dim)
        if hiddens is None:
            hiddens = [128, 64, 32]
        if representation is None:
            representation = build_mlp(self._dim, hiddens, activation, dropout_rate)
        self.net = MixtureDensityNetwork(representation, hiddens[-1], output_dim=out_dim, n_mixtures=n_mixtures)
        self.discrete = False

    def get_loss_function(self):
        return MixtureGaussianLoss()

class BernoulliTreatmentNetwork(TreatmentNetwork):
    def __init__(self, instrument_dim, feature_dim=None, out_dim=1, hiddens=None, 
                 activation=torch.nn.ReLU, dropout_rate=0., representation=None, **kwargs):
        super(BernoulliTreatmentNetwork, self).__init__(instrument_dim, feature_dim)
        if hiddens is None:
            hiddens = [128, 64, 32]
        if representation is None:
            representation = build_mlp(self._dim, hiddens, activation, dropout_rate)
        self.net = BernoulliNetwork(representation, hiddens[-1], output_dim=out_dim)
        self.discrete = True
    
    def get_loss_function(self):
        return torch.nn.BCEWithLogitsLoss()

class MultinomialTreatmentNetwork(TreatmentNetwork):
    def __init__(self, instrument_dim, feature_dim=None, out_dim=1, hiddens=None, 
                 activation=torch.nn.ReLU, dropout_rate=0., representation=None,
                 discretize=None, **kwargs):
        super(MultinomialTreatmentNetwork, self).__init__(instrument_dim, feature_dim)
        if hiddens is None:
            hiddens = [128, 64, 32]
        if representation is None:
            representation = build_mlp(self._dim, hiddens, activation, dropout_rate)
        self.net = MultinomialNetwork(representation, hiddens[-1], output_dim=out_dim)
        self.discrete = True
        self.discretizer = discretize
    
    def get_loss_function(self):
        if self.discretizer is None:
            return torch.nn.CrossEntropyLoss()
        else:
            ce = torch.nn.CrossEntropyLoss()
            def loss(pred, target):
                target_d = torch.from_numpy(self.discretizer(target.cpu().numpy())).to(target.device)
                return ce(pred, target_d.long().flatten())
            return loss

class DeepIV:
    def __init__(self, dataset, treatment_net="mixture_density", opt=None, response_net=None,
                 dropout_rate=None, weight_decay_treat=0., weight_decay_resp=0., 
                 biased=False, save_checkpoints=False, device="cpu",
                 treatment_representation=None, response_representation=None, hiddens=None, **kwargs):
        os.makedirs("./checkpoints", exist_ok=True)
        # heuristic to set dropout rate
        dropout_rate = min(1000./(1000. + dataset.n), 0.5) if dropout_rate is None else dropout_rate
        self.dataset = dataset
        self._means = [t.mean() for t in self.dataset.dataset.tensors]
        self._std = [t.std() for t in self.dataset.dataset.tensors]
        
        if isinstance(treatment_net, TreatmentNetwork):
            self.treatment_net = treatment_net
        elif isinstance(treatment_net, str):
            if treatment_net == "mixture_density":
                treatment_net = MixtureDensityTreatmentNetwork 
            elif treatment_net == "bernoulli":
                treatment_net = BernoulliTreatmentNetwork
            else:
                raise ValueError(f"Unrecognised treatment_net: {treatment_net}. "+
                                  "Must be one of ['mixture_density', 'bernoulli'] "+
                                  "or a custom TreatmentNetwork class.")

            self.treatment_net = treatment_net(dataset.instrument_dim, 
                                               dataset.feature_dim, 
                                               out_dim=dataset.treatment_dim,
                                               dropout_rate=dropout_rate, 
                                               representation=treatment_representation,
                                               hiddens=hiddens)
        else:
            raise ValueError("Treatment network can't be None")
        
        self.device = device
        self.treatment_net.to(device)
        
        if response_net is not None:
            self.response_net = response_net
        else:
            if response_representation is None:
                hiddens = [128, 64, 32] if hiddens is None else hiddens
                input_dim = dataset.treatment_dim + dataset.feature_dim if dataset.feature_dim is not None else dataset.treatment_dim
                representation = build_mlp(input_dim, hiddens, dropout_rate=dropout_rate)
            self.response_net = ResponseNetwork(representation, hiddens[-1])
        self.response_net.to(device)
        
        if opt is None:
            self.opt_treat = torch.optim.Adam(self.treatment_net.parameters(), weight_decay=weight_decay_treat)
            self.opt_response = torch.optim.Adam(self.response_net.parameters(), weight_decay=weight_decay_resp)
        else:
            self.opt_treat = opt(treatment_net.parameters(), weight_decay=weight_decay_treat)
            self.opt_response = opt(response_net.parameters(), weight_decay=weight_decay_resp)
        
        self.loss_treat = self.treatment_net.get_loss_function()
        self.loss_response = IV_MSELoss() if not biased else torch.nn.MSELoss()
        self.mse = torch.nn.MSELoss()
        self.biased = biased
        self._name = time.ctime(time.time()).replace(" ","_").replace(":","-") + "-" + str(np.random.randint(1000))
        self.save_checkpoints = save_checkpoints 

    def to(self, device):
        self.treatment_net.to(device)
        self.response_net.to(device)

    def _norm(self, x, mean, sd):
        return (x-mean) / sd

    def _denorm(self, y, mean, sd):
        return y * sd + mean

    def validate_treatment(self):
        dataloader = self.dataset.get_validation(None)
        with torch.no_grad():
            self.treatment_net.eval()
            total_loss = 0
            n_steps = 0
            for batch in dataloader:
                batch = [b.to(self.device) for b in batch]
                batch = [self._norm(i, m, s) for i, m, s in zip(batch, self._means, self._std)]
                if self.dataset.has_features:
                    x_train, z_train, target, _ = batch
                else:
                    z_train, target, _ = batch
                    x_train = None
                treatment_outputs = self.treatment_net(z_train, x_train)
                loss = self.loss_treat(treatment_outputs, target)
                total_loss += float(loss)
                n_steps += 1
            ave_loss = total_loss / float(n_steps)
        return ave_loss

    def validate_response(self, n_samples=100):
        dataloader = self.dataset.get_validation(100)
        with torch.no_grad():
            total_loss = 0
            n_steps = 0
            self.response_net.eval()
            for batch in dataloader:
                batch = [b.to(self.device) for b in batch]
                batch = [self._norm(i, m, s) for i, m, s in zip(batch, self._means, self._std)]
                if self.dataset.has_features:
                    x_train, z_train, _, target = batch
                else:
                    z_train, _, target = batch
                    x_train = None
                treatment_samples = self.treatment_net.sample(z_train, x_train, n_samples)
                prediction = self.response_net(treatment_samples.view(z_train.shape[0] * n_samples, -1), 
                                               x_train[None, :, :].repeat(n_samples, 1, 1).view(z_train.shape[0] * n_samples, -1) 
                                               if x_train is not None else None)
                prediction = prediction.view(n_samples, z_train.shape[0], -1).mean(dim=0)
                loss = self.mse(prediction, target)
                total_loss += float(loss)
                n_steps += 1
            ave_loss = total_loss / float(n_steps)
        return ave_loss
    
    def fit_treatment(self, epochs, batch_size=100):
        min_val = np.inf
        dataloader = self.dataset.get_dataloader(batch_size)
        with trange(epochs, desc="Treatment") as pbar:
            for ep in pbar:
                total_loss = 0
                n_steps = 0
                self.treatment_net.train()
                for batch in dataloader:
                    batch = [b.to(self.device) for b in batch]
                    batch = [self._norm(i, m, s) for i, m, s in zip(batch, self._means, self._std)]
                    if self.dataset.has_features:
                        x_train, z_train, target, _ = batch
                    else:
                        z_train, target, _ = batch
                        x_train = None
                    self.opt_treat.zero_grad()
                    treatment_outputs = self.treatment_net(z_train, x_train)
                    loss = self.loss_treat(treatment_outputs, target)
                    loss.backward()
                    self.opt_treat.step()
                    total_loss += float(loss)
                    n_steps += 1
                ave_loss = total_loss / float(n_steps)
                validation = self.validate_treatment()
                if validation < min_val:
                    min_val = validation
                    torch.save(self.treatment_net.state_dict(), f"checkpoints/{self._name}-treatment.pt")
                pbar.set_postfix(loss=ave_loss, validation=validation, min_val=min_val)
        self.treatment_net.load_state_dict(torch.load(f"checkpoints/{self._name}-treatment.pt"))
        self.treatment_net.eval()
        if not self.save_checkpoints:
            os.remove(f"checkpoints/{self._name}-treatment.pt")

    def fit_response(self, epochs, batch_size=100):
        dataloader = self.dataset.get_dataloader(batch_size)
        min_val = np.inf
        with trange(epochs, desc="Response") as pbar:
            self.response_net.train()
            for ep in pbar:
                total_loss = 0
                n_steps = 0
                for batch in dataloader:
                    batch = [b.to(self.device) for b in batch]
                    batch = [self._norm(i, m, s) for i, m, s in zip(batch, self._means, self._std)]
                    if self.dataset.has_features:
                        x_train, z_train, _, target = batch
                    else:
                        z_train, _, target = batch
                        x_train = None
                    self.opt_response.zero_grad()
                    treatment_samples = self.treatment_net.sample(z_train, x_train, 2 if not self.biased else 1)
                    prediction = self.response_net(treatment_samples[0,:], x_train)
                    if not self.biased:
                        with torch.no_grad():
                            samples = self.response_net(treatment_samples[1,:], x_train)
                        loss = self.loss_response(prediction, target, samples)
                    else:
                        loss = self.loss_response(prediction, target)
                    loss.backward()
                    self.opt_response.step()
                    total_loss += float(loss)
                    n_steps += 1
                ave_loss = total_loss / float(n_steps)
                validation = self.validate_response()
                if validation < min_val:
                    min_val = validation
                    torch.save(self.response_net.state_dict(), f"checkpoints/{self._name}-response.pt")
                pbar.set_postfix(loss=ave_loss, validation=validation, min_val=min_val)
        self.response_net.load_state_dict(torch.load(f"checkpoints/{self._name}-response.pt"))
        self.response_net.eval()
        if not self.save_checkpoints:
            os.remove(f"checkpoints/{self._name}-response.pt")

    def fit(self, epochs_response=None, epochs_treatment=None, batch_size=100):
        epochs = int(1500000./float(self.dataset.n)) # heuristic to set the number of epochs automatically
        if epochs_response is None:
            epochs_response = epochs
        if epochs_treatment is None:
            epochs_treatment =  epochs

        self.fit_treatment(epochs_treatment, batch_size)
        self.fit_response(epochs_response, batch_size)

    def predict(self, treatment, features):
        treatment = torch.from_numpy(treatment).float().to(self.device)
        if features is not None:
            features = torch.from_numpy(features).float().to(self.device)
            feat = self._norm(features, 
                          self._means[0], self._std[0])
        else:
            feat = None
        pred = self.response_net.predict(self._norm(treatment, 
                                                    self._means[-2], self._std[-2]), feat)
        return self._denorm(pred, self._means[-1], self._std[-1]).cpu().detach().numpy()


def one_hot(col, n_values):
    y = np.zeros((col.shape[0], n_values))
    y[np.arange(col.shape[0]), col] = 1
    return y

class DeepIVMulti:
    def __init__(self, dataset, treatment_net="mixture_density", opt=None, response_net=None,
                 dropout_rate=None, weight_decay_treat=0., weight_decay_resp=0., 
                 biased=False, save_checkpoints=False, device="cpu",
                 treatment_representation=None, response_representation=None, hiddens=None, 
                 valid_inst=None, discretizer=None, uid="", linear_reponse=False,
                 **kwargs):
        os.makedirs("./checkpoints", exist_ok=True)
        # heuristic to set dropout rate
        dropout_rate = min(1000./(1000. + dataset.n), 0.5) if dropout_rate is None else dropout_rate
        self.dataset = dataset
        self.valid_inst = valid_inst
        if valid_inst is None:
            feature_dim = dataset.feature_dim
            instrument_dim = dataset.instrument_dim
        else:
            feature_dim = dataset.feature_dim if dataset.feature_dim is not None else 0
            feature_dim += dataset.instrument_dim - len(valid_inst)
            instrument_dim = len(valid_inst)
        
        self._means1 = self.dataset._means#[t.mean() for t in self.dataset.dataset.tensors]
        self._means = list(self.dataset._means.values())
        #print(self._means1, self._means)
        self._std1 = self.dataset._std#[t.std() for t in self.dataset.dataset.tensors]
        self._std = list(self.dataset._std.values())
        #print([t.shape for t in self.dataset.dataset.tensors])
        if discretizer is None:
            self.discretizer = lambda x:x
        else:
            self.discretizer = discretizer
        if isinstance(treatment_net, TreatmentNetwork):
            self.treatment_net = treatment_net
        elif isinstance(treatment_net, str):
            if treatment_net == "mixture_density":
                treatment_net = MixtureDensityTreatmentNetwork 
            elif treatment_net == "bernoulli":
                treatment_net = BernoulliTreatmentNetwork
            elif treatment_net == "multinomial":
                treatment_net = MultinomialTreatmentNetwork
            else:
                raise ValueError(f"Unrecognised treatment_net: {treatment_net}. "+
                                  "Must be one of ['mixture_density', 'bernoulli'] "+
                                  "or a custom TreatmentNetwork class.")

            self.treatment_net = treatment_net(instrument_dim, 
                                               feature_dim, 
                                               out_dim=dataset.treatment_dim,
                                               dropout_rate=dropout_rate, 
                                               representation=treatment_representation,
                                               hiddens=hiddens, 
                                               discretize=discretizer)
            if opt is None:
                self.opt_treat = torch.optim.Adam(self.treatment_net.parameters(), weight_decay=weight_decay_treat)
            else:
                self.opt_treat = opt(treatment_net.parameters(), weight_decay=weight_decay_treat)
        else:
            raise ValueError("Treatment network can't be None")
        
        self.device = device
        self.treatment_net.to(device)
        
        if response_net is not None:
            self.response_net = response_net
        else:
            if response_representation is None:
                hiddens = [128, 64, 32] if hiddens is None else hiddens
                input_dim = feature_dim if feature_dim is not None else dataset.treatment_dim
                if not linear_reponse:
                    input_dim += dataset.treatment_dim
                representation = build_mlp(input_dim, hiddens, dropout_rate=dropout_rate)
            if linear_reponse:
                self.response_net = LinearResponseNetwork(representation, hiddens[-1])
            else:
                self.response_net = ResponseNetwork(representation, hiddens[-1])
        self.response_net.to(device)
        
        if opt is None:
            self.opt_response = torch.optim.Adam(self.response_net.parameters(), weight_decay=weight_decay_resp)
        else:
            self.opt_response = opt(response_net.parameters(), weight_decay=weight_decay_resp)
        
        self.loss_treat = self.treatment_net.get_loss_function()
        self.loss_response = IV_MSELoss() if not biased else torch.nn.MSELoss()
        self.mse = torch.nn.MSELoss()
        self.biased = biased
        self._name = str(uid) + time.ctime(time.time()).replace(" ","_").replace(":","-") + "-" + str(np.random.randint(1000))
        self.save_checkpoints = save_checkpoints 

    def to(self, device):
        self.treatment_net.to(device)
        self.response_net.to(device)

    def _norm(self, x, mean, sd):
        #print(mean, sd)
        return (x-mean) / sd

    def _norm1(self, x, mean, sd):
        #print(mean, sd)
        #return x
        return (x-mean) / sd

    def _denorm(self, y, mean, sd):
        #return y
        return y * sd + mean

    def validate_treatment(self):
        dataloader = self.dataset.get_validation(None, instrument_idx=self.valid_inst)
        with torch.no_grad():
            self.treatment_net.eval()
            total_loss = 0
            n_steps = 0
            for batch in dataloader:
                batch = [b.to(self.device) for b in batch]
                batch = [self._norm(i, m, s) for i, m, s in zip(batch, self._means, self._std)]
                if self.dataset.has_features:
                    x_train, z_train, target, _ = batch
                else:
                    z_train, target, _ = batch
                    x_train = None
                treatment_outputs = self.treatment_net(z_train, x_train)
                loss = self.loss_treat(treatment_outputs, target)
                total_loss += float(loss)
                n_steps += 1
            ave_loss = total_loss / float(n_steps)
        return ave_loss

    def validate_response(self, n_samples=100):
        dataloader = self.dataset.get_validation(100, instrument_idx=self.valid_inst)
        with torch.no_grad():
            total_loss = 0
            n_steps = 0
            self.response_net.eval()
            for batch in dataloader:
                batch = [b.to(self.device) for b in batch]
                batch = [self._norm(i, m, s) for i, m, s in zip(batch, self._means, self._std)]
                if self.dataset.has_features:
                    x_train, z_train, _, target = batch
                else:
                    z_train, _, target = batch
                    x_train = None
                if self.treatment_net.discrete:
                    treatment_probs = self.treatment_net.predicted_parameters(z_train, x_train)
                    treatments = torch.eye(treatment_probs.shape[1]).to(z_train.device)
                    treat_inp = treatments.repeat(x_train.shape[0], 1)
                    feat_inp = torch.repeat_interleave(x_train, treatment_probs.shape[1], dim=0) if x_train is not None else None
                    pred = self.response_net(treat_inp,  feat_inp)
                    pred = pred.view(x_train.shape[0], treatment_probs.shape[1], 1)
                    expected_value = (pred * treatment_probs[:,:,None]).sum(axis=1)
                    loss = self.loss_response(expected_value.flatten(), target.flatten())
                else:
                    treatment_samples = self.treatment_net.sample(z_train, x_train, n_samples)
                    prediction = self.response_net(treatment_samples.view(z_train.shape[0] * n_samples, -1), 
                                                x_train[None, :, :].repeat(n_samples, 1, 1).view(z_train.shape[0] * n_samples, -1) 
                                                if x_train is not None else None)
                    prediction = prediction.view(n_samples, z_train.shape[0], -1).mean(dim=0)
                    loss = self.loss_response(prediction, target)
                total_loss += float(loss)
                n_steps += 1
            ave_loss = total_loss / float(n_steps)
        return ave_loss
    
    def fit_treatment(self, epochs, batch_size=100, boot_index=None):
        if hasattr(self.treatment_net, "fit_with_dataset"):
            self.treatment_net.fit_with_dataset(self.dataset)
        else:
            min_val = np.inf
            dataloader = self.dataset.get_dataloader(batch_size, instrument_idx=self.valid_inst, boot_index=boot_index)
            with trange(epochs, desc="Treatment") as pbar:
                for ep in pbar:
                    total_loss = 0
                    n_steps = 0
                    self.treatment_net.train()
                    for batch in dataloader:
                        batch = [b.to(self.device) for b in batch]
                        batch = [self._norm(i, m, s) for i, m, s in zip(batch, self._means, self._std)]
                        if self.dataset.has_features:
                            x_train, z_train, target, _ = batch
                        else:
                            z_train, target, _ = batch
                            x_train = None
                        self.opt_treat.zero_grad()
                        treatment_outputs = self.treatment_net(z_train, x_train)
                        loss = self.loss_treat(treatment_outputs, target)
                        loss.backward()
                        self.opt_treat.step()
                        total_loss += float(loss)
                        n_steps += 1
                    ave_loss = total_loss / float(n_steps)
                    validation = self.validate_treatment()
                    if validation < min_val:
                        min_val = validation
                        torch.save(self.treatment_net.state_dict(), f"checkpoints/{self._name}-treatment.pt")
                    pbar.set_postfix(loss=ave_loss, validation=validation, min_val=min_val)
            self.treatment_net.load_state_dict(torch.load(f"checkpoints/{self._name}-treatment.pt"))
            self.treatment_net.eval()
            if not self.save_checkpoints:
                os.remove(f"checkpoints/{self._name}-treatment.pt")

    def fit_response(self, epochs, batch_size=100, boot_index=None):
        dataloader = self.dataset.get_dataloader(batch_size, instrument_idx=self.valid_inst, boot_index=boot_index)
        min_val = np.inf
        with trange(epochs, desc="Response") as pbar:
            self.response_net.train()
            for ep in pbar:
                total_loss = 0
                n_steps = 0
                for batch in dataloader:
                    batch = [b.to(self.device) for b in batch]
                    batch = [self._norm(i, m, s) for i, m, s in zip(batch, self._means, self._std)]
                    if self.dataset.has_features:
                        x_train, z_train, _, target = batch
                    else:
                        z_train, _, target = batch
                        x_train = None
                    self.opt_response.zero_grad()
                    if self.treatment_net.discrete:
                        treatment_probs = self.treatment_net.predicted_parameters(z_train, x_train)
                        treatments = torch.eye(treatment_probs.shape[1]).to(z_train.device)
                        treat_inp = treatments.repeat(x_train.shape[0], 1)
                        feat_inp = torch.repeat_interleave(x_train, treatment_probs.shape[1], dim=0) if x_train is not None else None
                        pred = self.response_net(treat_inp,  feat_inp)
                        pred = pred.view(x_train.shape[0], treatment_probs.shape[1], 1)
                        expected_value = (pred * treatment_probs[:,:,None]).sum(axis=1)
                        loss = self.loss_response(expected_value.flatten(), target.flatten())
                    else:
                        treatment_samples = self.treatment_net.sample(z_train, x_train, 2 if not self.biased else 1)
                        prediction = self.response_net(treatment_samples[0,:], x_train)
                        if not self.biased:
                            with torch.no_grad():
                                samples = self.response_net(treatment_samples[1,:], x_train)
                            loss = self.loss_response(prediction, target, samples)
                        else:
                            loss = self.loss_response(prediction, target)
                    loss.backward()
                    self.opt_response.step()
                    total_loss += float(loss)
                    n_steps += 1
                ave_loss = total_loss / float(n_steps)
                validation = self.validate_response()
                if validation < min_val:
                    min_val = validation
                    torch.save(self.response_net.state_dict(), f"checkpoints/{self._name}-response.pt")
                pbar.set_postfix(loss=ave_loss, validation=validation, min_val=min_val)
        self.response_net.load_state_dict(torch.load(f"checkpoints/{self._name}-response.pt"))
        self.response_net.eval()
        if not self.save_checkpoints:
            os.remove(f"checkpoints/{self._name}-response.pt")

    def fit(self, epochs_response=None, epochs_treatment=None, batch_size=100, boot_index=None):
        epochs = int(1500000./float(self.dataset.n)) # heuristic to set the number of epochs automatically
        if epochs_response is None:
            epochs_response = epochs
        if epochs_treatment is None:
            epochs_treatment =  epochs

        self.fit_treatment(epochs_treatment, batch_size, boot_index=boot_index)
        self.fit_response(epochs_response, batch_size, boot_index=boot_index)

    def gradient(self, treatment, features, instruments=None):
        if self.valid_inst is not None:
            k = instruments.shape[1]
            not_inst = list(set(range(k)).difference(self.valid_inst))
            x = [features] if features is not None else []
            features = np.concatenate(x + [instruments[:, not_inst]], axis=1)
            instruments = instruments[:,self.valid_inst]
        treatment = torch.tensor(treatment, requires_grad=True).float().to(self.device)
        if features is not None:
            features = torch.from_numpy(features).float().to(self.device)
            feat = self._norm1(features, 
                          self._means1['feat'], self._std1['feat'])
        else:
            feat = None
        pred = self.response_net.predict(self._norm1(treatment, 
                                                    self._means1['treat'], self._std1['treat']), feat)
        grad = torch.autograd.grad(pred, treatment, torch.ones_like(pred))[0]
        return grad.cpu().detach().numpy() * self._std1['feat']
    
    def _predict(self, treatment, features, instruments=None):
        if self.valid_inst is not None:
            k = instruments.shape[1]
            not_inst = list(set(range(k)).difference(self.valid_inst))
            x = [features] if features is not None else []
            features = torch.cat(x + [instruments[:, not_inst]], dim=1)
            instruments = instruments[:,self.valid_inst]
        
        if features is not None:
            feat = self._norm1(features, 
                          self._means1['feat'], self._std1['feat'])
        else:
            feat = None
        self.response_net.eval()
        pred = self.response_net.predict(self._norm1(treatment, 
                                                    self._means1['treat'], self._std1['treat']), feat)
        #print(pred.mean(), pred.std(), self._means1['response'], self._std1['response'])
        return self._denorm(pred, self._means1['response'], self._std1['response'])

    def predict(self, treatment, features, instruments=None, numpy=True):
        if isinstance(treatment, np.ndarray):
            if self.treatment_net.discrete:
                treatment = np.array(self.discretizer(treatment), dtype='int')
                treatment = one_hot(treatment.flatten(), 6)
            treatment = torch.from_numpy(treatment).float().to(self.device)
        else:
            if self.treatment_net.discrete:
                treatment = np.array(self.discretizer(treatment.cpu().detach().numpy()), dtype='int')
                treatment = torch.from_numpy(one_hot(treatment.flatten(), 6))
                treatment = treatment.float().to(self.device)
            else:
                treatment = treatment.float().to(self.device)
        if isinstance(instruments, np.ndarray):
            instruments = torch.from_numpy(instruments).float().to(self.device)
        elif instruments is not None:
            instruments = instruments.float().to(self.device)
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features).float().to(self.device)
        elif features is not None:
            features = features.float().to(self.device)
        pred = self._predict(treatment, features, instruments=instruments)
        if numpy:
            return pred.cpu().detach().numpy()
        else:
            return pred

    def _cate(self, treatment, features, instruments=None):
        if self.valid_inst is not None:
            k = instruments.shape[1]
            not_inst = list(set(range(k)).difference(self.valid_inst))
            x = [features] if features is not None else []
            features = torch.cat(x + [instruments[:, not_inst]], dim=1)
            instruments = instruments[:,self.valid_inst]
        
        if features is not None:
            feat = self._norm1(features, 
                          self._means1['feat'], self._std1['feat'])
        else:
            feat = None
        self.response_net.eval()
        pred = self.response_net.cate(self._norm1(treatment, 
                                                    self._means1['treat'], self._std1['treat']), feat)
        #print(pred.mean(), pred.std(), self._means1['response'], self._std1['response'])
        return pred * self._std1['response']

    def cate(self, treatment, features, instruments=None, numpy=True):
        if not hasattr(self.response_net, "cate"):
            print("Response net doesn't have a cate attribute")
            return None
        if isinstance(treatment, np.ndarray):
            if self.treatment_net.discrete:
                treatment = np.array(self.discretizer(treatment), dtype='int')
                treatment = one_hot(treatment.flatten(), 6)
            treatment = torch.from_numpy(treatment).float().to(self.device)
        else:
            if self.treatment_net.discrete:
                treatment = np.array(self.discretizer(treatment.cpu().detach().numpy()), dtype='int')
                treatment = torch.from_numpy(one_hot(treatment.flatten(), 6))
                treatment = treatment.float().to(self.device)
            else:
                treatment = treatment.float().to(self.device)
        if isinstance(instruments, np.ndarray):
            instruments = torch.from_numpy(instruments).float().to(self.device)
        elif instruments is not None:
            instruments = instruments.float().to(self.device)
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features).float().to(self.device)
        elif features is not None:
            features = features.float().to(self.device)
        pred = self._cate(treatment, features, instruments=instruments)
        if numpy:
            return pred.cpu().detach().numpy()
        else:
            return pred

        

