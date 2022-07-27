#define Encoder
import torch
from torch import nn
import pytorch_lightning as pl
import numpy as np
from contextualized.functions import identity_link
torch.set_default_tensor_type(torch.FloatTensor)

#local imports
from graph_utils import project_to_dag_torch
from torch_utils import DAG_loss, NOTEARS_loss

class NGAM(nn.Module):
    """
    Neural Generalized Additive Model for NOTMAD
    """
    def __init__(self, input_dim, output_dim, width, n_hidden_layers, activation=nn.SiLU, link_fn=identity_link):
        """ Initialize the NGAM encoder.

        Args:
            input_dim (int, 1): Input shape of Encoder
            output_dim (int, ): Output shape of Encoder (weights for Explainer)
            width (int): Size of hidden layers
            n_hidden_layers (int): # of hidden layers
            activation (nn.Module: nn.SiLU): Activation function to use.
            link_fn (lambda: identity_link): Link function to use.
        """
        super(NGAM, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        nam_input_layer = nn.Linear(1, width)
        init_hidden_layer = nn.Linear(width, width)
        nam_output_layer = nn.Linear(width, output_dim)
        
        with torch.no_grad(): #use uniform randomization
            init_hidden_layer.weight = self._get_new_weights(init_hidden_layer)
            nam_input_layer.weight = self._get_new_weights(nam_input_layer)
            nam_output_layer.weight = self._get_new_weights(nam_output_layer)
        
        hidden_layers = lambda: [layer for layer in (init_hidden_layer, activation()) for _ in range(n_hidden_layers)]
        nam_layers = lambda: [nam_input_layer, activation()] + hidden_layers() + [nam_output_layer]
        
        self.nams =  nn.ModuleList([nn.Sequential(*nam_layers()) for _ in range(self.input_dim)])
        self.link_fn = link_fn

    def forward(self, x):
        batch_size = x.shape[0]
        ret = torch.zeros((batch_size, self.output_dim))
        for i, nam in enumerate(self.nams):
            ret += nam(x[:, i].unsqueeze(-1)) #batch at contextual feature i
        
        return self.link_fn(ret)
    
    def _get_new_weights(self, layer_):
        with torch.no_grad():
            return nn.parameter.Parameter(
                    torch.tensor(np.random.uniform(-0.01, 0.01, size=layer_.weight.shape)).float()
                )
    

class Explainer(nn.Module):
    """
    Explainer module for 2D archetypes
    """
    def __init__(self, archetypes_shape, init_mat=None):
        """ Initialize the Explainer.

        Args:
            archetypes_shape (int, int): Shape of archetypes (n_archetypes, # features)
            init_mat (np.array): 3D Custom initial weights for each archetype. Defaults to None.
        """
        super(Explainer, self).__init__()
        self.k, self.d = archetypes_shape

        if init_mat is not None:
            self.init_mat = torch.tensor(init_mat)
        else:
            self.init_mat = torch.tensor(np.random.uniform(-0.01, 0.01, size=(self.k, self.d, self.d))).float()
        
        self.archs = nn.parameter.Parameter(self.init_mat, requires_grad=True)
        self.mask = torch.ones(self.d).float() - torch.eye(self.d).float()
    
    def forward(self,batch_weights):
        batch_size = batch_weights.shape[0]
        archetypes = self._archetypes()
        
        batch_archetypes = torch.tensor(np.zeros((batch_size, self.d, self.d)))
        for i,batch_w in enumerate(batch_weights):
            batch_archetypes[i] = torch.tensordot(batch_w.float(), archetypes.float(), dims=1)
        
        return batch_archetypes
                
    
    def _archetypes(self):
        return torch.multiply(self.archs, self.mask)
    

class NOTMAD_model(pl.LightningModule):
    """
    NOTMAD model
    """
    def __init__(self, 
                 datamodule,
                 n_archetypes = 4,
                 sample_specific_loss_params = {'l1': 0., 'alpha': 1e-1, 'rho': 1e-2},
                 archetype_loss_params = {'l1': 0., 'alpha': 1e-1, 'rho': 1e-2},
                 opt_lr = 1e-3,
                 opt_step= 50,
                 n_encoder_hidden_layers=2,
                 encoder_width=32,
                 init_mat=None,
                 auto_opt=True,
                 encoder_type="NGAM"
            ):
        """ Initialize NOTMAD.
        
        Args:
            datamodule (pl.LightningDataModule): Lightning datamodule to use for training
        
        Kwargs:
            Explainer Kwargs
            ----------------
            init_mat (np.array): 3D Custom initial weights for each archetype. Defaults to None.
            n_archetypes (int:4): Number of archetypes in explainer
            
            Encoder Kwargs
            ----------------
            n_encoder_hidden_layers(int:2): Number encoder hidden layers
            encoder_width(int:32): Width of encoder hidden layers
            encoder_type (str: NGAM): Encoder module to use
            
            Optimization Kwargs
            -------------------
            auto_opt(bool: True): Use torch's backprop vs manual (customizable backprop in self.training_step)
            opt_lr(float): Optimizer learning rate
            opt_step(int): Optimizer step size
            
            Loss Kwargs
            -----------
            sample_specific_loss_params (dict of str: int): Dict of params used by NOTEARS loss (l1, alpha, rho)
            archetype_specific_loss_params (dict of str: int): Dict of params used by Archetype loss (l1, alpha, rho)

        """
        super(NOTMAD_model, self).__init__()

        #dataset params
        self.datamodule = datamodule
        self.n_archetypes = n_archetypes
        self.context_shape = self.datamodule.C_train.shape
        self.feature_shape = self.datamodule.X_train.shape

        #dag/loss params
        self.project_distance = 0.1
        self.archetype_loss_params = archetype_loss_params
        self.alpha, self.rho, self.use_dynamic_alpha_rho = self._parse_alpha_rho(sample_specific_loss_params)

        #torch params
        self.opt_lr = opt_lr
        self.opt_step = opt_step
        self.automatic_optimization = auto_opt
        self.encoder_type = encoder_type

        #layer shapes 
        encoder_input_shape = (self.context_shape[1], 1)
        encoder_output_shape = (self.n_archetypes, )
        arch_shape = (self.n_archetypes, self.feature_shape[-1])
        
        #layer params
        self.init_mat = init_mat

        #layers
        if self.encoder_type == "NGAM":
            self.encoder = NGAM(encoder_input_shape[0], encoder_output_shape[0],
                                n_hidden_layers=n_encoder_hidden_layers, 
                                width=encoder_width)
        else:
            self.encoder = self._build_encoder(encoder_input_shape[0], 
                                                encoder_output_shape[0])
        self.explainer = Explainer(arch_shape, init_mat=self.init_mat) 
        
        #loss
        self.my_loss = lambda x,y: self._build_arch_loss(archetype_loss_params) \
                                + NOTEARS_loss(x,y,sample_specific_loss_params['l1'],
                                        self.alpha,
                                        self.rho)

    def forward(self,c):
        c = c.float()
        c = self.encoder(c)
        out = self.explainer(c)
        return out.float()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.opt_lr)
        sch = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size  = self.opt_step , gamma = 0.5)
        # learning rate scheduler
        return {
            "optimizer":optimizer,
            "lr_scheduler" : {
                "scheduler" : sch,
                "monitor" : "train_loss",
                
            }
        }

    def training_step(self,batch,batch_idx):
        C , x_true = batch
        w_pred = self.forward(C).float()
        loss = self.my_loss(x_true.float(), w_pred.float()).float()
        self.log("train_loss", loss)

        if not self.automatic_optimization: #custom backwards
            opt = self.optimizers()
            opt.zero_grad()
            loss.backward()        
            opt.step()
        
        return loss
    
    def test_step(self,batch,batch_idx):
        C , x_true = batch
        w_pred = self.forward(C).float()
        loss = self.my_loss(x_true.float(), w_pred.float())
        self.log("test_loss", loss)
        return loss
    
    def validation_step(self,batch,batch_idx):
        C , x_true = batch
        w_pred = self.forward(C).float()
        # print(w_pred[0])
        loss = self.my_loss(x_true.float(), w_pred.float())
        self.log("val_loss", loss)
        return loss
    
    def predict_step(self, batch, batch_idx):
        C , x_true = batch
        w_pred = self.forward(C.float()).float()
        return w_pred

    def predict_w(self, C, confirm_project_to_dag=False):
        w_preds = self.forward(torch.tensor(C))

        if confirm_project_to_dag:
            try:
                return np.array([project_to_dag_torch(w.detach().numpy())[0] for w in w_preds])
            except:
                print("Error, couldn't project to dag. Returning normal predictions.")
        
        return w_preds

    #helpers
    def _parse_alpha_rho(self, params):
        alpha = params['alpha']
        rho = params['rho']
        dynamic = False
        return alpha, rho, dynamic

    def _build_arch_loss(self, params):
        archs = [self.explainer.archs[i] for i in range(self.explainer.k)]
        arch_loss = torch.sum(torch.tensor([
                params['l1']*torch.linalg.norm(archs[i], ord=1) + \
                DAG_loss(archs[i], 
                        alpha=params['alpha'],
                        rho=params['rho'], 
                    )
                for i in range(self.explainer.k)
            ]))
        return arch_loss
    
    def _build_encoder(self,in_dim,out_dim):
        #builds a linear encoder
        encoder = nn.Linear(in_dim, out_dim)
            
        with torch.no_grad(): #adjust weights to use uniform distribution
            self.encoder.weight =  torch.nn.parameter.Parameter(
                        torch.tensor(np.random.uniform(low=-0.01, high=0.01, size = self.encoder.weight.shape)).float(),
                        requires_grad=True
                        )
            self.encoder.bias =  torch.nn.parameter.Parameter(
                        torch.tensor(np.random.uniform(low=-0.01, high=0.01, size = self.encoder.bias.shape)).float(),
                        requires_grad=True
                        )
        
        return encoder