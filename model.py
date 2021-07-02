# model.py

import math
import models
import losses
import evaluate
from torch import nn
import config

def weights_init(m):
    args = config.parse_args()
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, args.variance)
        if m.bias is not None:
            m.bias.data.zero_()


class Model:
    def __init__(self, args):
        self.ngpu = args.ngpu
        self.device = args.device
        self.adverserial_type=args.adverserial_type
        #########################################
        # encoder
        self.model_type_E = args.model_type_e
        self.model_type_EA = args.model_type_ea
        self.model_type_ET = args.model_type_et
        #########################################
        # targ-adv
        self.model_type_A = args.model_type_a
        self.model_type_T = args.model_type_t
        ###########################################
        # encoder
        self.model_options_E = args.model_options_e
        self.model_options_EA = args.model_options_EA
        self.model_options_ET = args.model_options_ET
        ###########################################
        # targ-adv
        self.model_options_A = args.model_options_A
        self.model_options_T = args.model_options_T
        ############################################
        # encoder
        self.loss_type_E = args.loss_type_e
        # self.loss_type_EA = args.loss_type_ea
        self.loss_type_ET = args.loss_type_et
        self.loss_type_ED = args.loss_type_ed
        ############################################
        self.loss_type_A = args.loss_type_a
        self.loss_type_T = args.loss_type_t
        #############################################
        # encoder
        self.loss_options_E = args.loss_options_E
        # self.loss_options_EA = args.loss_options_EA
        self.loss_options_ET = args.loss_options_ET
        self.loss_options_ED = args.loss_options_ED
        ##############################################
        self.loss_options_A = args.loss_options_A
        self.loss_options_T = args.loss_options_T
        ################################################
        # encoder
        self.evaluation_type_EA = args.evaluation_type_ea
        self.evaluation_type_ET = args.evaluation_type_et
        ################################################
        self.evaluation_type_A = args.evaluation_type_a
        self.evaluation_type_T = args.evaluation_type_t
        #######################################################
        # encoder
        self.evaluation_options_EA = args.evaluation_options_EA
        self.evaluation_options_ET = args.evaluation_options_ET
        ######################################################
        self.evaluation_options_A = args.evaluation_options_A
        self.evaluation_options_T = args.evaluation_options_T
        # import pdb
        # pdb.set_trace()

    def setup(self, checkpoints):

        # encoder
        model_E = getattr(models, self.model_type_E)(**self.model_options_E)
        model_EA = getattr(models, self.model_type_EA)(**self.model_options_EA)
        model_ET = getattr(models, self.model_type_ET)(**self.model_options_ET)
        #####################################################################
        model_A = getattr(models, self.model_type_A)(**self.model_options_A)
        model_T = getattr(models, self.model_type_T)(**self.model_options_T)
        ######################################################################
        # encoder
        if self.adverserial_type == 'OptNet':
            criterion_E = getattr(losses, self.loss_type_E)(**self.loss_options_E)
        # criterion_EA = getattr(losses, self.loss_type_EA)(**self.loss_options_EA)
        criterion_ET = getattr(losses, self.loss_type_ET)(**self.loss_options_ET)
        criterion_ED = getattr(losses, self.loss_type_ED)(**self.loss_options_ED)
        #######################################################################
        criterion_A = getattr(losses, self.loss_type_A)(**self.loss_options_A)
        criterion_T = getattr(losses, self.loss_type_T)(**self.loss_options_T)
        #######################################################################
        # encoder
        evaluation_EA = getattr(evaluate, self.evaluation_type_EA)(
            **self.evaluation_options_EA)
        evaluation_ET = getattr(evaluate, self.evaluation_type_ET)(
            **self.evaluation_options_ET)
        #######################################################################
        evaluation_A = getattr(evaluate, self.evaluation_type_A)(
            **self.evaluation_options_A)
        evaluation_T = getattr(evaluate, self.evaluation_type_T)(
            **self.evaluation_options_T)

        if self.ngpu > 1:
            #####################################################################
            # encoder
            model_E = nn.DataParallel(model_E, device_ids=list(range(self.ngpu)))
            model_EA = nn.DataParallel(model_EA, device_ids=list(range(self.ngpu)))
            model_ET = nn.DataParallel(model_ET, device_ids=list(range(self.ngpu)))
            ########################################################################
            model_A = nn.DataParallel(model_A, device_ids=list(range(self.ngpu)))
            model_T = nn.DataParallel(model_T, device_ids=list(range(self.ngpu)))

        #########################################################
        # encoder
        model_E = model_E.to(self.device)
        model_EA = model_EA.to(self.device)
        model_ET = model_ET.to(self.device)
        ##########################################################
        model_A = model_A.to(self.device)
        model_T = model_T.to(self.device)
        ###########################################################
        # encoder
        # criterion_E = criterion_E.to(self.device)
        # criterion_EA = criterion_EA.to(self.device)
        criterion_ET = criterion_ET.to(self.device)
        criterion_ED = criterion_ED.to(self.device)
        ###########################################################
        criterion_A = criterion_A.to(self.device)
        criterion_T = criterion_T.to(self.device)

        if checkpoints.latest('resume') is None:

            # model_A.apply(weights_init)
            # model_T.apply(weights_init)
            # model_E.apply(weights_init)
            # model_EA.apply(weights_init)
            # model_ET.apply(weights_init)
            pass

        else:
            ##################################################################
            # encoder
            model_E = checkpoints.load(model_E, checkpoints.latest('resume'))
            # model_EA = checkpoints.load(model_EA, checkpoints.latest('resume'))
            # model_ET = checkpoints.load(model_ET, checkpoints.latest('resume'))
            ###################################################################
            # model_A = checkpoints.load(model_A, checkpoints.latest('resume'))
            # model_T = checkpoints.load(model_T, checkpoints.latest('resume'))

        model ={}
        model['Encoder'] = model_E
        model['E-Adversary'] = model_EA
        model['E-Target'] = model_ET
        model['Adversary'] = model_A
        model['Target'] = model_T

        criterion = {}
        if self.adverserial_type == 'OptNet':
            criterion['Encoder'] = criterion_E
        # criterion['E-Adversary'] = criterion_EA
        criterion['E-Target'] = criterion_ET
        criterion['E-Discriminator'] = criterion_ED
        criterion['Adversary'] = criterion_A
        criterion['Target'] = criterion_T

        evaluation = {}
        evaluation['E-Adversary'] = evaluation_EA
        evaluation['E-Target'] = evaluation_ET
        evaluation['Adversary'] = evaluation_A
        evaluation['Target'] = evaluation_T

        return model, criterion, evaluation
