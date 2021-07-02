
import torch


__all__ = ['Projection', 'Projection_poly', 'Projection_IMQ', 'Projection_gauss', 'Projection_gauss_linear', 'Projection_ideal']

class Projection_ideal:
    def __init__(self):
        # self.eye = torch.eye(1).to(device)
        pass

    def __call__(self, S, Y, lam, reg, device):


        S_bar = S - torch.mean(S, dim=0)

        Y_bar = Y - torch.mean(Y, dim=0)

        # Project_Y = (torch.norm(torch.mm(P_M, Y_bar)) ** 2 )/batch_size

        B = lam*torch.mm(S_bar, torch.t(S_bar)) - (1-lam)*torch.mm(Y_bar, torch.t(Y_bar))
        U, Sig, V = torch.eig(B)
        # import pdb; pdb.set_trace()
        # loss = lam*Project_S / (1-lam)*Project_Y

        return  U


class Projection:
    def __init__(self):
        # self.eye = torch.eye(1).to(device)
        pass

    def __call__(self, Z, S, Y, lam, reg, device):
        # import pdb;
        # pdb.set_trace()
        # if lam>0.2:
        # Z = Z / torch.norm(Z, 'fro')
        # Z = Z / torch.max(Z)
        M = Z - torch.mean(Z, dim=0) # Z is already transposed
        # import pdb; pdb.set_trace()
        # M = M/torch.norm(M,'fro')

        batch_size = Z.size(0)

        P1 = torch.mm(torch.t(M), M)
        # self.eye.resize_(P1.shape[0])

        P2 = torch.inverse(P1+ reg*torch.eye(P1.shape[0]).to(device))
        # P2 = torch.inverse(P1+ reg*self.eye)
        P3 = torch.mm(P2, torch.t(M))
        P_M = torch.mm(M, P3)

        # U, Sigma, V = torch.svd(M)
        # import pdb; pdb.set_trace()
        # P_M = torch.mm(U, torch.t(U))

        S_bar = S - torch.mean(S, dim=0)
        Project_S = (torch.norm(torch.mm(P_M, S_bar)) ** 2 + reg*torch.norm(torch.mm(P3, S_bar)) ** 2)/torch.norm(S_bar) ** 2
        # Project_S = (torch.norm(torch.mm(P_M, S_bar)) ** 2)/batch_size

        Y_bar = Y - torch.mean(Y, dim=0)
        # import pdb; pdb.set_trace()
        Project_Y = (torch.norm(torch.mm(P_M, Y_bar)) ** 2  + reg*torch.norm(torch.mm(P3, Y_bar)) ** 2)/torch.norm(Y_bar) ** 2
        # Project_Y = (torch.norm(torch.mm(P_M, Y_bar)) ** 2 )/batch_size

        loss = lam*Project_S - (1-lam)*Project_Y
        # import pdb; pdb.set_trace()
        # loss = lam*Project_S / (1-lam)*Project_Y

        return  loss, Project_S, Project_Y


class Projection_poly:
    def __init__(self):
        # self.eye = torch.eye(1).to(device)
        pass

    def __call__(self, Z, S, Y, lam, reg, device, c, d):
        # import pdb; pdb.set_trace()

        # M = Z - torch.mean(Z, dim=0) # Z is already transposed
        # Z = Z / torch.norm(Z, 'fro')

        # Z = Z / torch.max(Z)
        K = torch.mm(Z,torch.t(Z)) # Z is already transposed


        # K = torch.matrix_power(K+c*torch.eye(K.shape[0]).to(device),d)

        # K = torch.pow(K+c*torch.eye(K.shape[0]).to(device),d)
        K = torch.pow(K+c,d)
        batch_size = Z.size(0)
        # M = M/torch.norm(M,'fro')

        D = torch.eye(batch_size) - torch.ones([batch_size, batch_size])/batch_size
        D = D.to(device)
        M = torch.mm(torch.mm(D,K), D)

        # import pdb; pdb.set_trace()

        P1 = torch.mm(torch.t(M), M)
        # self.eye.resize_(P1.shape[0])

        # import pdb; pdb.set_trace()
        P2 = torch.inverse(P1+ reg*torch.eye(P1.shape[0]).to(device))
        # P2 = torch.inverse(P1+ reg*self.eye)
        P3 = torch.mm(P2, torch.t(M))
        P_M = torch.mm(M, P3)

        # U, Sigma, V = torch.svd(M)
        # import pdb; pdb.set_trace()
        # P_M = torch.mm(U, torch.t(U))

        S_bar = S - torch.mean(S, dim=0)
        Project_S = (torch.norm(torch.mm(P_M, S_bar)) ** 2 + reg*torch.norm(torch.mm(P3, S_bar)) ** 2)/torch.norm(S_bar) ** 2
        # Project_S = (torch.norm(torch.mm(P_M, S_bar)) ** 2 )/batch_size

        Y_bar = Y - torch.mean(Y, dim=0)
        Project_Y = (torch.norm(torch.mm(P_M, Y_bar)) ** 2 + reg*torch.norm(torch.mm(P3, Y_bar)) ** 2)/torch.norm(Y_bar) ** 2
        # Project_Y = (torch.norm(torch.mm(P_M, Y_bar)) ** 2)/batch_size

        loss = lam*Project_S - (1-lam)*Project_Y
        # import pdb; pdb.set_trace()
        # loss = lam*Project_S / (1-lam)*Project_Y

        return  loss, Project_S, Project_Y

class Projection_IMQ:
    def __init__(self):
        # self.eye = torch.eye(1).to(device)
        pass

    def __call__(self, Z, S, Y, lam, reg, device, c):
        # import pdb; pdb.set_trace()

        # M = Z - torch.mean(Z, dim=0) # Z is already transposed
        Z = Z / torch.norm(Z, 'fro')
        # import pdb;
        # pdb.set_trace()
        # Z = Z / torch.max(Z)
        batch_size = Z.size(0)
        ONES = torch.ones([1,batch_size]).to(device)
        NORM = torch.norm(Z,dim=1).reshape([1,batch_size])
        NORM = NORM ** 2
        K = c* torch.pow(torch.mm(torch.t(NORM), ONES) + torch.mm(torch.t(ONES), NORM) + 2*torch.mm(Z, torch.t(Z))+c, -1)
        # import pdb; pdb.set_trace()

        # K = torch.matrix_power(K+c*torch.eye(K.shape[0]).to(device),d)
        # K = torch.pow(K+c,d)
        # M = M/torch.norm(M,'fro')



        D = torch.eye(batch_size) - torch.ones([batch_size, batch_size])/batch_size
        D = D.to(device)
        M = torch.mm(torch.mm(D,K), D)

        # import pdb; pdb.set_trace()

        P1 = torch.mm(torch.t(M), M)
        # self.eye.resize_(P1.shape[0])

        # import pdb; pdb.set_trace()
        P2 = torch.inverse(P1+ reg*torch.eye(P1.shape[0]).to(device))
        # P2 = torch.inverse(P1+ reg*self.eye)
        P3 = torch.mm(P2, torch.t(M))
        P_M = torch.mm(M, P3)

        # U, Sigma, V = torch.svd(M)
        # import pdb; pdb.set_trace()
        # P_M = torch.mm(U, torch.t(U))

        S_bar = S - torch.mean(S, dim=0)
        Project_S = (torch.norm(torch.mm(P_M, S_bar)) ** 2 + reg*torch.norm(torch.mm(P3, S_bar)) ** 2)/torch.norm(S_bar) ** 2
        # Project_S = torch.norm(torch.mm(P_M, S_bar)) ** 2 /torch.norm(S_bar) ** 2
        # Project_S = (torch.norm(torch.mm(P_M, S_bar)) ** 2 )/batch_size

        Y_bar = Y - torch.mean(Y, dim=0)
        Project_Y = (torch.norm(torch.mm(P_M, Y_bar)) ** 2 + reg*torch.norm(torch.mm(P3, Y_bar)) ** 2)/torch.norm(Y_bar) ** 2
        # Project_Y = torch.norm(torch.mm(P_M, Y_bar)) ** 2 /torch.norm(Y_bar) ** 2
        # Project_Y = (torch.norm(torch.mm(P_M, Y_bar)) ** 2)/batch_size

        loss = lam*Project_S - (1-lam)*Project_Y
        # import pdb; pdb.set_trace()
        # loss = lam*Project_S / (1-lam)*Project_Y

        return  loss, Project_S, Project_Y

class Projection_gauss:
    def __init__(self):
        # self.eye = torch.eye(1).to(device)
        pass

    def __call__(self, Z, S, Y, lam, reg, device, sigma):
        # import pdb; pdb.set_trace()

        # M = Z - torch.mean(Z, dim=0) # Z is already transposed
        Z = Z / torch.norm(Z, 'fro')
        # import pdb;
        # pdb.set_trace()
        # Z = Z / torch.max(Z)
        batch_size = Z.size(0)
        ONES = torch.ones([1,batch_size]).to(device)
        NORM = torch.norm(Z,dim=1).reshape([1,batch_size])
        NORM = NORM ** 2
        K = torch.exp((-torch.mm(torch.t(NORM), ONES)-torch.mm(torch.t(ONES), NORM)+2*torch.mm(Z, torch.t(Z)))/sigma) # Z is already transposed
        # import pdb; pdb.set_trace()

        # K = torch.matrix_power(K+c*torch.eye(K.shape[0]).to(device),d)
        # K = torch.pow(K+c,d)
        # M = M/torch.norm(M,'fro')



        D = torch.eye(batch_size) - torch.ones([batch_size, batch_size])/batch_size
        D = D.to(device)
        M = torch.mm(torch.mm(D,K), D)

        # import pdb; pdb.set_trace()

        P1 = torch.mm(torch.t(M), M)
        # self.eye.resize_(P1.shape[0])

        # import pdb; pdb.set_trace()
        P2 = torch.inverse(P1+ reg*torch.eye(P1.shape[0]).to(device))
        # P2 = torch.inverse(P1+ reg*self.eye)
        P3 = torch.mm(P2, torch.t(M))
        P_M = torch.mm(M, P3)

        # U, Sigma, V = torch.svd(M)
        # import pdb; pdb.set_trace()
        # P_M = torch.mm(U, torch.t(U))

        S_bar = S - torch.mean(S, dim=0)
        Project_S = (torch.norm(torch.mm(P_M, S_bar)) ** 2 + reg*torch.norm(torch.mm(P3, S_bar)) ** 2)/torch.norm(S_bar) ** 2
        # Project_S = torch.norm(torch.mm(P_M, S_bar)) ** 2 /torch.norm(S_bar) ** 2
        # Project_S = (torch.norm(torch.mm(P_M, S_bar)) ** 2 )/batch_size

        Y_bar = Y - torch.mean(Y, dim=0)
        Project_Y = (torch.norm(torch.mm(P_M, Y_bar)) ** 2 + reg*torch.norm(torch.mm(P3, Y_bar)) ** 2)/torch.norm(Y_bar) ** 2
        # Project_Y = torch.norm(torch.mm(P_M, Y_bar)) ** 2 /torch.norm(Y_bar) ** 2
        # Project_Y = (torch.norm(torch.mm(P_M, Y_bar)) ** 2)/batch_size

        loss = lam*Project_S - (1-lam)*Project_Y
        # import pdb; pdb.set_trace()
        # loss = lam*Project_S / (1-lam)*Project_Y

        return  loss, Project_S, Project_Y

class Projection_gauss_linear:
    def __init__(self):
        # self.eye = torch.eye(1).to(device)
        pass

    def __call__(self, Z, S, Y, lam, reg, device, sigma):
        # import pdb; pdb.set_trace()

        # M = Z - torch.mean(Z, dim=0) # Z is already transposed
        # Z = Z / torch.norm(Z, 'fro')
        batch_size = Z.size(0)
        ONES = torch.ones([1,batch_size]).to(device)
        NORM = torch.norm(Z,dim=1).reshape([1,batch_size])
        K = torch.exp((-torch.mm(torch.t(NORM), ONES)-torch.mm(torch.t(ONES), NORM)+2*torch.mm(Z, torch.t(Z)))/sigma) # Z is already transposed

        D = torch.eye(batch_size) - torch.ones([batch_size, batch_size])/batch_size
        D = D.to(device)
        M = torch.mm(torch.mm(D,K), D)

        P1 = torch.mm(torch.t(M), M)

        P2 = torch.inverse(P1+ reg*torch.eye(P1.shape[0]).to(device))
        # P2 = torch.inverse(P1+ reg*self.eye)
        P3 = torch.mm(P2, torch.t(M))
        P_M = torch.mm(M, P3)

        S_bar = S - torch.mean(S, dim=0)
        Project_S = (torch.norm(torch.mm(P_M, S_bar)) ** 2 + reg*torch.norm(torch.mm(P3, S_bar)) ** 2)/torch.norm(S_bar) ** 2
        # Project_S = (torch.norm(torch.mm(P_M, S_bar)) ** 2 )/batch_size

################################################ Linear
        M1 = Z - torch.mean(Z, dim=0)  # Z is already transposed
        # import pdb; pdb.set_trace()
        # M = M/torch.norm(M,'fro')

        Q1 = torch.mm(torch.t(M1), M1)
        # self.eye.resize_(P1.shape[0])

        Q2 = torch.inverse(Q1 + reg * torch.eye(Q1.shape[0]).to(device))
        # P2 = torch.inverse(P1+ reg*self.eye)
        Q3 = torch.mm(Q2, torch.t(M1))
        P_M1 = torch.mm(M1, Q3)

        Y_bar = Y - torch.mean(Y, dim=0)
        Project_Y = (torch.norm(torch.mm(P_M1, Y_bar)) ** 2 + reg*torch.norm(torch.mm(Q3, Y_bar)) ** 2)/torch.norm(Y_bar) ** 2
        # Project_Y = (torch.norm(torch.mm(P_M, Y_bar)) ** 2)/batch_size
##################################################

        loss = lam*Project_S - (1-lam)*Project_Y
        # import pdb; pdb.set_trace()
        # loss = lam*Project_S / (1-lam)*Project_Y

        return  loss, Project_S, Project_Y