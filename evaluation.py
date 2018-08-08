import numpy as np


class Evaluator(object):
    def __init__(self, y, t, y_cf=None, mu0=None, mu1=None):
        self.y = np.squeeze(y)
        self.t = np.squeeze(t)
        self.y_cf = y_cf.reshape(y_cf.shape[0])
        self.mu0 = mu0.reshape(mu0.shape[0])
        self.mu1 = mu1.reshape(mu1.shape[0])
        if mu0 is not None and mu1 is not None:
            self.true_ite = y - y_cf
        self.true_ite = self.true_ite.reshape(self.true_ite.shape[0])

    def rmse_ite(self, ypred1, ypred0):
        pred_ite = np.squeeze(np.zeros_like(self.true_ite))
        idx1, idx0 = np.where(self.t == 1), np.where(self.t == 0)
        #print(idx1)
        ite1, ite0 = self.y[idx1] - ypred0[idx1], ypred1[idx0] - self.y[idx0]
        pred_ite[idx1] = ite1
        pred_ite[idx0] = ite0
        # self.true_ite = self.true_ite.reshape(self.true_ite.shape[0])
        return np.sqrt(np.mean(np.square(self.true_ite - pred_ite)))

    # def abs_ate(self, ypred1, ypred0):
    #     return np.abs(np.mean(ypred1 - ypred0) - np.mean(self.true_ite))

    def abs_ate(self, ypred1,ypred0):
        #new abs_ate considers the special part of our dataset
        #idx for unique elements in y
        y_uni, idx_uni = np.unique(self.y, return_index=True)
        t_uni = self.t[idx_uni]
        y_pred1_uni, y_pred0_uni = ypred1[idx_uni], ypred0[idx_uni]

        #idx for those unique nodes whose treatment is 1
        #idx for those unique nodes whose treatment is 0
        idx_uni_1 = np.where(t_uni==1)
        idx_uni_0 = np.where(t_uni==0)

        #for the amazon dataset, we need to calculate the true ate using the cf

        y_cf_uni = self.y_cf[idx_uni]
        ate_true = (np.sum(y_uni[idx_uni_1] - y_cf_uni[idx_uni_1]) + np.sum(-y_uni[idx_uni_0] + y_cf_uni[idx_uni_0]))/float(y_uni.shape[0])
        ate_pred = (np.sum(y_uni[np.where(t_uni==1)]-y_pred0_uni[np.where(t_uni==1)])+np.sum(-y_uni[np.where(t_uni==0)]+y_pred1_uni[np.where(t_uni==0)]))/float(y_uni.shape[0])
        return np.abs(ate_true-ate_pred)

    # def pehe(self, ypred1, ypred0):
    #
    #     return np.sqrt(np.mean(np.square((self.mu1 - self.mu0) - (ypred1 - ypred0))))

    def pehe(self,ypred1,ypred0):
        #new abs_ate considers the special part of our dataset
        #idx for unique elements in y
        y_uni, idx_uni = np.unique(self.y, return_index=True)
        t_uni = self.t[idx_uni]
        y_pred1_uni, y_pred0_uni = ypred1[idx_uni], ypred0[idx_uni]

        #idx for those unique nodes whose treatment is 1
        #idx for those unique nodes whose treatment is 0
        idx_uni_1 = np.where(t_uni==1)
        idx_uni_0 = np.where(t_uni==0)

        #for the amazon dataset, we need to calculate the true ate using the cf

        y_cf_uni = self.y_cf[idx_uni]

        return np.sqrt(np.sum(np.square((y_uni[idx_uni_1] - y_cf_uni[idx_uni_1]) - (y_uni[idx_uni_1]-y_pred0_uni[idx_uni_1]))) + \
            np.sum(np.square((y_uni[idx_uni_0] - y_cf_uni[idx_uni_0]) - (y_uni[idx_uni_0]-y_pred1_uni[idx_uni_0]))) / float(y_uni.shape[0]))

        #return

    def y_errors(self, y0, y1):
        ypred = (1 - self.t) * y0 + self.t * y1
        ypred_cf = self.t * y0 + (1 - self.t) * y1
        return self.y_errors_pcf(ypred, ypred_cf)

    def y_errors_pcf(self, ypred, ypred_cf):
        rmse_factual = np.sqrt(np.mean(np.square(ypred - self.y)))
        rmse_cfactual = np.sqrt(np.mean(np.square(ypred_cf - self.y_cf)))
        return rmse_factual, rmse_cfactual

    def calc_stats(self, ypred1, ypred0):
        ite = self.rmse_ite(ypred1, ypred0)
        ate = self.abs_ate(ypred1, ypred0)
        pehe = self.pehe(ypred1, ypred0)
        return ite, ate, pehe, np.shape(ypred0)[0]

