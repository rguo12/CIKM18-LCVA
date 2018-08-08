import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import coo_matrix,load_npz

class KNN_IHDP(object):
    def __init__(self, path_data="datasets/Amazon", replications=10, neighbors=5, metric="mahalanobis", mode = "connectivity"):
        self.path_data = path_data
        self.replications = replications # how many files
        # which features are binary
        # leave this list empty for the amazon dataset
        self.binfeats = []
        # which features are continuous

        self.contfeats = [i for i in range(25) if i not in self.binfeats]

        # do not need all three of them  for the amazon dataset
        self.neighbors = neighbors
        self.metric = metric
        self.mode = mode

    # def __iter__(self):
    #     for i in xrange(self.replications):
    #         data = np.loadtxt(self.path_data + '/ihdp_npci_' + str(i + 1) + '.csv', delimiter=',')
    #         t, y, y_cf = data[:, 0], data[:, 1][:, np.newaxis], data[:, 2][:, np.newaxis]
    #         mu_0, mu_1, x = data[:, 3][:, np.newaxis], data[:, 4][:, np.newaxis], data[:, 5:]
    #         yield (x, t, y), (y_cf, mu_0, mu_1)

    def get_train_valid_test(self):
        for i in range(self.replications):
            data = np.loadtxt(self.path_data + '/AmazonItmFeatures_pos.csv', delimiter=',')
            t, y, y_cf = data[:, 0][:, np.newaxis], data[:, 1][:, np.newaxis], data[:, 2][:, np.newaxis]
            mu_0, mu_1, x = data[:, 3][:, np.newaxis], data[:, 4][:, np.newaxis], data[:, 5:]
            # this binary feature is in {1, 2}
            x[:, 13] -= 1

            # for the amazon dataset, comment out the KNN
            # read the graph file as the adjacency matrix A
            A = load_npz(self.path_data + '/new_product_graph_pos.npz')

            print(A)

            #sample pairs from the KNN graph or the given graph for amazon dataset to train the model
            #replace cA with your coo_matrix
            
            cA = coo_matrix(A)
            nodei_list = []
            nodej_list = []

            for i,j,v in zip(cA.row, cA.col, cA.data):
                if i<j:
                    # only need to consider each pair once
                    nodei_list.append(i)
                    nodej_list.append(j)

            #TODO: negative samples

            nodei_idx = np.asarray(nodei_list)
            nodej_idx = np.asarray(nodej_list)

            #print(nodei_idx.shape,nodej_idx.shape)

            idxtrain, ite = train_test_split(np.arange(nodei_idx.shape[0]), test_size=0.1, random_state=1)
            itr, iva = train_test_split(idxtrain, test_size=0.3, random_state=1)

            train_i = (nodei_idx[itr], x[nodei_idx[itr]], t[nodei_idx[itr]], y[nodei_idx[itr]]), (y_cf[nodei_idx[itr]], mu_0[nodei_idx[itr]], mu_1[nodei_idx[itr]])
            train_j = (nodej_idx[itr], x[nodej_idx[itr]], t[nodej_idx[itr]], y[nodej_idx[itr]]), (y_cf[nodej_idx[itr]], mu_0[nodej_idx[itr]], mu_1[nodej_idx[itr]])

            valid_i = (nodei_idx[iva],x[nodei_idx[iva]], t[nodei_idx[iva]], y[nodei_idx[iva]]), (y_cf[nodei_idx[iva]], mu_0[nodei_idx[iva]], mu_1[nodei_idx[iva]])
            valid_j = (nodej_idx[iva],x[nodej_idx[iva]], t[nodej_idx[iva]], y[nodej_idx[iva]]), (y_cf[nodej_idx[iva]], mu_0[nodej_idx[iva]], mu_1[nodej_idx[iva]])

            test_i = (nodei_idx[ite],x[nodei_idx[ite]], t[nodei_idx[ite]], y[nodei_idx[ite]]), (y_cf[nodei_idx[ite]], mu_0[nodei_idx[ite]], mu_1[nodei_idx[ite]])
            test_j = (nodej_idx[ite],x[nodej_idx[ite]], t[nodej_idx[ite]], y[nodej_idx[ite]]), (y_cf[nodej_idx[ite]], mu_0[nodej_idx[ite]], mu_1[nodej_idx[ite]])

            # train = (x[itr], t[itr], y[itr]), (y_cf[itr], mu_0[itr], mu_1[itr])
            # valid = (x[iva], t[iva], y[iva]), (y_cf[iva], mu_0[iva], mu_1[iva])
            # test = (x[ite], t[ite], y[ite]), (y_cf[ite], mu_0[ite], mu_1[ite])
            yield train_i, valid_i, test_i, train_j, valid_j, test_j, self.contfeats, self.binfeats
