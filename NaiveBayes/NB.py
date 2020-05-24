import numpy as np 


class MultinomialNB(object):

    def __init__(self,alpha= 0.1,fit_prior=True,class_prior=None):
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior
        self.classes = None
        self.conditional_prob = None 

    def cal_feature_prob(self,feature):   # calculate the prob of the classes P(yk) = (Nyk + α) / (N + k* α)
        # print("feature =",feature)
        values = np.unique(feature)
        # print("values = ",values)
        total_num = float(len(feature))
        prob_dict = {}
        for v in values:
            prob_dict[v] = ((np.sum(np.equal(feature,v)) + self.alpha)) / (total_num + self.alpha * len(values))
        return prob_dict

    def fit(self,X,y):                    #This is the model function to train the data

        self.classes = np.unique(y)         #self.classes = [-1,1]
        # print("self.classes =",self.classes)
        if self.class_prior == None:
            class_num = len(self.classes)     #y's classes here is 2 : -1 , 1
            if not self.fit_prior :
                self.class_prior = [1.0/class_num for _ in range(class_num)] 
                # print("self.class_prior = ", self.class_prior)
            else:
                self.class_prior = []
                sample_num = float(len(y))
                # print("sample_num = ",sample_num)     #len(y) == 15
                for c in self.classes:
                    c_num = np.sum(np.equal(y,c))
                    self.class_prior.append((c_num+self.alpha)/(sample_num+class_num*self.alpha))   #pro of classes:P(yk)
        # print("self.class_prior = ", self.class_prior)
        # calculate the conditinonal prob example: P(xj|yk) 
        self.conditional_prob = {}  # like { c0:{ x0:{ value0:0.2, value1:0.8 }, x1:{} }, c1:{...} }
        for c in self.classes:
            self.conditional_prob[c] = {}
            for i in range(len(X[0])):  #for each feature
                # print("X[0] = ",X[1])
                feature = X[np.equal(y,c)][:,i]  # feature is x's match -1 and 1 numbers 
                # print('feature = ',feature)   
                self.conditional_prob[c][i] = self.cal_feature_prob(feature)
        # print("self.conditional_prob = ", self.conditional_prob)
        return self
    #given values_prob {value0:0.2,value1:0.1,value3:0.3,.. } and target_value
    #return the probability of target_value
    def _get_xj_prob(self,values_prob,target_value):
        return values_prob[target_value]

    #predict a single sample based on (class_prior,conditional_prob)
    def _predict_single_sample(self,x):
        label = -1
        max_posterior_prob = 0

        #for each category, calculate its posterior probability: class_prior * conditional_prob
        for c_index in range(len(self.classes)):
            current_class_prior = self.class_prior[c_index]
            current_conditional_prob = 1.0
            feature_prob = self.conditional_prob[self.classes[c_index]]
            j = 0
            for feature_i in feature_prob.keys():
                current_conditional_prob *= self._get_xj_prob(feature_prob[feature_i],x[j])
                j += 1
                # print("current_conditional_prob = ",current_conditional_prob)
            #compare posterior probability and update max_posterior_prob, label
            if current_class_prior * current_conditional_prob > max_posterior_prob:
                max_posterior_prob = current_class_prior * current_conditional_prob
                label = self.classes[c_index]
        return label

    #predict samples (also single sample)           
    def predict(self,X):
        #TODO1:check and raise NoFitError 
        #ToDO2:check X
        if X.ndim == 1:
            # print("X.ndim =",X.ndim)
            return self._predict_single_sample(X)
        else:
                #classify each sample   
                labels = []
                for i in range(X.shape[0]):
                        label = self._predict_single_sample(X[i])
                        labels.append(label)
                return labels

if __name__ == "__main__" :
    X = np.array([
                      [1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,3,3,3],
                      [4,5,5,4,4,4,5,5,6,6,6,5,5,6,6,4,5,6]
             ])
X = X.T
y = np.array([-1,-1,1,1,-1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,1])

nb = MultinomialNB(alpha=1.0,fit_prior=True)
nb.fit(X,y)
print (nb.predict(np.array([2,4])))



        