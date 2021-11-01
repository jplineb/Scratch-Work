from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn import svm
from sklearn.linear_model import ElasticNet
from sklearn.feature_selection import SelectFromModel

class ElasticSVC(BaseEstimator, ClassifierMixin):

    def __init__(self, random_state=None,
                 alpha=1, l1_ratio=0.5, selection="cyclic", max_iterations=1000, fit_intercept=True,
                 threshold=None, max_features = None,
                 kernel='rbf', svm_C=1000, gamma='auto'):

        self.random_state = random_state
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.selection = selection
        self.max_iterations = max_iterations
        self.fit_intercept = fit_intercept
        self.threshold = threshold
        self.max_features = max_features
        self.kernel = kernel
        self.svm_C = svm_C
        self.gamma = gamma
    
    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X,y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        # Initialize feature model
        self.feature_model = ElasticNet(alpha = self.alpha,
                                        l1_ratio = self.l1_ratio,
                                        selection = self.selection,
                                        max_iter = self.max_iterations,
                                        fit_intercept = self.fit_intercept,
                                        random_state = self.random_state)
        
        # Transform data using feature_selc
        self.selector = SelectFromModel(self.feature_model,
                                        max_features=self.max_features).fit(X,y)
        
        ## Metrics for the feature selection
        self.features_selected = self.selector.get_feature_names_out()
        self.num_features = len(self.features_selected)
        ## Transform X to X_new
        X_new = self.selector.transform(X)
        
        # Initialize SVM
        self.pred_model = svm.SVC(C = self.svm_C,
                                  kernel = self.kernel,
                                  gamma = self.gamma,
                                  random_state = self.random_state)

        ## Fit Prediction model
        self.pred_model.fit(X_new, y)

        return self
    
    def predict(self, X):
        # Check if fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        # Transform data for comparison
        X_new = self.selector.transform(X)

        # label for sample
        final_pred = self.pred_model.predict(X_new)

        return(final_pred)