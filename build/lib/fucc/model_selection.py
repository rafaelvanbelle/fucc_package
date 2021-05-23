#from sklearn.model_selection._split import _BaseKFold
from sklearn.utils import indexable
from sklearn.utils.validation import _deprecate_positional_args, _num_samples
import datetime 
import numpy as np
__all__ = ['RollingWindowSplit']

class RollingWindowSplit():
    
    @_deprecate_positional_args
    def __init__(self, train_size=None, test_size=None, start_date=None):
        
                  
        self.train_size = train_size
        self.test_size = test_size
        self.start_date = start_date

        # If these are specified, check they belong to the right type. 
        if (train_size is not None) & (not isinstance(train_size, datetime.timedelta)):
            raise TypeError('train_size should be datetime.timedelta')
        if (test_size is not None ) & (not isinstance(test_size, datetime.timedelta)):
            raise TypeError('test_size should be datetime.timedelta')


    def split(self, X, y=None, timestamps=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        timestamps : array-like of shape (n_samples,)
            containing timestamp associated with every sample.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """

        X, y, timestamps = indexable(X, y, timestamps)
        n_samples = _num_samples(X)


        if (self.train_size is not None) & (self.test_size is not None):
            
            indices = np.arange(n_samples)
            if self.start_date is not None:
                if self.start_date < timestamps[0] + self.train_size:
                    raise ValueError(
                    "The provided start_date is too close to the start of the datasets to allow for sufficient training data.")
                
                start_time = self.start_date
            else:
                start_time = timestamps[0] + self.train_size
            
            end_time = timestamps[-1]
            
            while start_time + self.test_size <= end_time:
                yield (indices[(timestamps > start_time - self.train_size) & (timestamps <= start_time)], 
                    indices[(timestamps > start_time) & (timestamps <= (start_time + self.test_size))])
                # Update start_time
                start_time += self.test_size