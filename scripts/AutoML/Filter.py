from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class AbstractBaseFilter(ABC):
    @abstractmethod
    def transform(self, data):
        pass

    @abstractmethod
    def inverse_transform(self, data):
        pass


class AbstractBaseFilterChain(ABC):
    @abstractmethod
    def add_filters(self, filter):
        pass

    @abstractmethod
    def transform(self, data):
        pass

    @abstractmethod
    def inverse_transforme(self, data):
        pass


class BaseFilter(AbstractBaseFilter):
    def transform(self, data):
        raise NotImplementedError('subclass must implement transform method')

    def inverse_transform(self, data):
        raise NotImplementedError('subclass must implement inverse_transform method')


class FilterChain(AbstractBaseFilterChain):
    def __init__(self, filters):
        self.filters = filters

    def add_filters(self, filter):
        self.filters.extend(filter)

    def transform(self, data):
        for filter in self.filters:
            data = filter.transform(data)

        return data

    def inverse_transforme(self, data):
        inverse_filter_lst = self.filters[::-1]

        for filter in inverse_filter_lst:
            data = filter.inverse_transform(data)

        return data


class AdjacentDifferenceFilter(BaseFilter):
    def transform(self, data):
        if isinstance(data, pd.DataFrame):
            origin_data = data['col_1'].values
        elif isinstance(data, np.ndarray):
            origin_data = data.copy()

        else:
            raise ValueError("wrong type!")

        diff_data = np.diff(origin_data)
        result = np.insert(diff_data, 0, origin_data[0])
        if isinstance(data, pd.DataFrame):
            for i in range(len(data)):
                data.iloc[i] = result[i]
            return data
        else:
            return result

    def inverse_transform(self, diff_data):
        if isinstance(diff_data, pd.DataFrame):
            origin_diff_data = diff_data['col_1'].values
        elif isinstance(diff_data, np.ndarray):
            origin_diff_data = diff_data.copy()

        else:
            raise ValueError("wrong type!")

        data = np.cumsum(origin_diff_data)
        if isinstance(diff_data, pd.DataFrame):
            for i in range(len(diff_data)):
                diff_data.iloc[i] = data[i]
            return diff_data
        else:
            return data


class StandardScalerFilter(BaseFilter):
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, data):
        self.scaler.fit(data)

    def transform(self, data):
        try:
            processed_data = self.scaler.transform(data)
        except:
            self.scaler.fit(data)
            processed_data = self.scaler.transform(data)
            
        if isinstance(data, pd.DataFrame):
            for i in range(len(data)):
                data.iloc[i] = processed_data[i]
                return data
        elif isinstance(data, np.ndarray):
            return processed_data
        else:
            raise ValueError("wrong type!")

    def inverse_transform(self, data):

        processed_data = self.scaler.inverse_transform(data)
        if isinstance(data, pd.DataFrame):
            for i in range(len(data)):
                data.iloc[i] = processed_data[i]
                return data
        elif isinstance(data, np.ndarray):
            return processed_data
        else:
            raise ValueError("wrong type!")


class PositiveFilter(BaseFilter):
    def transform(self, data):
        data[data<0] = 0
        return data
    def inverse_transform(self, data):
        return data
