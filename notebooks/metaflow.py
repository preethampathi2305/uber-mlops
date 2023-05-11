from metaflow import FlowSpec, step

class Uber(FlowSpec):

    @step
    def start(self):
        
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split

        uber = pd.read_csv('uber_cleaned.csv')
        uber.drop('Unnamed: 0', axis=1, inplace=True)

        X = uber.drop(['fare_amount','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'], axis=1)
        y = uber['fare_amount']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=101)

        print("Data loaded successfully")
        self.next(self.train_lr, self.train_rr, self.train_gr, self.train_ar)

    @step
    def train_lr(self):
        from sklearn.linear_model import LinearRegression

        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)
        self.next(self.choose_model)

    @step
    def train_rr(self):
        from sklearn.ensemble import RandomForestRegressor

        self.model = RandomForestRegressor()
        self.model.fit(self.X_train, self.y_train)
        self.next(self.choose_model)

    @step
    def train_gr(self):
        from sklearn.ensemble import GradientBoostingRegressor

        self.model = GradientBoostingRegressor()
        self.model.fit(self.X_train, self.y_train)
        self.next(self.choose_model)

    @step
    def train_ar(self):
        from sklearn.ensemble import AdaBoostRegressor

        self.model = AdaBoostRegressor()
        self.model.fit(self.X_train, self.y_train)
        self.next(self.choose_model)

    @step
    def choose_model(self, inputs):
        def score(inp):
            return inp.model, inp.model.score(inp.X_test, inp.y_test)

        self.results = sorted(map(score, inputs), key=lambda x: -x[1])
        self.model = self.results[0][0]
        self.next(self.end)

    @step
    def end(self):
        print('Scores:')
        print('\n'.join('%s %f' % res for res in self.results))
        print('Model:', self.model)


if __name__=='__main__':
    Uber()