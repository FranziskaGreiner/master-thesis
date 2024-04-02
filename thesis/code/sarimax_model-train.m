p = 2, d = 0, q = 0, P = 1, D = 0, Q = 0, s = 12

sarimax_model = SARIMAX(train_data['moer'], exog=exog_train, order=(p, d, q), seasonal_order=(P, D, Q, s))
results = sarimax_model.fit()
