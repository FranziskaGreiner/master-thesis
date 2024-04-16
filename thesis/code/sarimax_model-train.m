sarimax_model = SARIMAX(train_data['moer'], exog=exog_train, order=(p, d, q), seasonal_order=(P, D, Q, s))
results = sarimax_model.fit()
