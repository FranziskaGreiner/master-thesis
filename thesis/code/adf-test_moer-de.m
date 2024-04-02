# ADF test
# H0: data is non stationary
# H1: data is stationary

result = adfuller(moer_de['moer'])

adf_statistic = result[0]
p_value = result[1]
print(f'ADF Statistic: {adf_statistic}')
print(f'p-value: {p_value}')

if p_value < 0.05:
    print('null hypothesis rejected --> data is stationary')
else:
    print('Failed to reject null hypothesis --> data is not stationary')

ADF Statistic: -12.99035452395651
p-value: 2.82098093226533e-24
null hypothesis rejected --> data is stationary
