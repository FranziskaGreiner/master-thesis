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

ADF Statistic: -12.319783117952014
p-value: 6.792049207828507e-23
null hypothesis rejected --> data is stationary
