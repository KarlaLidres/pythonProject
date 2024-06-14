import pandas as pd

data = {
    "Year": range(2005, 2016),
    "Car_Sales": [988269, 962666, 1049982, 1012165, 937328, 1035574, 1008437, 1112032, 1136227, 1113230, 1155408]
}
df = pd.DataFrame(data)

print(df.isnull().sum())


from sklearn.linear_model import LinearRegression

X = df[['Year']]
y = df['Car_Sales']

model = LinearRegression()
model.fit(X, y)

slope = model.coef_[0]
intercept = model.intercept_

print("Slope:", slope)
print("Intercept:", intercept)
# Predict for the next 3 years
next_years = [[year] for year in range(2016, 2023)]
predicted_sales = model.predict(next_years)

for year, sales in zip(range(2016, 2023), predicted_sales):
    print("Predicted sales for {}: {:.2f}".format(year, sales))
import matplotlib.pyplot as plt

plt.scatter(X, y, color='blue', label='Actual data')

plt.plot(X, model.predict(X), color='red', label='Regression line')

plt.plot(range(2016, 2019), predicted_sales, color='green', linestyle='--', label='Predicted values')

plt.title('Car Sales in Australia')
plt.xlabel('Year')
plt.ylabel('Car Sales')
plt.legend()
plt.grid(True)
plt.show()