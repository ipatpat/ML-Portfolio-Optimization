import mean_variance_optimization as mv
import machine_learning_strategies as mls
import black_litterman_model as bl
import portfolio_statistics as ps
import factor_analysis as fa   # For future use
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns


# Define current user portfolio
portfolio = {
    '600000.SS': 1000.0,  # 浦发银行
    '600009.SS': 1000.0,  # 上海机场
    '600016.SS': 1000.0,  # 民生银行
    '600028.SS': 1000.0,  # 中国石化
    '600030.SS': 1000.0,  # 中信证券
    '600031.SS': 1000.0,  # 三一重工
    '600036.SS': 1000.0,  # 招商银行
    '600048.SS': 1000.0,  # 保利地产
    '600050.SS': 1000.0,  # 中国联通
    '600104.SS': 1000.0,  # 上汽集团
    '600196.SS': 1000.0,  # 复星医药
    '600276.SS': 1000.0,  # 恒瑞医药
    '600309.SS': 1000.0,  # 万华化学
    '600340.SS': 1000.0,  # 华夏幸福
    '600519.SS': 1000.0,  # 贵州茅台
    '600547.SS': 1000.0,  # 山东黄金
    '600585.SS': 1000.0,  # 海螺水泥
    '600690.SS': 1000.0,  # 青岛海尔
    '600703.SS': 1000.0,  # 三安光电
    '600837.SS': 1000.0,  # 海通证券
    '600887.SS': 1000.0,  # 伊利股份
    '601012.SS': 1000.0,  # 隆基股份
    '601066.SS': 1000.0,  # 中信建投
    '601088.SS': 1000.0,  # 中国神华
    '601111.SS': 1000.0,  # 中国国航
    '601138.SS': 1000.0,  # 工业富联
    '601166.SS': 1000.0,  # 兴业银行
    '601186.SS': 1000.0,  # 中国铁建
    '601211.SS': 1000.0,  # 国泰君安
    '601236.SS': 1000.0,  # 红塔证券
    '601288.SS': 1000.0,  # 农业银行
    '601318.SS': 1000.0,  # 中国平安
    '601319.SS': 1000.0,  # 中国人保
    '601328.SS': 1000.0,  # 交通银行
    '601336.SS': 1000.0,  # 新华保险
    '601390.SS': 1000.0,  # 中国中铁
    '601398.SS': 1000.0,  # 工商银行
    '601601.SS': 1000.0,  # 中国太保
    '601628.SS': 1000.0,  # 中国人寿
    '601668.SS': 1000.0,  # 中国建筑
    '601688.SS': 1000.0,  # 华泰证券
    '601766.SS': 1000.0,  # 中国中车
    '601818.SS': 1000.0,  # 光大银行
    '601857.SS': 1000.0,  # 中国石油
    '601888.SS': 1000.0,  # 中国国旅
    '601939.SS': 1000.0,  # 建设银行
    '601988.SS': 1000.0,  # 中国银行
    '601989.SS': 1000.0,  # 中国重工
    '603259.SS': 1000.0,  # 药明康德
    '603993.SS': 1000.0,  # 洛阳钼业
}
# Define market representation
market_representation = ['510050.SS']
#market_representation = ['000001.SS']


# Define a dictionary for storing weights of portfolios
portfolio_weights = {}

# Define dates for training and backtesting
training_start_date = '2018-01-01'
training_end_date = '2023-08-30'
backtesting_start_date = training_end_date
backtesting_end_date = '2024-08-30'
risk_free_rate = 0.04

# Define risk sensitivity for Mean-Variance Optimization
max_volatility = 0.25

# Define minimum and maximum asset weights for Mean-Variance Optimization
min_weight = .01
max_weight = .25

# Perform Mean-Variance Optimization
tickers, weights = mv.calculate_weights(portfolio)
optimized_weights_mv = mv.mean_variance_optimization(tickers, training_start_date, training_end_date, max_volatility, min_weight=min_weight, max_weight=max_weight)

# Begin ML Training on stock ticker data for Black Litterman Model
investor_views = {}
view_confidences = {}

for ticker in tickers:
    investor_views[ticker], view_confidences[ticker] = mls.generate_investor_views(ticker, training_start_date, training_end_date, model_type='Gradient Boosting')

market_caps = bl.get_market_caps(tickers)
index_data = mv.download_stock_data(market_representation, training_start_date, training_end_date)
index_return = (index_data['Adj Close'].iloc[-1] / index_data['Adj Close'].iloc[0]) - 1

# Calculate market returns for each asset
market_returns = bl.get_market_returns(market_caps, index_return)

historical_data = mv.download_stock_data(tickers, training_start_date, training_end_date)
predicted_returns = bl.black_litterman_adjustment(market_returns, investor_views, view_confidences, historical_data)

# Map adjusted returns to tickers
predicted_returns = dict(zip(tickers, predicted_returns))

# Convert adjusted returns to the format expected by the optimization function
adjusted_returns_vector = np.array([predicted_returns[ticker] for ticker in tickers])

# Perform mean-variance optimization with generated predicted returns
optimized_weights_ml_mv = mv.mean_variance_optimization(tickers, training_start_date, training_end_date, max_volatility, adjusted_returns_vector, min_weight, max_weight)

# Download market data for backtesting and calculate performance of each asset
historical_data_backtest = mv.download_stock_data(tickers, backtesting_start_date, backtesting_end_date)
daily_returns_backtest = historical_data_backtest['Adj Close'].pct_change()

# Calculate the cumulative performance of the machine learning mean variance optimized portfolio
portfolio_returns_ml_mv = daily_returns_backtest.dot(optimized_weights_ml_mv)
cumulative_returns_ml_mv = (1 + portfolio_returns_ml_mv).cumprod()

# Calculate cumulative returns for the first mean variance optimized portfolio
portfolio_returns_mv = daily_returns_backtest.dot(optimized_weights_mv)
cumulative_returns_mv = (1 + portfolio_returns_mv).cumprod()

# Download and calculate market index cumulative returns
market_data = mv.download_stock_data(market_representation, backtesting_start_date, backtesting_end_date)['Adj Close']
market_returns = market_data.pct_change()
cumulative_market_returns = (1 + market_returns).cumprod()

# Calculate cumulative returns for the unoptimized original portfolio
portfolio_returns_unoptimized = daily_returns_backtest.dot(weights)
cumulative_returns_unoptimized = (1 + portfolio_returns_unoptimized).cumprod()

# Convert weights to percentages with 2 decimal places for formatting
weights_pct = [f'{weight * 100:.2f}%' for weight in weights]
optimized_weights_pct = [f'{weight * 100:.2f}%' for weight in optimized_weights_mv]
optimized_weights_with_adjusted_returns_pct = [f'{weight * 100:.2f}%' for weight in optimized_weights_ml_mv]

# Create a DataFrame and output it to show comparison between portfolio weights
portfolio_comparison = pd.DataFrame({'1/N': weights_pct,'MV Optimization': optimized_weights_pct, 'ML MV Optimization': optimized_weights_with_adjusted_returns_pct}, index=tickers)
print(portfolio_comparison)

portfolio_returns_ml_mv = portfolio_returns_ml_mv.squeeze()
portfolio_returns_mv = portfolio_returns_mv.squeeze()
portfolio_returns_unoptimized = portfolio_returns_unoptimized.squeeze()
market_returns = market_returns.squeeze()

# Calculate statistics for ML MV optimized portfolio
sharpe_ratio_ml_mv = ps.sharpe_ratio(portfolio_returns_ml_mv, risk_free_rate)
#print('sharpe_ratio_ml_mv:', sharpe_ratio_ml_mv)
sortino_ratio_ml_mv = ps.sortino_ratio(portfolio_returns_ml_mv, risk_free_rate)
# print("info_ratio_ml_mv_debug:", portfolio_returns_ml_mv, market_returns)
info_ratio_ml_mv = ps.information_ratio(portfolio_returns_ml_mv, market_returns)
#print(info_ratio_ml_mv)

# Calculate statistics for MV optimized portfolio
sharpe_ratio_mv = ps.sharpe_ratio(portfolio_returns_mv, risk_free_rate)
sortino_ratio_mv = ps.sortino_ratio(portfolio_returns_mv, risk_free_rate)
info_ratio_mv = ps.information_ratio(portfolio_returns_mv, market_returns)

# Calculate statistics for original unoptimized portfolio
sharpe_ratio_unoptimized = ps.sharpe_ratio(portfolio_returns_unoptimized, risk_free_rate)
sortino_ratio_unoptimized = ps.sortino_ratio(portfolio_returns_unoptimized, risk_free_rate)
info_ratio_unoptimized = ps.information_ratio(portfolio_returns_unoptimized, market_returns)

# Calculate statistics for the market representation
sharpe_ratio_market = ps.sharpe_ratio(market_returns, risk_free_rate)
#print('sharpe_ratio_market:', sharpe_ratio_market)
sortino_ratio_market = ps.sortino_ratio(market_returns, risk_free_rate)
info_ratio_market = ps.information_ratio(market_returns, market_returns)

# Basic Plot Setup
plt.figure(figsize=(16, 9), constrained_layout=True)
ax = plt.gca()

sns.set_palette("bright")  # You can choose any palette like "deep", "muted", "bright", etc.
colors = sns.color_palette()

background_color = 'white'
text_color = 'black'
grid_color = 'lightgray'

# Set plot aesthetics for readability
plt.gcf().set_facecolor(background_color)
# ax.set_facecolor('white')
# ax.xaxis.label.set_color('black')
# ax.yaxis.label.set_color('black')
# ax.tick_params(axis='x', colors='black')
# ax.tick_params(axis='y', colors='black')
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.2f}%'.format(y)))

# ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{100 * y:.2f}%'))
for spine in ax.spines.values():
    spine.set_edgecolor('white')

# Convert cumulative returns to percentage gain
cumulative_returns_ml_mv_percent = (cumulative_returns_ml_mv - 1) * 100
cumulative_returns_mv_percent = (cumulative_returns_mv - 1) * 100
cumulative_returns_unoptimized_percent = (cumulative_returns_unoptimized - 1) * 100
cumulative_market_returns_percent = (cumulative_market_returns - 1) * 100
# 调试代码，打印列表长度和内容
#print('cumulative_market_returns_percent length:', len(cumulative_market_returns_percent))
#print('cumulative_market_returns_percent:', cumulative_market_returns_percent)


final_returns_ml_mv = cumulative_returns_ml_mv_percent[-1]
#print('final_returns_ml_mv:', final_returns_ml_mv)
#print('cumulative_returns_ml_mv_percent length:', cumulative_returns_ml_mv_percent)
final_returns_mv = cumulative_returns_mv_percent[-1]
final_returns_unoptimized = cumulative_returns_unoptimized_percent[-1]
# 打印最后一个元素
#print('cumulative_returns_unoptimized_percent:', cumulative_market_returns_percent)
#print("cumulative_returns_unoptimized_percent:", cumulative_market_returns_percent.shape)
#get the final element of cumulative_market_returns_percent
#cumulative_returns_unoptimized_percent.shape: (1257, 1)
#suqqze the array to 1D and get the last element
final_returns_market = cumulative_market_returns_percent.squeeze()[-1]

# Plot lines representing percentage gain returns
plt.plot(cumulative_returns_ml_mv_percent, label='Portfolio Optimized with ML and MV', color=colors[0])
plt.plot(cumulative_returns_mv_percent, label='Portfolio Optimized with MV', color=colors[1])
plt.plot(cumulative_market_returns_percent, label='SSE 50 Index', color=colors[2])
plt.plot(cumulative_returns_unoptimized_percent, label='1/N Unoptimized Portfolio', color=colors[3])



# Generate box for ML MV Optimized Portfolio
# print("sharpe_ratio_ml_mv",sharpe_ratio_ml_mv)
# print("sortino_ratio_ml_mv",sortino_ratio_ml_mv)
# print("info_ratio_ml_mv",info_ratio_ml_mv)
# print("final_returns_ml_mv",final_returns_ml_mv)

stats_text_ml_mv = f"ML & MV Optimized Portfolio:\nSharpe Ratio: {sharpe_ratio_ml_mv:.2f}\nSortino Ratio: {sortino_ratio_ml_mv:.2f}\nInfo Ratio: {info_ratio_ml_mv:.2f}\nReturn: {final_returns_ml_mv:.2f}%"
plt.text(x=0.0655, y=0.77, s=stats_text_ml_mv, transform=plt.gcf().transFigure, fontsize=10, color=text_color, bbox=dict(boxstyle="round,pad=0.3", edgecolor=colors[0], facecolor=background_color))

# Generate box for MV Optimized Portfolio
stats_text_mv = f"MV Optimized Portfolio:\nSharpe Ratio: {sharpe_ratio_mv:.2f}\nSortino Ratio: {sortino_ratio_mv:.2f}\nInfo Ratio: {info_ratio_mv:.2f}\nReturn: {final_returns_mv:.2f}%"
plt.text(x=0.0655, y=0.67, s=stats_text_mv, transform=plt.gcf().transFigure, fontsize=10, color=text_color, bbox=dict(boxstyle="round,pad=0.3", edgecolor=colors[1], facecolor=background_color))

# Generate box for Unoptimized Portfolio
stats_text_unoptimized = f"Market ({market_representation[0]} INDEX):\nSharpe Ratio: {sharpe_ratio_market:.2f}\nSortino Ratio: {sortino_ratio_market:.2f}\nInfo Ratio: {info_ratio_market:.2f}\nReturn: {final_returns_market:.2f}%"
plt.text(x=0.0655, y=0.57, s=stats_text_unoptimized, transform=plt.gcf().transFigure, fontsize=10, color=text_color, bbox=dict(boxstyle="round,pad=0.3", edgecolor=colors[2], facecolor=background_color))

# Generate box for market
stats_text_market = f"Unoptimized Portfolio\nSharpe Ratio: {sharpe_ratio_unoptimized:.2f}\nSortino Ratio: {sortino_ratio_unoptimized:.2f}\nInfo Ratio: {info_ratio_unoptimized:.2f}\nReturn: {final_returns_unoptimized:.2f}%"
plt.text(x=0.0655, y=0.47, s=stats_text_market, transform=plt.gcf().transFigure, fontsize=10, color=text_color, bbox=dict(boxstyle="round,pad=0.3", edgecolor=colors[3], facecolor=background_color))


plt.title('Comparative Cumulative Returns', color=text_color)
plt.xlabel('Date', color=text_color)
plt.ylabel('Percentage Gain (%)', color=text_color)

plt.legend(loc='best', facecolor=background_color, edgecolor=text_color, labelcolor=text_color)
plt.grid(True)
plt.show()
