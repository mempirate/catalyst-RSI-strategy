import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from logbook import Logger

from catalyst import run_algorithm
from catalyst.api import order_target_percent, record, symbol
from catalyst.exchange.utils.stats_utils import extract_transactions

NAMESPACE = "RSI_strategy"
log = Logger(NAMESPACE)

def initialize(context):

    context.asset = symbol("btc_usd")
    context.i = 0

    context.base_price = None

def handle_data(context, data):

    RSI_periods = 14

    context.i += 1
    if context.i < RSI_periods:
        return

    RSI_data = data.history(context.asset,
                            "price",
                            bar_count = RSI_periods,
                            frequency = "30T")

    # compute RSI
    oversold = 30
    overbought = 70

    deltas = RSI_data.diff()
    seed = deltas[:RSI_periods + 1]
    up = seed[seed >= 0].sum() / RSI_periods
    down = -seed[seed < 0].sum() / RSI_periods

    RS = up / down
    RSI = 100 - (100 / (1 + RS))

    # get current price
    price = data.current(context.asset, "price")

    if context.base_price == None:
        context.base_price = price

    price_change = (price - context.base_price) / context.base_price

    record(price = price,
           cash = context.portfolio.cash,
           price_change = price_change,
           RSI = RSI)

    orders = context.blotter.open_orders
    if len(orders) > 0:
        return

    if not data.can_trade(context.asset):
        print("Cannot trade right now")
        return

    pos_amount = context.portfolio.positions[context.asset].amount

    # strategy logic
    if pos_amount == 0 and RSI <= oversold:
        order_target_percent(context.asset, 1)

    elif pos_amount < 0 and RSI <= 40:
        order_target_percent(context.asset, 0)

    elif pos_amount == 0 and RSI >= overbought:
        order_target_percent(context.asset, -1)

    elif pos_amount > 0 and RSI >= 60:
        order_target_percent(context.asset, 0)


def analyze(context, perf):
    # get the quote_currency that was passed as a parameter to the simulation
    exchange = list(context.exchanges.values())[0]
    quote_currency = exchange.base_currency.upper()

    # first chart: portfolio value
    ax1 = plt.subplot(411)
    perf.loc[:, ['portfolio_value']].plot(ax=ax1)
    ax1.legend_.remove()
    ax1.set_ylabel('Portfolio Value\n({})'.format(quote_currency))
    start, end = ax1.get_ylim()
    ax1.yaxis.set_ticks(np.arange(start, end, 30))

    # second chart: asset price, buys & sells
    ax2 = plt.subplot(412, sharex=ax1)
    perf.loc[:, ['price']].plot(ax = ax2, label = 'Price')
    ax2.legend_.remove()
    ax2.set_ylabel('{asset}\n({quote})'.format(asset = context.asset.symbol, quote = quote_currency
    ))
    start, end = ax2.get_ylim()
    ax2.yaxis.set_ticks(np.arange(start, end, 300))

    transaction_df = extract_transactions(perf)
    if not transaction_df.empty:
        buy_df = transaction_df[transaction_df['amount'] > 0]
        sell_df = transaction_df[transaction_df['amount'] < 0]
        ax2.scatter(
            buy_df.index.to_pydatetime(),
            perf.loc[buy_df.index, 'price'],
            marker = '^',
            s = 50,
            c = 'green',
            label = ''
        )
        ax2.scatter(
            sell_df.index.to_pydatetime(),
            perf.loc[sell_df.index, 'price'],
            marker = 'v',
            s = 50,
            c = 'red',
            label = ''
        )

    # third chart: relative strength index
    ax3 = plt.subplot(413, sharex = ax1)
    perf.loc[:, ['RSI']].plot(ax = ax3)
    plt.axhline(y = 30, linestyle = 'dotted', color = 'grey')
    plt.axhline(y = 70, linestyle = 'dotted', color = 'grey')
    ax3.legend_.remove()
    ax3.set_ylabel('RSI')
    start, end = ax3.get_ylim()
    ax3.yaxis.set_ticks(np.arange(0, 100, 10))

    # fourth chart: percentage return of the algorithm vs holding
    ax4 = plt.subplot(414, sharex=ax1)
    perf.loc[:, ['algorithm_period_return', 'price_change']].plot(ax=ax4)
    ax4.legend_.remove()
    ax4.set_ylabel('Percent Change')
    start, end = ax4.get_ylim()
    ax4.yaxis.set_ticks(np.arange(start, end, (end - start) / 5))

    plt.show()

if __name__ == "__main__":
    run_algorithm(
            capital_base = 1000,
            data_frequency = "minute",
            initialize = initialize,
            handle_data = handle_data,
            analyze = analyze,
            exchange_name = "bitfinex",
            algo_namespace = NAMESPACE,
            base_currency= "usd",
            start = pd.to_datetime("2018-5-10", utc = True),
            end = pd.to_datetime("2018-5-14", utc = True)
    )
